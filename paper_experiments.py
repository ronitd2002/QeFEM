"""One-button experiment orchestration for the QeFEM paper.

This module deliberately imports the validated implementation from qefem_prof.
It does not reimplement the algorithm.  It adds only experiment orchestration,
coherent trajectory capture, reporting, and the directory layout used by the
paper notebook.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch


PAPER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("QEFEM_REPO_ROOT", "/Users/ronit/Desktop/QeFEM/qefem_prof")).resolve()
MAIN_ROOT = Path(os.environ.get("QEFEM_MAIN_ROOT", "/Users/ronit/Desktop/QeFEM/main")).resolve()
RESULTS_ROOT = Path(os.environ.get("QEFEM_RESULTS_ROOT", PAPER_ROOT / "results")).resolve()
REFERENCE_ROOT = MAIN_ROOT / "da_family_reproduction"
EXPECTED_REVISION = "b14f73c53305b55242b418ef2013fba0ac3cce54"
K2000_TARGET = 33337.0

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_ising_benchmarks import (  # noqa: E402
    configurations,
    input_metadata,
    ising_expectation,
    load_instance,
    load_replay_inputs,
    make_trial_inputs,
    parse_args,
    run_fem,
    run_lqa,
    score_trial,
    sha256,
    synchronize,
)
from benchmarks.Gset.da_manual import (  # noqa: E402
    iterate_manual_ising,
    validate_gradient_config,
)
from benchmarks.Gset.run_gset_accuracy import (  # noqa: E402
    derived_seed as gset_derived_seed,
    load_graph,
    make_trial_inputs as make_gset_inputs,
)
from double_annealing import DoubleAnnealing  # noqa: E402
from double_annealing.reproducibility import (  # noqa: E402
    configure_reproducibility,
    runtime_metadata,
    validate_device_dtype,
)
from double_annealing.solver import quantum_binary_state  # noqa: E402


GSET_COHORTS = {
    # The labels follow the benchmark table supplied with the request.
    "random": [*(f"G{i}" for i in range(1, 11)),
               *(f"G{i}" for i in range(22, 32)),
               *(f"G{i}" for i in range(43, 48))],
    "toroidal": [*(f"G{i}" for i in range(11, 14)),
                 *(f"G{i}" for i in range(32, 35)),
                 *(f"G{i}" for i in range(48, 51))],
    "planar": [*(f"G{i}" for i in range(14, 22)),
               *(f"G{i}" for i in range(35, 43)),
               *(f"G{i}" for i in range(51, 55))],
}
GSET_ALL = [f"G{i}" for i in range(1, 55)]
GSET_BIG = [*(f"G{i}" for i in range(55, 68)), "G70", "G72", "G77", "G81"]
GSET_REPRESENTATIVE_15 = frozenset({
    "G10", "G12", "G13", "G18", "G19", "G20", "G21", "G24",
    "G25", "G46", "G47", "G48", "G49", "G50", "G54",
})
K2000_REPLICAS = [128, 512, 1024, 2048, 4096, 8192]


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    dtype: torch.dtype
    dtype_name: str
    threads: int
    revision: str


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def initialize_runtime(
    device: str = "cpu", dtype_name: str = "float32", threads: int = 10,
    strict_revision: bool = True,
) -> Runtime:
    """Configure deterministic execution before any experiment tensors exist."""
    if not REPO_ROOT.is_dir() or not MAIN_ROOT.is_dir():
        raise FileNotFoundError("Expected qefem_prof and main reference directories were not found")
    revision = _revision()
    if strict_revision and revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"Expected qefem_prof revision {EXPECTED_REVISION}, found {revision}. "
            "Set strict_revision=False only as an intentional new experiment."
        )
    device = resolve_device(device)
    dtype = getattr(torch, dtype_name)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    torch_device = validate_device_dtype(device, dtype)
    configure_reproducibility("strict", torch_device)
    torch.empty(0, device=torch_device, dtype=dtype)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in ("gset", "gset-big", "k2000", "wishart", "chook-tile"):
        (RESULTS_ROOT / folder).mkdir(parents=True, exist_ok=True)
    return Runtime(torch_device, dtype, dtype_name, threads, revision)


def _complete(path: Path) -> bool:
    manifest = path / "manifest.json"
    return manifest.exists() and _read_json(manifest).get("status") == "complete"


def _case_has_rows(path: Path, method: str, count: int) -> bool:
    """Return true only when a completed case contains the requested evidence."""
    endpoint = path / "endpoints.csv"
    if not _complete(path) or not endpoint.exists():
        return False
    frame = pd.read_csv(endpoint)
    return "method" in frame and int((frame.method == method).sum()) >= count


def _instance_path(family: str, index: int) -> Path:
    if family == "k2000":
        return REPO_ROOT / "benchmarks/k2000/WK2000_1.rud"
    folder = "wishart" if family == "wishart" else "chunk"
    return REPO_ROOT / "benchmarks" / folder / "instances" / f"instance_{index:02d}.npz"


def _derived_seed(base: int, family: str, index: int, trial: int) -> int:
    family_index = ["k2000", "wishart", "chook"].index(family)
    return int(np.random.SeedSequence([base, family_index, index, trial]).generate_state(1)[0])


def _reference_trial(family: str, index: int) -> tuple[Path, dict[str, Any]]:
    case = REFERENCE_ROOT / "cases" / f"{family}_{index:02d}"
    rows = [json.loads(line) for line in (case / "trials.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["solver"] == "double" and row["status"] == "ok"]
    if len(rows) != 1:
        raise ValueError(f"Expected one audited reference trial in {case}")
    return case, rows[0]


def _reference_inputs(instance: Any, config: dict[str, Any], seed: int) -> dict[str, np.ndarray] | None:
    """Return the audited local trial-0 inputs when the complete config matches."""
    family = instance.family
    index = instance.index
    try:
        case, row = _reference_trial(family, index)
    except (FileNotFoundError, ValueError):
        return None
    if int(row["seed"]) != int(seed) or row["config"] != config:
        return None
    expected = input_metadata(instance, config, seed, "double")
    return load_replay_inputs(case, row, expected)


def _endpoint(instance: Any, spins: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    return score_trial(instance, spins, tolerance=1e-9)


class QuantumWinnerTrajectory:
    """Coherent history of the replica selected by the final readout."""

    def __init__(self, score_one: Callable[[np.ndarray], dict[str, float]], winner: int):
        self.score_one = score_one
        self.winner = int(winner)
        self.rows: list[dict[str, float]] = []

    def __call__(self, step: int, fields: torch.Tensor) -> None:
        selected = fields[self.winner:self.winner + 1]
        if selected.ndim != 3 or selected.shape[2] != 2:
            raise ValueError(f"Expected quantum fields [replica,node,2], got {tuple(fields.shape)}")
        _, z, entropy = quantum_binary_state(selected)
        spins = (2 * (0.5 * (1 + z)).round() - 1).to(torch.int8).cpu().numpy()
        row = {"step": int(step), **self.score_one(spins)}
        row["entropy"] = float(entropy.item())
        self.rows.append(row)


class LQATrajectory:
    """Discrete LQA history. Entropy is blank because LQA has no QeFEM entropy."""

    def __init__(self, score_one: Callable[[np.ndarray], dict[str, float]]):
        self.score_one = score_one
        self.rows: list[dict[str, float | str]] = []

    def __call__(self, step: int, fields: torch.Tensor) -> None:
        spins = torch.sign(fields[0]).to(torch.int8).cpu().numpy()[None, :]
        spins[spins == 0] = 1
        self.rows.append({"step": int(step), **self.score_one(spins), "entropy": ""})


def _ising_score_one(instance: Any) -> Callable[[np.ndarray], dict[str, float]]:
    def score(spins: np.ndarray) -> dict[str, float]:
        energy = float(instance.score(spins)[0])
        if instance.family == "k2000":
            cut = float((instance.edges[2].sum() - energy) / 2)
            gap = float(instance.target_cut - cut)
            return {"discrete_cut": cut, "gap": gap}
        gap = float(energy - instance.ground_energy)
        return {"discrete_energy": energy, "gap": gap}
    return score


def _gset_score_one(graph: Any) -> Callable[[np.ndarray], dict[str, float]]:
    def score(spins: np.ndarray) -> dict[str, float]:
        cut = float(graph.score(spins)[0])
        return {"discrete_cut": cut, "gap": float(graph.target - cut)}
    return score


def _fresh_or_reference_inputs(
    instance: Any, config: dict[str, Any], seed: int, runtime: Runtime,
) -> dict[str, np.ndarray]:
    inputs = _reference_inputs(instance, config, seed)
    return (inputs if inputs is not None else
            make_trial_inputs(instance, config, seed, runtime.device, runtime.dtype, "double"))


def _run_double_endpoint(
    instance: Any, config: dict[str, Any], seed: int, inputs: dict[str, np.ndarray], runtime: Runtime,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    synchronize(runtime.device)
    started = time.perf_counter()
    spins = run_fem(instance, config, seed, runtime.device, runtime.dtype, inputs=inputs, callback=callback)
    synchronize(runtime.device)
    score, energies = _endpoint(instance, spins)
    score["runtime_sec"] = time.perf_counter() - started
    return score, energies


def _run_lqa_endpoint(
    instance: Any, config: dict[str, Any], seed: int, inputs: dict[str, np.ndarray], runtime: Runtime,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    synchronize(runtime.device)
    started = time.perf_counter()
    spins = run_lqa(instance, config, seed, runtime.device, runtime.dtype, REPO_ROOT,
                    inputs=inputs, callback=callback)
    synchronize(runtime.device)
    score, energies = _endpoint(instance, spins)
    score["runtime_sec"] = time.perf_counter() - started
    return score, energies


def _load_plan_job(path: Path, method: str) -> dict[str, Any]:
    jobs = _read_json(path)
    return next(job for job in jobs if job["name"] == method)


def _job_protocol(path: Path, method: str, instance: Any) -> tuple[Any, dict[str, Any]]:
    job = _load_plan_job(path, method)
    args = parse_args(job["args"])
    return args, configurations(args, instance)[method]


def _trial_row(
    method: str, trial: int, seed: int, score: dict[str, Any],
    ground_energy: float | None = None,
) -> dict[str, Any]:
    row = {
        "method": method,
        "trial": trial,
        "seed": seed,
        "energy": score["energy"],
        "cut": score["cut"],
        "gap": (score["gap_to_target"] if score["gap_to_target"] is not None else
                float(score["energy"] - ground_energy)),
        "relative_error": score["relative_error"],
        "ground_hit": score["ground_hit"],
        "target_hit": score["target_hit"],
        "best_replica": score["best_replica"],
        "runtime_sec": score["runtime_sec"],
    }
    return row


def _best_row(rows: list[dict[str, Any]], family: str, method: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["method"] == method]
    key = (lambda row: row["cut"]) if family == "k2000" else (lambda row: -row["relative_error"])
    return max(candidates, key=key)


def _write_case_report(
    folder: Path, title: str, rows: list[dict[str, Any]], config: dict[str, Any],
    trajectory_note: str,
) -> None:
    double = [row for row in rows if row["method"] == "qefem"]
    lqa = [row for row in rows if row["method"] == "lqa"]
    lines = [f"# {title}", "", trajectory_note, "", "## Endpoints", ""]
    if double and double[0]["cut"] is not None:
        cuts = [float(row["cut"]) for row in double]
        lines += [f"- QeFEM mean cut: `{np.mean(cuts):.6g}`", f"- QeFEM best cut: `{max(cuts):.6g}`"]
        if lqa:
            values = [float(row["cut"]) for row in lqa]
            lines += [f"- LQA mean cut: `{np.mean(values):.6g}`", f"- LQA best cut: `{max(values):.6g}`"]
    elif double:
        errors = [float(row["relative_error"]) for row in double]
        lines += [f"- QeFEM mean relative error: `{100*np.mean(errors):.6f}%`",
                  f"- QeFEM best relative error: `{100*min(errors):.6f}%`"]
        if lqa:
            errors = [float(row["relative_error"]) for row in lqa]
            lines += [f"- LQA mean relative error: `{100*np.mean(errors):.6f}%`",
                      f"- LQA best relative error: `{100*min(errors):.6f}%`"]
    lines += ["", "## Files", "", "- `endpoints.csv`: every completed endpoint.",
              "- `trajectory.csv`: coherent QeFEM final-winner history.",
              "- `trajectory_lqa.csv`: coherent LQA history when LQA is enabled.",
              "- `configuration.json`: complete settings and seeds.", ""]
    (folder / "report.md").write_text("\n".join(lines))
    _write_json(folder / "configuration.json", config)


def run_planted_family(
    family: str, runtime: Runtime, *, overwrite: bool = False, smoke: bool = False,
    include_lqa: bool = True,
) -> pd.DataFrame:
    """Run all Wishart or Chook instances and trace the best local trial."""
    if family not in ("wishart", "chook"):
        raise ValueError("family must be wishart or chook")
    out_family = RESULTS_ROOT / ("wishart" if family == "wishart" else "chook-tile")
    plans = ([REPO_ROOT / "benchmarks/wishart/results/portable_accuracy_recheck/final_wishart/plan.json"]
             if family == "wishart" else [
                 REPO_ROOT / "benchmarks/chunk/results/portable_accuracy_recheck/final_chook_a/plan.json",
                 REPO_ROOT / "benchmarks/chunk/results/portable_accuracy_recheck/final_chook_b/plan.json",
             ])
    count = 1 if smoke else (10 if family == "wishart" else 11)
    family_rows: list[dict[str, Any]] = []
    for index in range(count):
        instance = load_instance(_instance_path(family, index), family, index, target_cut=K2000_TARGET)
        label = f"alpha-{instance.parameter:.1f}" if family == "wishart" else f"p-c3-{instance.parameter:.1f}"
        folder = out_family / label
        folder.mkdir(parents=True, exist_ok=True)
        required_trials = 1 if smoke else (5 if family == "wishart" else 10)
        ready = (_case_has_rows(folder, "qefem", required_trials) and
                 (not include_lqa or _case_has_rows(folder, "lqa", required_trials)))
        if ready and not overwrite:
            family_rows.extend(pd.read_csv(folder / "endpoints.csv").to_dict("records"))
            print(f"{family}/{label}: complete; skipped")
            continue
        endpoint_rows: list[dict[str, Any]] = []
        double_inputs: dict[int, tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]] = {}
        trial_number = 0
        for plan in plans:
            args_d, double_config = _job_protocol(plan, "double", instance)
            args_l, lqa_config = _job_protocol(plan, "lqa", instance)
            trials = 1 if smoke else int(args_d.trials)
            if smoke:
                for config in (double_config, lqa_config):
                    config["steps"] = 4
                    config["replicas"] = 4 if config is double_config else 1
            for local_trial in range(trials):
                seed = _derived_seed(int(args_d.seed), family, index, local_trial)
                inputs = _fresh_or_reference_inputs(instance, double_config, seed, runtime)
                score, _ = _run_double_endpoint(instance, double_config, seed, inputs, runtime)
                endpoint_rows.append(_trial_row(
                    "qefem", trial_number, seed, score, float(instance.ground_energy)
                ))
                double_inputs[trial_number] = (double_config, inputs, score)
                if include_lqa:
                    lqa_inputs = make_trial_inputs(instance, lqa_config, seed, runtime.device, runtime.dtype, "lqa")
                    lqa_score, _ = _run_lqa_endpoint(instance, lqa_config, seed, lqa_inputs, runtime)
                    endpoint_rows.append(_trial_row(
                        "lqa", trial_number, seed, lqa_score, float(instance.ground_energy)
                    ))
                trial_number += 1
        best = _best_row(endpoint_rows, family, "qefem")
        config, inputs, expected = double_inputs[int(best["trial"])]
        recorder = QuantumWinnerTrajectory(_ising_score_one(instance), int(expected["best_replica"]))
        replay, _ = _run_double_endpoint(instance, config, int(best["seed"]), inputs, runtime, recorder)
        if replay["energy"] != expected["energy"]:
            raise RuntimeError(f"{family}/{label}: deterministic QeFEM replay changed the endpoint")
        _write_csv(folder / "trajectory.csv", recorder.rows)
        if include_lqa:
            best_lqa = _best_row(endpoint_rows, family, "lqa")
            _, lqa_config = _job_protocol(plans[0], "lqa", instance)
            if smoke:
                lqa_config["steps"] = 4
            lqa_inputs = make_trial_inputs(instance, lqa_config, int(best_lqa["seed"]),
                                           runtime.device, runtime.dtype, "lqa")
            lqa_recorder = LQATrajectory(_ising_score_one(instance))
            _run_lqa_endpoint(instance, lqa_config, int(best_lqa["seed"]), lqa_inputs,
                              runtime, lqa_recorder)
            _write_csv(folder / "trajectory_lqa.csv", lqa_recorder.rows)
        _write_csv(folder / "endpoints.csv", endpoint_rows)
        config_record = {
            "family": family, "instance": instance.name, "parameter": instance.parameter,
            "instance_sha256": sha256(instance.path), "plans": [str(path) for path in plans],
            "qefem": config, "seeds": [row["seed"] for row in endpoint_rows if row["method"] == "qefem"],
            "lqa_included": include_lqa, "device": str(runtime.device), "dtype": runtime.dtype_name,
        }
        _write_case_report(
            folder, f"{family.title()} {label}", endpoint_rows, config_record,
            "The trajectory follows the replica selected by the best trial's final QeFEM readout; "
            "it does not splice together replicas.",
        )
        _write_json(folder / "manifest.json", {
            "status": "complete", "lqa_included": include_lqa,
            "qefem_trials": len([r for r in endpoint_rows if r["method"] == "qefem"]),
            "lqa_trials": len([r for r in endpoint_rows if r["method"] == "lqa"]),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        family_rows.extend(endpoint_rows)
        print(f"{family}/{label}: complete")
    frame = pd.DataFrame(family_rows)
    frame.to_csv(out_family / "endpoints.csv", index=False)
    _family_report(out_family, family, frame)
    return frame


def _family_report(folder: Path, family: str, frame: pd.DataFrame) -> None:
    lines = [f"# {family.title()} paper experiment", ""]
    for method, group in frame.groupby("method"):
        if family == "k2000":
            lines.append(f"- {method}: mean cut `{group['cut'].mean():.6g}`, best `{group['cut'].max():.6g}`")
        else:
            lines.append(f"- {method}: mean relative error `{100*group['relative_error'].mean():.6f}%`")
    lines += ["", "Each instance subdirectory contains endpoints, a coherent trajectory, settings, and a report.", ""]
    (folder / "report.md").write_text("\n".join(lines))


def run_k2000(
    runtime: Runtime, *, replica_counts: Iterable[int] = K2000_REPLICAS,
    overwrite: bool = False, smoke: bool = False, include_lqa: bool = True,
) -> pd.DataFrame:
    """Run nested population prefixes at several replica counts plus LQA."""
    out = RESULTS_ROOT / "k2000"
    instance = load_instance(_instance_path("k2000", 0), "k2000", 0, target_cut=K2000_TARGET)
    plan = REPO_ROOT / "benchmarks/k2000/results/portable_accuracy_recheck/final_k2000/plan.json"
    args_d, base_config = _job_protocol(plan, "double", instance)
    args_l, lqa_config = _job_protocol(plan, "lqa", instance)
    counts = [4, 8] if smoke else list(replica_counts)
    trials = 1 if smoke else int(args_d.trials)
    if smoke:
        base_config["steps"] = 4
        lqa_config["steps"] = 4
    all_rows: list[dict[str, Any]] = []
    for replicas in counts:
        folder = out / f"replicas-{replicas:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        if _case_has_rows(folder, "qefem", trials) and not overwrite:
            all_rows.extend(pd.read_csv(folder / "endpoints.csv").to_dict("records"))
            print(f"k2000/{folder.name}: complete; skipped")
            continue
        config = json.loads(json.dumps(base_config))
        config["replicas"] = int(replicas)
        endpoint_rows: list[dict[str, Any]] = []
        saved: dict[int, tuple[int, dict[str, np.ndarray], dict[str, Any]]] = {}
        for trial in range(trials):
            seed = _derived_seed(int(args_d.seed), "k2000", 0, trial)
            full = _fresh_or_reference_inputs(instance, base_config, seed, runtime)
            inputs = dict(full)
            inputs["initial_fields"] = full["initial_fields"][:replicas].copy()
            score, _ = _run_double_endpoint(instance, config, seed, inputs, runtime)
            row = _trial_row("qefem", trial, seed, score)
            row["replicas"] = replicas
            endpoint_rows.append(row)
            saved[trial] = (seed, inputs, score)
        best = _best_row(endpoint_rows, "k2000", "qefem")
        seed, inputs, expected = saved[int(best["trial"])]
        recorder = QuantumWinnerTrajectory(_ising_score_one(instance), int(expected["best_replica"]))
        replay, _ = _run_double_endpoint(instance, config, seed, inputs, runtime, recorder)
        if replay["cut"] != expected["cut"]:
            raise RuntimeError(f"k2000/{replicas}: deterministic QeFEM replay changed the endpoint")
        _write_csv(folder / "trajectory.csv", recorder.rows)
        _write_csv(folder / "endpoints.csv", endpoint_rows)
        config_record = {
            "family": "k2000", "target": K2000_TARGET, "replicas": replicas,
            "population_semantics": "nested prefix of the same trial population at each seed",
            "qefem": config, "seeds": [row["seed"] for row in endpoint_rows],
            "device": str(runtime.device), "dtype": runtime.dtype_name,
        }
        _write_case_report(
            folder, f"K2000 with {replicas:,} replicas", endpoint_rows, config_record,
            "The trajectory follows the final winner from the best local endpoint trial.",
        )
        _write_json(folder / "manifest.json", {
            "status": "complete", "qefem_trials": trials,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        all_rows.extend(endpoint_rows)
        print(f"k2000/{folder.name}: complete")
    if include_lqa:
        folder = out / "lqa"
        folder.mkdir(parents=True, exist_ok=True)
        if not _complete(folder) or overwrite:
            lqa_rows = []
            saved_lqa = {}
            for trial in range(trials):
                seed = _derived_seed(int(args_l.seed), "k2000", 0, trial)
                inputs = make_trial_inputs(instance, lqa_config, seed, runtime.device, runtime.dtype, "lqa")
                score, _ = _run_lqa_endpoint(instance, lqa_config, seed, inputs, runtime)
                lqa_rows.append(_trial_row("lqa", trial, seed, score))
                saved_lqa[trial] = (seed, inputs)
            best = _best_row(lqa_rows, "k2000", "lqa")
            seed, inputs = saved_lqa[int(best["trial"])]
            recorder = LQATrajectory(_ising_score_one(instance))
            _run_lqa_endpoint(instance, lqa_config, seed, inputs, runtime, recorder)
            _write_csv(folder / "trajectory_lqa.csv", recorder.rows)
            _write_csv(folder / "endpoints.csv", lqa_rows)
            _write_json(folder / "configuration.json", {
                "family": "k2000", "target": K2000_TARGET,
                "instance_sha256": sha256(instance.path), "plan": str(plan),
                "lqa": lqa_config, "seeds": [r["seed"] for r in lqa_rows],
                "device": str(runtime.device), "dtype": runtime.dtype_name,
            })
            (folder / "report.md").write_text(
                "# K2000 LQA endpoint report\n\n"
                f"- Mean cut: `{np.mean([float(r['cut']) for r in lqa_rows]):.6g}`.\n"
                f"- Best cut: `{max(float(r['cut']) for r in lqa_rows):.6g}`.\n"
                f"- Target: `{K2000_TARGET:.0f}`.\n"
                "- `trajectory_lqa.csv` follows the best local LQA endpoint.\n"
            )
            _write_json(folder / "manifest.json", {
                "status": "complete", "lqa_trials": trials,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            })
        lqa_rows = pd.read_csv(folder / "endpoints.csv").to_dict("records")
        all_rows.extend(lqa_rows)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out / "endpoints.csv", index=False)
    lines = ["# K2000 paper experiment", "",
             "QeFEM population sizes are nested prefixes, so quality versus population size is directly auditable.", ""]
    q = frame[frame.method == "qefem"]
    for replicas, group in q.groupby("replicas"):
        lines.append(f"- {int(replicas):,} replicas: mean cut `{group.cut.mean():.6g}`, best `{group.cut.max():.6g}`, "
                     f"best target gap `{K2000_TARGET-group.cut.max():.6g}`.")
    if include_lqa:
        l = frame[frame.method == "lqa"]
        lines.append(f"- LQA: mean cut `{l.cut.mean():.6g}`, best `{l.cut.max():.6g}`.")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return frame


def _gset_jobs(big: bool) -> list[dict[str, Any]]:
    if big:
        final = _read_json(REPO_ROOT / "benchmarks/Gset/results/accuracy_tuning/plans/final_all.json")
        return [next(job for job in final if job["instance"] == name and job["solver"] == "double")
                for name in GSET_BIG]
    smooth = _read_json(REPO_ROOT / "benchmarks/Gset/results/target_attainment/plans/evaluate.json")
    smooth_by_name = {job["instance"]: job for job in smooth if job["solver"] == "double"}
    jobs = []
    for name in GSET_ALL:
        discrete = REPO_ROOT / f"benchmarks/Gset/results/remaining_discrete/plans/evaluation_{name}.json"
        jobs.append(_read_json(discrete)[0] if discrete.exists() else smooth_by_name[name])
    return jobs


def _run_gset_auto(
    graph: Any, config: dict[str, Any], inputs: dict[str, np.ndarray], runtime: Runtime,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> np.ndarray:
    mode = validate_gradient_config(config)
    if config["coupling_scale"] == "rms":
        scale = max(float(np.sqrt(2 * np.square(graph.w).sum() / graph.n)), np.finfo(float).eps)
    else:
        scale = float(config["coupling_scale"])
    storage = "dense" if runtime.device.type == "mps" else config.get("storage", "csr")
    coupling = graph.couplings(runtime.device, runtime.dtype, storage, scale=scale)
    case = DoubleAnnealing.from_couplings(
        "customize", graph.n, len(graph.w), coupling, customize_expected_func=ising_expectation,
    )
    case.set_up_solver(
        num_trials=config["replicas"], num_steps=config["steps"], binary=True,
        dev=runtime.device, dtype=runtime.dtype, seed=0, optimizer=config["optimizer"],
        optimizer_kwargs=config["optimizer_kwargs"], learning_rate=config["learning_rate"],
        initial_fields=inputs["initial_fields"], beta_schedule=inputs["betas"],
        transverse_field_schedule=inputs["transverse_fields"],
    )
    if mode == "autograd":
        p = case.solver.iterate(callback=callback)
    else:
        p = iterate_manual_ising(case.solver, inputs["discretization_weights"], callback=callback)
    if tuple(p.shape) != (config["replicas"], graph.n) or not torch.isfinite(p).all():
        raise RuntimeError("Invalid G-set marginal population")
    return (2 * p.detach().cpu().round() - 1).numpy().astype(np.int8)


def run_gset(
    runtime: Runtime, *, big: bool = False, overwrite: bool = False, smoke: bool = False,
) -> pd.DataFrame:
    """Run every G1-G54 graph or every supplied G55+ graph."""
    jobs = _gset_jobs(big)
    if big and any(int(job["config"]["replicas"]) != 8192 for job in jobs):
        raise RuntimeError("The selected G55+ plans must all use the highest 8,192-replica setting")
    if smoke:
        jobs = jobs[:1]
    family_folder = RESULTS_ROOT / ("gset-big" if big else "gset")
    family_folder.mkdir(parents=True, exist_ok=True)
    if big:
        settings = []
        for job in jobs:
            folder = family_folder / job["instance"]
            preserved = _complete(folder) and not overwrite
            recorded = _read_json(folder / "configuration.json") if preserved else None
            seeds_for_record = recorded["seed_bases"] if recorded else job["seeds"][:1]
            settings.append({
                "graph": job["instance"], "plan_job": job["name"],
                "protocol": "preserved-complete" if preserved else "one-seed endpoint",
                "steps": job["config"]["steps"], "replicas": job["config"]["replicas"],
                "seed_bases": ";".join(str(seed) for seed in seeds_for_record),
                "optimizer": job["config"]["optimizer"],
                "learning_rate": job["config"]["learning_rate"],
                "momentum": job["config"]["optimizer_kwargs"].get("momentum"),
                "weight_decay": job["config"]["optimizer_kwargs"].get("weight_decay"),
                "gradient_mode": job["config"].get("gradient_mode", "autograd"),
                "temperature_schedule": job["config"]["temperature_schedule"],
                "gamma_schedule": job["config"]["gamma_schedule"],
            })
        _write_csv(family_folder / "selected_settings.csv", settings)
    all_rows: list[dict[str, Any]] = []
    topology = {name: label for label, names in GSET_COHORTS.items() for name in names}
    for job in jobs:
        name = job["instance"]
        graph = load_graph(name)
        folder = family_folder / name
        folder.mkdir(parents=True, exist_ok=True)
        if _complete(folder) and not overwrite:
            all_rows.extend(pd.read_csv(folder / "endpoints.csv").to_dict("records"))
            print(f"{'gset-big' if big else 'gset'}/{name}: complete; skipped")
            continue
        config = json.loads(json.dumps(job["config"]))
        seeds = list(job["seeds"][:1] if big else job["seeds"])
        if smoke:
            config["steps"] = 4
            config["replicas"] = 4
            seeds = seeds[:1]
            config.pop("spectral_schedule", None)
            config.pop("discretization_start", None)
            config.pop("discretization_end", None)
            config.pop("discretization_max", None)
            config["gradient_mode"] = "autograd"
        endpoint_rows = []
        saved: dict[int, tuple[int, dict[str, np.ndarray], int]] = {}
        for trial, base in enumerate(seeds):
            seed = gset_derived_seed(int(base), name)
            inputs = make_gset_inputs(graph, config, seed, runtime.device, runtime.dtype, "double")
            synchronize(runtime.device)
            started = time.perf_counter()
            spins = _run_gset_auto(graph, config, inputs, runtime)
            synchronize(runtime.device)
            cuts = graph.score(spins)
            winner = int(np.argmax(cuts))
            row = {
                "graph": name, "topology": topology.get(name, "large-supplied"),
                "trial": trial, "seed_base": base, "seed": seed,
                "replicas": config["replicas"], "steps": config["steps"],
                "target": graph.target, "best_cut": float(cuts[winner]),
                "gap": float(graph.target - cuts[winner]), "winner": winner,
                "target_hits": int(np.sum(cuts >= graph.target)),
                "runtime_sec": time.perf_counter() - started,
            }
            endpoint_rows.append(row)
            if not big:
                saved[trial] = (seed, inputs, winner)
        best = max(endpoint_rows, key=lambda row: row["best_cut"])
        if not big:
            seed, inputs, winner = saved[int(best["trial"])]
            recorder = QuantumWinnerTrajectory(_gset_score_one(graph), winner)
            replay_spins = _run_gset_auto(graph, config, inputs, runtime, callback=recorder)
            replay_cut = float(graph.score(replay_spins)[winner])
            if replay_cut != best["best_cut"]:
                raise RuntimeError(f"{name}: deterministic trajectory replay changed the endpoint")
            _write_csv(folder / "trajectory.csv", recorder.rows)
        _write_csv(folder / "endpoints.csv", endpoint_rows)
        record = {
            "graph": name, "topology": topology.get(name, "large-supplied"),
            "instance_sha256": sha256(graph.path), "target": graph.target,
            "plan_job": job["name"], "solver": job["solver"], "config": config,
            "seed_bases": seeds, "derived_seeds": [row["seed"] for row in endpoint_rows],
            "device": str(runtime.device), "dtype": runtime.dtype_name,
            "trajectory_semantics": ("endpoint-only; one 8,192-replica seed" if big else
                                     "coherent final-winner history from the best local seed"),
        }
        _write_json(folder / "configuration.json", record)
        lines = [f"# {name} QeFEM endpoint report", "",
                 f"- Graph type: `{record['topology']}`", f"- Vertices: `{graph.n}`",
                 f"- Edges: `{len(graph.w)}`", f"- Reference cut: `{graph.target:g}`",
                 f"- Best local cut: `{best['best_cut']:g}`", f"- Residual: `{best['gap']:g}`",
                 f"- Seeds: `{', '.join(str(s) for s in seeds)}`", "",
                 ("One 8,192-replica endpoint run; no trajectory replay." if big else
                  "`trajectory.csv` contains only step, discrete cut, reference gap, and entropy for one coherent replica."), ""]
        (folder / "report.md").write_text("\n".join(lines))
        _write_json(folder / "manifest.json", {
            "status": "complete", "scope": "endpoint-only" if big else "trajectory",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        all_rows.extend(endpoint_rows)
        print(f"{'gset-big' if big else 'gset'}/{name}: complete")
    frame = pd.DataFrame(all_rows)
    frame.to_csv(family_folder / "endpoints.csv", index=False)
    if not frame.empty:
        bests = frame.groupby("graph", as_index=False).agg(
            target=("target", "first"), best_cut=("best_cut", "max"), min_gap=("gap", "min")
        )
        bests.to_csv(family_folder / "summary.csv", index=False)
        lines = [f"# {'G-set G55+' if big else 'Complete G-set G1-G54'} experiment", "",
                 f"Completed graphs: `{len(bests)}`.",
                 f"Reference cuts reached: `{int((bests.min_gap <= 0).sum())}/{len(bests)}`.",
                 f"Median residual: `{bests.min_gap.median():.6g}`.", ""]
        if big:
            lines.append("New G55+ cases use one 8,192-replica seed and one endpoint pass. Previously completed cases are preserved; inspect selected_settings.csv for each case's protocol.")
        (family_folder / "report.md").write_text("\n".join(lines))
    return frame


def run_gset_remaining_endpoints(
    runtime: Runtime, *, preserve_trajectories: Iterable[str],
) -> pd.DataFrame:
    """Fill the 39 missing G1-G54 endpoints without rerunning the 15 trajectory cases.

    This uses the current QeFEM solver and the archived per-instance plans.
    G17 and G28 replay the particular discovery/promotion plans that supplied
    their hits to the accumulated 33/54 result; the frozen evaluation plans
    for those two did not themselves reach the reference cuts.
    """
    preserved = set(preserve_trajectories)
    if len(preserved) != 15 or not preserved <= set(GSET_ALL):
        raise ValueError("Expected exactly 15 distinct G1-G54 trajectory cases")
    family_folder = RESULTS_ROOT / "gset"
    for name in sorted(preserved, key=lambda value: int(value[1:])):
        folder = family_folder / name
        if not (_complete(folder) and (folder / "endpoints.csv").exists()
                and (folder / "trajectory.csv").exists()):
            raise RuntimeError(f"{name}: preserved trajectory case is incomplete")

    jobs = _gset_jobs(False)
    if len(jobs) != 54 or {job["instance"] for job in jobs} != set(GSET_ALL):
        raise RuntimeError("Current G-set plan does not cover G1-G54 exactly once")
    jobs_by_name = {job["instance"]: job for job in jobs}
    plan_sources = {
        name: ("remaining_discrete/evaluation" if
               (REPO_ROOT / f"benchmarks/Gset/results/remaining_discrete/plans/evaluation_{name}.json").exists()
               else "target_attainment/evaluate")
        for name in GSET_ALL
    }
    special = {
        "G17": ("refine01.json", "G17_refine01"),
        "G28": ("promotion_G28.json", "G28_incumbent_exact"),
    }
    for name, (filename, candidate) in special.items():
        path = REPO_ROOT / "benchmarks/Gset/results/remaining_discrete/plans" / filename
        matches = [job for job in _read_json(path)
                   if job["instance"] == name and job["name"] == candidate
                   and job["solver"] == "double"]
        if len(matches) != 1:
            raise RuntimeError(f"{name}: historical hit plan is missing or ambiguous")
        jobs_by_name[name] = matches[0]
        plan_sources[name] = f"remaining_discrete/{filename}"

    pending = [name for name in GSET_ALL if name not in preserved]
    if len(pending) != 39:
        raise RuntimeError("Expected exactly 39 endpoint-only cases")
    topology = {name: label for label, names in GSET_COHORTS.items() for name in names}
    settings = []
    for name in pending:
        job = jobs_by_name[name]
        config = job["config"]
        settings.append({
            "graph": name, "plan_job": job["name"], "plan_source": plan_sources[name],
            "steps": config["steps"], "replicas": config["replicas"],
            "seeds": ";".join(str(seed) for seed in job["seeds"]),
            "optimizer": config["optimizer"], "learning_rate": config["learning_rate"],
            "momentum": config["optimizer_kwargs"].get("momentum"),
            "weight_decay": config["optimizer_kwargs"].get("weight_decay"),
            "gradient_mode": config.get("gradient_mode", "autograd"),
            "temperature_schedule": config["temperature_schedule"],
            "gamma_schedule": config["gamma_schedule"],
        })
    _write_csv(family_folder / "remaining_39_settings.csv", settings)
    print("Endpoint-only cases (39):", ", ".join(pending), flush=True)
    print("G17 and G28 use their recorded hit-producing plans; all other cases use the latest selected plans.", flush=True)

    for name in pending:
        job = jobs_by_name[name]
        graph = load_graph(name)
        folder = family_folder / name
        folder.mkdir(parents=True, exist_ok=True)
        if _complete(folder):
            saved = _read_json(folder / "configuration.json")
            if (saved.get("plan_job") != job["name"] or
                    saved.get("config") != job["config"] or
                    saved.get("seed_bases") != [int(seed) for seed in job["seeds"]] or
                    saved.get("instance_sha256") != sha256(graph.path) or
                    saved.get("device") != str(runtime.device) or
                    saved.get("dtype") != runtime.dtype_name):
                raise RuntimeError(f"{name}: existing completed result uses different settings or graph; refusing to mix it")
            print(f"gset/{name}: complete; skipped", flush=True)
            continue
        config = json.loads(json.dumps(job["config"]))
        seeds = [int(seed) for seed in job["seeds"]]
        endpoint_rows = []
        for trial, base in enumerate(seeds):
            seed = gset_derived_seed(base, name)
            inputs = make_gset_inputs(graph, config, seed, runtime.device, runtime.dtype, "double")
            synchronize(runtime.device)
            started = time.perf_counter()
            spins = _run_gset_auto(graph, config, inputs, runtime)
            synchronize(runtime.device)
            cuts = graph.score(spins)
            winner = int(np.argmax(cuts))
            endpoint_rows.append({
                "graph": name, "topology": topology.get(name, "unknown"),
                "trial": trial, "seed_base": base, "seed": seed,
                "replicas": config["replicas"], "steps": config["steps"],
                "target": graph.target, "best_cut": float(cuts[winner]),
                "gap": float(graph.target - cuts[winner]), "winner": winner,
                "target_hits": int(np.sum(cuts >= graph.target)),
                "runtime_sec": time.perf_counter() - started,
            })
            print(f"gset/{name}: seed {trial + 1}/{len(seeds)}; best cut {cuts[winner]:g}", flush=True)
        best = max(endpoint_rows, key=lambda row: row["best_cut"])
        _write_csv(folder / "endpoints.csv", endpoint_rows)
        _write_json(folder / "configuration.json", {
            "graph": name, "topology": topology.get(name, "unknown"),
            "instance_sha256": sha256(graph.path), "target": graph.target,
            "plan_job": job["name"], "plan_source": plan_sources[name],
            "solver": job["solver"], "config": config, "seed_bases": seeds,
            "derived_seeds": [row["seed"] for row in endpoint_rows],
            "device": str(runtime.device), "dtype": runtime.dtype_name,
            "source_revision": runtime.revision, "trajectory_semantics": "endpoint-only",
        })
        (folder / "report.md").write_text(
            f"# {name} QeFEM endpoint report\n\n"
            f"- Graph type: `{topology.get(name, 'unknown')}`.\n"
            f"- Reference cut: `{graph.target:g}`.\n"
            f"- Best local cut: `{best['best_cut']:g}`; residual: `{best['gap']:g}`.\n"
            f"- Selected plan: `{job['name']}` ({plan_sources[name]}).\n"
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.\n"
            "- This is an endpoint-only case; the 15 representative graphs retain full trajectories.\n"
        )
        _write_json(folder / "manifest.json", {
            "status": "complete", "scope": "endpoint-only",
            "source_revision": runtime.revision,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"gset/{name}: complete", flush=True)

    rows = []
    for name in GSET_ALL:
        folder = family_folder / name
        if not _complete(folder) or not (folder / "endpoints.csv").exists():
            raise RuntimeError(f"{name}: missing canonical endpoint after fill")
        rows.extend(pd.read_csv(folder / "endpoints.csv").to_dict("records"))
    frame = pd.DataFrame(rows)
    frame.to_csv(family_folder / "endpoints.csv", index=False)
    bests = frame.groupby("graph", as_index=False).agg(
        target=("target", "first"), best_cut=("best_cut", "max"), min_gap=("gap", "min")
    )
    bests.to_csv(family_folder / "summary.csv", index=False)
    (family_folder / "report.md").write_text(
        "# Complete G-set G1-G54 experiment\n\n"
        f"Completed graphs: `{len(bests)}`.\n"
        f"Reference cuts reached locally: `{int((bests.min_gap <= 0).sum())}/{len(bests)}`.\n"
        f"Median residual: `{bests.min_gap.median():.6g}`.\n"
        "Fifteen representative graphs include trajectories; the remaining 39 have endpoints only.\n"
        "The historical accumulated 33/54 result is not a guarantee for this local device.\n"
    )
    return frame


def import_existing_local_results() -> dict[str, int]:
    """Normalize already-completed laptop runs into the paper results tree.

    The validated 8,192-replica K2000, Wishart, and Chook runs are complete
    experiments and become resumable canonical records.  The older Adam G-set
    runs are useful historical baselines, but are deliberately stored with an
    ``imported_local_`` prefix so the Feng-tuned G-set cells still run.
    Candidate populations and plots are not copied: this workspace keeps only
    endpoint tables, concise reports, settings, and the requested trajectories.
    """
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("gset", "gset-big", "k2000", "wishart", "chook-tile"):
        (RESULTS_ROOT / name).mkdir(parents=True, exist_ok=True)

    trajectory_root = MAIN_ROOT / "da_trajectory_reproduction" / "cases"
    family_root = MAIN_ROOT / "da_family_reproduction" / "cases"
    imported = {"complete_cases": 0, "gset_endpoints": 0, "gset_trajectories": 0}
    family_rows: dict[str, list[dict[str, Any]]] = {"wishart": [], "chook": [], "k2000": []}

    for family, count in (("wishart", 10), ("chook", 11), ("k2000", 1)):
        for index in range(count):
            case_name = f"{family}_{index:02d}"
            source = trajectory_root / case_name
            result = _read_json(source / "result.json")
            parameter = result.get("parameter")
            if family == "wishart":
                folder = RESULTS_ROOT / "wishart" / f"alpha-{float(parameter):.1f}"
            elif family == "chook":
                folder = RESULTS_ROOT / "chook-tile" / f"p-c3-{float(parameter):.1f}"
            else:
                folder = RESULTS_ROOT / "k2000" / "replicas-08192"
            folder.mkdir(parents=True, exist_ok=True)
            prior_manifest = (_read_json(folder / "manifest.json")
                              if (folder / "manifest.json").exists() else {})
            preserve_canonical = (prior_manifest.get("status") == "complete" and
                                  prior_manifest.get("provenance") != "imported-local")

            local_trial = json.loads((family_root / case_name / "trials.jsonl").read_text().splitlines()[0])
            score = result["score"]
            row = {
                "method": "qefem", "trial": 0, "seed": int(result["seed"]),
                "energy": float(score["energy"]),
                "cut": score.get("cut"),
                "gap": score.get("gap_to_target"),
                "relative_error": score.get("relative_error"),
                "ground_hit": score.get("ground_hit"),
                "target_hit": score.get("target_hit"),
                "best_replica": int(score["best_replica"]),
                "runtime_sec": float(local_trial["runtime_sec"]),
            }
            if family == "k2000":
                row["replicas"] = int(result["replicas"])
            family_rows[family].append(row)

            raw = pd.read_csv(source / "trajectory.csv")
            if family == "k2000":
                trajectory = pd.DataFrame({
                    "step": raw["step"].astype(int),
                    "discrete_cut": raw["discrete_cut"],
                    "gap": K2000_TARGET - raw["discrete_cut"],
                    "entropy": raw["entropy"],
                })
            else:
                instance = load_instance(_instance_path(family, index), family, index, target_cut=K2000_TARGET)
                ground = float(instance.ground_energy)
                trajectory = pd.DataFrame({
                    "step": raw["step"].astype(int),
                    "discrete_energy": raw["discrete_ising_energy"],
                    "gap": raw["discrete_ising_energy"] - ground,
                    "entropy": raw["entropy"],
                })
            trajectory.to_csv(folder / "imported_local_trajectory.csv", index=False)
            _write_csv(folder / "imported_local_endpoint.csv", [row])
            shutil.copy2(source / "REPORT.md", folder / "imported_local_source_report.md")
            config_record = {
                "family": family, "case": case_name, "parameter": parameter,
                "instance_sha256": local_trial["instance_sha256"],
                "qefem": result["config"], "seeds": [int(result["seed"])],
                "device": result["device"], "dtype": result["dtype"],
                "lqa_included": False,
                "provenance": {
                    "kind": "imported validated local run",
                    "source_result": str(source / "result.json"),
                    "source_result_sha256": sha256(source / "result.json"),
                    "source_trajectory": str(source / "trajectory.csv"),
                    "source_trajectory_sha256": sha256(source / "trajectory.csv"),
                    "source_git_revision": result["git_revision"],
                    "two_pass_replay_exact": bool(result["two_pass_replay_exact"]),
                    "exact_reference_energy": bool(result["exact_reference_energy"]),
                },
            }
            _write_json(folder / "imported_local_configuration.json", config_record)
            title = "K2000, 8,192 replicas" if family == "k2000" else f"{family.title()} parameter {parameter}"
            metric = (f"cut `{float(score['cut']):.0f}`, target gap `{float(score['gap_to_target']):.0f}`"
                      if family == "k2000" else
                      f"energy `{float(score['energy']):.10g}`, relative error `{100*float(score['relative_error']):.6f}%`")
            report = [f"# {title}: imported local reference", "",
                      f"- Endpoint: {metric}.",
                      f"- Seed: `{int(result['seed'])}`; final winner: `{int(score['best_replica'])}`.",
                      f"- Local endpoint runtime: `{float(local_trial['runtime_sec']):.3f}` seconds.",
                      "- Replay check: exact endpoint reproduced in the recorded two-pass trajectory run.", "",
                      "`trajectory.csv` contains only the discrete objective, gap/error, and entropy of one coherent final-winner replica.",
                      "The original local report is retained as `imported_local_source_report.md`.", ""]
            if not preserve_canonical:
                trajectory.to_csv(folder / "trajectory.csv", index=False)
                _write_csv(folder / "endpoints.csv", [row])
                _write_json(folder / "configuration.json", config_record)
                (folder / "report.md").write_text("\n".join(report))
                _write_json(folder / "manifest.json", {
                    "status": "complete", "provenance": "imported-local",
                    "lqa_included": False, "imported_utc": datetime.now(timezone.utc).isoformat(),
                })
            imported["complete_cases"] += 1

    for family, rows in family_rows.items():
        root_name = "chook-tile" if family == "chook" else family
        out = RESULTS_ROOT / root_name
        _write_csv(out / "imported_local_endpoints.csv", rows)
        if family != "k2000":
            label = "alpha" if family == "wishart" else "p(C3)"
            report = [f"# Imported local {family.title()} endpoints", "",
                      f"Validated local CPU endpoints across every supplied {label} instance.",
                      f"Mean relative error: `{100*np.mean([float(r['relative_error']) for r in rows]):.6f}%`.", ""]
            (out / "imported_local_report.md").write_text("\n".join(report))
        else:
            (out / "imported_local_report.md").write_text(
                "# Imported local K2000 endpoint\n\n"
                f"The validated 8,192-replica run reached cut `{float(rows[0]['cut']):.0f}` "
                f"(gap `{float(rows[0]['gap']):.0f}` to 33,337).\n"
            )

    old_root = MAIN_ROOT / "maxcut_results" / "v2"
    small = pd.read_csv(old_root / "qefem_anneal_G1_G54_final_results.csv")
    big = pd.read_csv(old_root / "qefem_g55-g81_summary.csv")
    big = big[big["graph"].astype(str).str.match(r"^G\d+$", na=False)]
    selections = GSET_ALL
    topology = {name: kind for kind, names in GSET_COHORTS.items() for name in names}
    imported_small_rows: list[dict[str, Any]] = []
    imported_big_rows: list[dict[str, Any]] = []
    for name in selections + GSET_BIG:
        is_big = name in GSET_BIG
        source_row = (big[big.graph == name].iloc[0].to_dict() if is_big
                      else small[small.instance == name].iloc[0].to_dict())
        folder = RESULTS_ROOT / ("gset-big" if is_big else "gset") / name
        folder.mkdir(parents=True, exist_ok=True)
        endpoint = {
            "graph": name, "topology": "G55+" if is_big else topology[name],
            "vertices": int(source_row["N"] if is_big else source_row["nodes"]),
            "edges": int(source_row["M"] if is_big else source_row["edges"]),
            "reference_cut": float(source_row["best_known_cut"] if is_big else source_row["best_value"]),
            "best_cut": float(source_row["qefem_best_cut"] if is_big else source_row["best_cut"]),
            "gap": float(source_row["gap"]), "seed": None if is_big else int(source_row["seed"]),
            "replicas": int(source_row["replicas"]), "steps": int(source_row["steps"] if is_big else source_row["n_steps"]),
            "optimizer": str(source_row["optimizer"] if is_big else "Adam"),
            "learning_rate": float(source_row["lr"]), "runtime_sec": float(source_row["runtime_sec"]),
            "provenance": "older local Adam baseline; not Feng-tuned",
        }
        _write_csv(folder / "imported_local_endpoint.csv", [endpoint])
        _write_json(folder / "imported_local_configuration.json", {
            "graph": name,
            "source_summary": str(old_root / ("qefem_g55-g81_summary.csv" if is_big
                                                else "qefem_anneal_G1_G54_final_results.csv")),
            "source_summary_sha256": sha256(old_root / ("qefem_g55-g81_summary.csv" if is_big
                                                         else "qefem_anneal_G1_G54_final_results.csv")),
            "seed": endpoint["seed"], "replicas": endpoint["replicas"],
            "steps": endpoint["steps"], "optimizer": endpoint["optimizer"],
            "learning_rate": endpoint["learning_rate"],
            "provenance": endpoint["provenance"],
        })
        imported["gset_endpoints"] += 1
        (imported_big_rows if is_big else imported_small_rows).append(endpoint)
        history = old_root / "full_res" / f"{name}_qefem_v2_anneal_history.csv"
        history_note = "No per-step history was present in the local archive."
        if history.exists():
            raw = pd.read_csv(history)
            normalized = pd.DataFrame({
                "step": raw["global_step"].astype(int),
                "discrete_cut": raw["best_cut_so_far"],
                "gap": raw["gap_so_far"],
                "entropy": raw["entropy_mean"],
            })
            normalized.to_csv(folder / "imported_local_trajectory.csv", index=False)
            imported["gset_trajectories"] += 1
            history_note = "A normalized historical trajectory is available as `imported_local_trajectory.csv`."
        (folder / "imported_local_report.md").write_text(
            f"# {name}: imported older local baseline\n\n"
            f"This 128-replica Adam run reached cut `{endpoint['best_cut']:.0f}` against reference "
            f"`{endpoint['reference_cut']:.0f}` (gap `{endpoint['gap']:.0f}`). {history_note}\n\n"
            "It is retained for workflow continuity and is not presented as a Feng-tuned result.\n"
        )
    _write_csv(RESULTS_ROOT / "gset" / "imported_local_endpoints.csv", imported_small_rows)
    _write_csv(RESULTS_ROOT / "gset-big" / "imported_local_endpoints.csv", imported_big_rows)
    _write_json(RESULTS_ROOT / "import_manifest.json", {
        **imported, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_roots": [str(trajectory_root), str(family_root), str(old_root)],
        "note": "No raw candidate populations were copied; the paper workspace contains only endpoints, compact trajectories, settings, and reports.",
    })
    return imported


def write_root_report(runtime: Runtime) -> None:
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": REPO_ROOT, "repo_revision": runtime.revision,
        "main_reference_root": MAIN_ROOT, "results_root": RESULTS_ROOT,
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "runtime": runtime_metadata(runtime.device),
    }
    _write_json(RESULTS_ROOT / "experiment_environment.json", metadata)
    lines = ["# QeFEM paper experiment outputs", "",
             "The notebook writes each completed case immediately and can be resumed safely.", "",
             "- `k2000/`: nested replica-count experiments plus LQA.",
             "- `wishart/`: alpha 0.1-1.0 endpoints and best-trial trajectories.",
             "- `chook-tile/`: p(C3) 0.0-1.0 endpoints and best-trial trajectories.",
             "- `gset/`: every supplied graph from G1 through G54.",
             "- `gset-big/`: every supplied graph with index greater than 54.", "",
             "All configuration files retain seeds, hyperparameters, source plan names, and data hashes.", ""]
    (RESULTS_ROOT / "README.md").write_text("\n".join(lines))


def run_all(
    runtime: Runtime, *, overwrite: bool = False, smoke: bool = False,
    include_lqa: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the paper suite using the same G-set completion path as the notebook."""
    if overwrite and not smoke:
        raise ValueError("run_all preserves the 15 representative G-set trajectories; use overwrite=False")
    outputs = {
        "k2000": run_k2000(runtime, overwrite=overwrite, smoke=smoke, include_lqa=include_lqa),
        "wishart": run_planted_family("wishart", runtime, overwrite=overwrite, smoke=smoke,
                                       include_lqa=include_lqa),
        "chook": run_planted_family("chook", runtime, overwrite=overwrite, smoke=smoke,
                                     include_lqa=include_lqa),
        "gset": (run_gset(runtime, big=False, overwrite=overwrite, smoke=True) if smoke else
                 run_gset_remaining_endpoints(
                     runtime, preserve_trajectories=GSET_REPRESENTATIVE_15,
                 )),
        "gset_big": run_gset(runtime, big=True, overwrite=overwrite, smoke=smoke),
    }
    write_root_report(runtime)
    return outputs
