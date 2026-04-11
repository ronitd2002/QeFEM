# experiments/run_er_experiment.py
# Orchestrates one ER-graph experiment:
#   1. Generate graph
#   2. Run classical FEM stage
#   3. Run quantum MF stage
#   4. (Optional) brute-force verification for N <= 20
#   5. Print results + save plot
#
# Usage:
#   python -m experiment.run_er_experiment
# or import run_experiment() and call it from a sweep script.

import time
from dataclasses import dataclass
from typing import Optional

import torch

from qafem.interface import QAFEM
from qafem.utils import get_device, generate_erdos_renyi_graph
from experiment.plotting import build_nx_layout, plot_experiment_stages


# ---------------------------
# Brute-force verifier
# ---------------------------

def brute_force_maxcut(W: torch.Tensor):
    """
    Exact MaxCut by full enumeration. CPU only. Safe only for N <= 20.
    Returns (max_cut_value, best_config_tensor).
    """
    W_cpu = W.detach().cpu()
    N = W_cpu.shape[0]
    assert N <= 20, f"N={N} > 20: brute force will OOM or take forever."

    indices = torch.arange(2 ** N, dtype=torch.long)
    bits = torch.arange(N)
    configs = ((indices.unsqueeze(1) >> bits) & 1).float()   # [2^N, N]
    cut_values = ((configs @ W_cpu) * (1 - configs)).sum(dim=1)

    max_val, max_idx = cut_values.max(dim=0)
    return max_val.item(), configs[max_idx]


# ---------------------------
# Result container
# ---------------------------

@dataclass
class ExperimentResult:
    cut_classical: float
    cut_quantum: float
    gain: float
    exact_cut: Optional[float]
    acc_classical: Optional[float]   # cut_classical / exact_cut
    acc_quantum: Optional[float]     # cut_quantum   / exact_cut
    t_classical: float
    t_quantum: float
    t_total: float
    optimal_found: Optional[bool]    # None if exact unknown


# ---------------------------
# Single experiment
# ---------------------------

def run_experiment(
    # Graph
    N: int = 15,
    edge_prob: float = 0.4,
    weight_mode: str = "ones",
    graph_seed: int = 7,
    # Solver — classical
    replicas: int = 128,
    beta_min: float = 1e-2,
    beta_max: float = 40.0,
    beta_steps: int = 300,
    lr_classical: float = 0.04,
    # Solver — quantum
    gamma_max: float = 1.5,
    gamma_min: float = 0.0,
    gamma_steps: int = 200,
    beta_q: float = 60.0,
    lr_a: float = 0.02,
    lr_theta: float = 0.02,
    # Misc
    solver_seed: int = 123,
    verify_exact: bool = True,
    save_path: str = "results/experiment_plot.png",
    verbose: bool = True,
) -> ExperimentResult:

    device = get_device()
    if verbose:
        print(f"Device : {device}")
        print(f"Graph  : N={N}, p={edge_prob}, mode={weight_mode}, seed={graph_seed}")

    # --- Graph ---
    W = generate_erdos_renyi_graph(
        n=N, edge_prob=edge_prob,
        weight_mode=weight_mode, seed=graph_seed, device=device,
    )
    G, pos = build_nx_layout(W)

    # --- Random init (for visualisation only — do NOT use best_c index here) ---
    torch.manual_seed(solver_seed)
    p_init_all = torch.sigmoid(
        1e-3 * torch.randn((replicas, N), device=device)
    )
    # Show replica 0 — representative of "before any optimisation"
    p_init_vis = p_init_all[0].detach().cpu()

    # --- Run QAFEM pipeline ---
    qa = QAFEM.from_couplings(
        W,
        num_trials=replicas,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_steps=beta_steps,
        beta_schedule="inverse",
        gamma_max=gamma_max,
        gamma_min=gamma_min,
        gamma_steps=gamma_steps,
        gamma_schedule="linear",
        beta_q_min=beta_q,
        beta_q_max=beta_q,
        beta_q_schedule="fixed",
        lr=lr_classical,
        lr_a=lr_a,
        lr_theta=lr_theta,
        device=device,
        seed=solver_seed,
    )

    t0 = time.perf_counter()
    result = qa.solve()
    t_total = time.perf_counter() - t0
    # Solver doesn't split timing internally; approximate classical vs quantum
    # as proportional to their step counts
    classical_fraction = beta_steps / (beta_steps + gamma_steps)
    t_classical = t_total * classical_fraction
    t_quantum = t_total * (1 - classical_fraction)

    p_classical_vis = result["best_classical_p"].detach().cpu()
    spins_classical = result["best_spins_classical"].detach().cpu()
    cut_classical = float(result["best_classical_cut"])

    p_quantum_vis = result["best_quantum_p"].detach().cpu()
    spins_quantum = result["best_spins_quantum"].detach().cpu()
    cut_quantum = float(result["best_quantum_cut"])

    # --- Brute-force verification ---
    exact_cut = acc_c = acc_q = optimal_found = None
    if verify_exact:
        if N <= 20:
            if verbose:
                print("Running brute-force exact MaxCut...")
            t_bf = time.perf_counter()
            exact_cut, _ = brute_force_maxcut(W)
            if verbose:
                print(f"Brute-force done in {time.perf_counter()-t_bf:.3f}s")
            acc_c = cut_classical / exact_cut if exact_cut > 0 else float("nan")
            acc_q = cut_quantum   / exact_cut if exact_cut > 0 else float("nan")
            optimal_found = (max(cut_classical, cut_quantum) >= exact_cut - 1e-4)
        else:
            if verbose:
                print(f"Skipping brute-force: N={N} > 20.")

    # --- Print ---
    if verbose:
        print("\n" + "=" * 48)
        print(f"{'Stage':<22} {'Cut':>10}  {'Time':>8}")
        print("-" * 48)
        print(f"{'Classical FEM':<22} {cut_classical:>10.4f}  {t_classical:>7.2f}s")
        print(f"{'Quantum MF':<22} {cut_quantum:>10.4f}  {t_quantum:>7.2f}s")
        print(f"{'Total':<22} {'':>10}  {t_total:>7.2f}s")
        if exact_cut is not None:
            print("-" * 48)
            print(f"{'Exact (brute force)':<22} {exact_cut:>10.4f}")
            print(f"Classical accuracy : {acc_c*100:>6.2f}%")
            print(f"Quantum   accuracy : {acc_q*100:>6.2f}%")
            print(f"Stage-2 gain       : {cut_quantum - cut_classical:>+10.4f}")
            if optimal_found:
                winner = "Quantum" if cut_quantum >= exact_cut - 1e-4 else "Classical"
                print(f"✓ {winner} stage found the global optimum.")
            else:
                print("✗ Neither stage found the global optimum.")
        print("=" * 48)
        # Improvement 3: quantum stage diagnostics
        qh = result["quantum_history"]
        print("\nQuantum stage diagnostics (final step):")
        print(f"  |rx| mean : {qh['rx_mean'][-1]:.4f}  (~0 = collapsed to classical, no quantum effect)")
        print(f"  S_vN mean : {qh['S_vN_mean'][-1]:.4f}  (~0 = fully decided spins)")
        print(f"  U0 mean   : {qh['U0_mean'][-1]:.4f}  (Ising energy)")

    # --- Plot ---
    plot_experiment_stages(
        G=G, pos=pos,
        p_init=p_init_vis,
        p_classical=p_classical_vis,
        p_quantum=p_quantum_vis,
        spins_classical=spins_classical,
        spins_quantum=spins_quantum,
        cut_classical=cut_classical,
        cut_quantum=cut_quantum,
        N=N, edge_prob=edge_prob, graph_seed=graph_seed,
        exact_cut=exact_cut,
        acc_classical=acc_c,
        acc_quantum=acc_q,
        t_classical=t_classical,
        t_quantum=t_quantum,
        save_path=save_path,
    )
    if verbose:
        print(f"Plot saved → {save_path}")

    return ExperimentResult(
        cut_classical=cut_classical,
        cut_quantum=cut_quantum,
        gain=cut_quantum - cut_classical,
        exact_cut=exact_cut,
        acc_classical=acc_c,
        acc_quantum=acc_q,
        t_classical=t_classical,
        t_quantum=t_quantum,
        t_total=t_total,
        optimal_found=optimal_found,
    )


# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    run_experiment(
        N=15,
        edge_prob=0.4,
        weight_mode="ones",
        graph_seed=7,
        replicas=128,
        beta_min=1e-2,
        beta_max=40.0,
        beta_steps=300,
        lr_classical=0.04,
        gamma_max=1.5,
        gamma_min=0.0,
        gamma_steps=200,
        beta_q=60.0,
        lr_a=0.02,
        lr_theta=0.02,
        solver_seed=123,
        verify_exact=True,
        save_path="results/experiment_plot.png",
    )