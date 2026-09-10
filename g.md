# Reproducible planted-Ising benchmark package

## Wishart and Chook C2/C3 tile ensembles for QeFEM–LQA comparison

### 1. Purpose

This package provides a fixed set of planted Ising instances for evaluating independent implementations of QeFEM and LQA under a common objective. The principal deliverable is the `instances/` directory. Its `.npz` files contain only problem coefficients, a planted binary configuration, the corresponding ground-state energy, and generation metadata. They contain no optimizer state, annealing schedule, learned parameters, replica state, or solver-specific object.

The exact instances and random seeds used for the corresponding figures in the LQA paper were not published. The supplied banks reproduce the reported problem families and experiment sizes using deterministic local implementations of the Chook constructions. They should therefore be described as new instances from the same benchmark families, rather than reproductions of the authors’ unpublished matrices.

### 2. Package inventory

| Location | Contents | Required for solving the supplied instances? |
|---|---|---:|
| `instances/wishart_n500/` | Ten dense, 500-spin Wishart-planted Ising instances | Yes |
| `instances/chook_tile_l32/` | Eleven sparse, 1,024-spin C2/C3 tile-planted Ising instances | Yes |
| `generation_source/wishart_generator.py` | Existing Wishart generator and local LQA experiment wrapper | Only for regeneration |
| `generation_source/chook_tile_generator.py` | Existing deterministic C2/C3 tile generator | Only for regeneration |
| `generation_source/local_lqa.py` | Import dependency of `wishart_generator.py` | Only for importing that combined module |
| `generation_source/coupling.py` | Shared tensor edge-list definition used by the existing modules | Only for importing the generator modules |
| `generation_source/requirements-generation.txt` | Python dependencies for importing the existing modules | Only for regeneration |
| `SHA256SUMS.txt` | File-integrity manifest | Recommended |

The ZIP does contain the local LQA implementation in `generation_source/local_lqa.py`. It is included because the existing `wishart_generator.py` imports it at module load time and combines two responsibilities in one source file:

1. `generate_wishart()` constructs a Wishart instance.
2. `run_experiment()` and `main()` run LQA trials on generated instances.

Consequently, executing `python wishart_generator.py` invokes `main()` and performs both generation and the full LQA experiment. For generation alone, import and call `generate_wishart()` as shown in Section 6. `local_lqa.py` is an import dependency of that combined module; it is not required to load the `.npz` files and should not be incorporated into the recipient’s QeFEM implementation. All supplied Python source files are unchanged copies from the working repository.

### 3. Benchmark banks

| Family | Instances | Spins | Connectivity | Control parameter | Values |
|---|---:|---:|---|---|---|
| Wishart planting | 10 | 500 | Fully connected | `alpha = M/N` | 0.1 to 1.0 in increments of 0.1 |
| Chook C2/C3 tile planting | 11 | 1,024 | 32×32 periodic square lattice; 2,048 bonds | Requested `p(C3)` | 0.0 to 1.0 in increments of 0.1 |

Both banks use a random spin gauge. The gauge conceals the simple ungauged planted state while preserving the complete energy spectrum. Every file contains the gauged planted configuration and its exact energy.

### 4. Wishart data contract

Each `instances/wishart_n500/instance_XX.npz` file contains:

| Key | Type and shape | Definition |
|---|---|---|
| `couplings` | `float64`, `(500,500)` | Symmetric coupling matrix `K` with zero diagonal |
| `planted_spins` | `int8`, `(500,)` | Planted configuration with entries in `{−1,+1}` |
| `ground_energy` | scalar | Exact energy of `planted_spins` under the stored convention |
| `alpha_requested` | scalar | Requested Wishart ratio `M/N` |
| `M` | scalar integer | Number of columns in the underlying Gaussian matrix |
| `seed` | scalar integer | Instance-generation seed |
| `gauge` | scalar Boolean | Whether a random spin gauge was applied |

The objective is

```text
E(s) = 0.5 sᵀKs = Σ_(i<j) K_ij s_i s_j,    s_i ∈ {−1,+1}.
```

Reference loading and validation:

```python
import numpy as np

with np.load("instances/wishart_n500/instance_00.npz", allow_pickle=False) as data:
    K = data["couplings"].copy()
    planted = data["planted_spins"].copy()
    E0 = float(data["ground_energy"])
    alpha = float(data["alpha_requested"])

assert K.shape == (500, 500)
assert np.allclose(K, K.T)
assert np.allclose(np.diag(K), 0.0)
assert np.isclose(0.5 * planted @ K @ planted, E0)
```

A solver that accepts a symmetric Ising matrix should receive `K` with the corresponding factor of `0.5` in its energy evaluation. A solver that accepts each undirected interaction once should receive the strict upper triangle. The matrix must not be negated or rescaled unless the solver’s input convention is explicitly transformed so that its externally rescored energy still equals the equation above.

### 5. Chook tile data contract

Each `instances/chook_tile_l32/instance_XX.npz` file contains:

| Key | Type and shape | Definition |
|---|---|---|
| `edge_i`, `edge_j` | integer, `(2048,)` | Zero-based endpoints; every undirected bond appears once |
| `edge_w` | `float64`, `(2048,)` | Ising weight for each stored bond |
| `planted_spins` | `int8`, `(1024,)` | Planted configuration with entries in `{−1,+1}` |
| `ground_energy` | scalar | Exact planted ground-state energy |
| `c3_probability` | scalar | Requested probability of selecting a C3 tile |
| `c2_tiles`, `c3_tiles` | scalar integers | Realized tile counts |
| `lattice_size` | scalar integer | Linear lattice size; 32 in this bank |
| `seed` | scalar integer | Instance-generation seed |
| `gauge` | scalar Boolean | Whether a random spin gauge was applied |

The sparse objective is

```text
E(s) = Σ_e edge_w[e] s[edge_i[e]] s[edge_j[e]].
```

Reference loading and validation:

```python
import numpy as np

with np.load("instances/chook_tile_l32/instance_00.npz", allow_pickle=False) as data:
    edge_i = data["edge_i"].copy()
    edge_j = data["edge_j"].copy()
    edge_w = data["edge_w"].copy()
    planted = data["planted_spins"].copy()
    E0 = float(data["ground_energy"])
    p_c3 = float(data["c3_probability"])

assert np.isclose(np.sum(edge_w * planted[edge_i] * planted[edge_j]), E0)
```

For a solver requiring a dense symmetric matrix:

```python
n = planted.size
K = np.zeros((n, n), dtype=np.float64)
K[edge_i, edge_j] = edge_w
K[edge_j, edge_i] = edge_w
assert np.isclose(0.5 * planted @ K @ planted, E0)
```

### 6. Regenerating the Wishart bank

Python 3.10 or newer is recommended. From `generation_source/`, prepare an isolated environment:

```bash
python3 -m venv instance-env
source instance-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-generation.txt
```

For matrix generation without running LQA, execute the following in Python or a Jupyter cell while `generation_source/` is the working directory:

```python
from pathlib import Path
import json
from wishart_generator import generate_wishart

output = Path("../generated/wishart_n500_seed2026")
output.mkdir(parents=True, exist_ok=False)

alphas = [i / 10 for i in range(1, 11)]
seeds = [2026 + i for i in range(10)]

for index, (alpha, seed) in enumerate(zip(alphas, seeds)):
    instance = generate_wishart(
        n=500,
        alpha=alpha,
        seed=seed,
        gauge=True,
    )
    instance.save(output / f"instance_{index:02d}.npz")

metadata = {
    "problem_family": "continuous Wishart planting",
    "spins": 500,
    "alphas": alphas,
    "instance_seeds": seeds,
    "gauge": True,
    "energy_convention": "E(s) = 0.5 * s.T @ K @ s",
}
(output / "generation_settings.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)
```

This reproduces the supplied Wishart bank. To generate an independent bank, change the seed sequence while retaining the 500-spin size and the same `alpha` grid. The output directory must be new because the generator deliberately refuses to overwrite an existing directory.

### 7. Regenerating the Chook C2/C3 tile bank

From `generation_source/`, using the same environment:

```bash
python chook_tile_generator.py \
  --output ../generated/chook_tile_l32_seed32000 \
  --lattice-size 32 \
  --seed 32000
```

This command produces 11 independent periodic lattices for requested `p(C3) = 0.0, 0.1, ..., 1.0` and records the derived instance seeds in `generation_settings.json`. Changing `--seed` produces an independent bank.

The local generator is a deterministic implementation of Chook’s two-dimensional C2/C3 tile construction and gauge transformation. It uses NumPy’s current `Generator` interface; consequently, its numerical seeds are not expected to reproduce files generated by the official Chook command-line implementation byte for byte.

### 8. Integration with an independent QeFEM implementation

Only the solver’s input adapter should depend on its local codebase. The recommended procedure is:

1. Load a fixed `.npz` instance.
2. Convert the stored dense matrix or sparse edge list to the local coupling representation.
3. Confirm that the local exact scorer assigns `planted_spins` the stored `ground_energy`.
4. Execute QeFEM using the desired local implementation.
5. Convert every returned candidate to a vector in `{−1,+1}`.
6. Recompute candidate energies with the NumPy expressions in Sections 4 and 5.
7. Select the final replica with minimum rescored energy.
8. Report

```text
relative error = |E_best − E0| / |E0|.
```

The planted-state check is required before a benchmark run. A mismatch identifies an interface error involving sign, normalization, indexing, diagonal terms, or duplicate edges. It should be resolved at the input boundary rather than by modifying the distributed instance files.

### 9. Comparative experiment protocol

| Setting | LQA | QeFEM |
|---|---:|---:|
| Independent trials per fixed instance | 500 | 500 |
| Optimization steps per trial | 500 | 500 |
| Replicas per trial | 1 | 128 |
| Initial condition for each trial | Fresh | Fresh population of 128 replicas |
| State carried between trials | None | None |
| Reported configuration | Final LQA configuration | Lowest-energy final QeFEM replica |

QeFEM’s use of 128 replicas is part of its implementation budget. The comparison should therefore report both solution quality and measured wall-clock time, together with device, dtype, replica count, optimizer, schedules, and all numerical hyperparameters. Replica-average energy is not the head-to-head QeFEM score; only the best final replica is compared with the final LQA configuration.

### 10. Recommended result record

For every trial, retain:

- instance filename and SHA-256 hash;
- solver name and code revision;
- random seed;
- number of steps and replicas;
- complete solver hyperparameters;
- device and numerical dtype;
- direct wall-clock time;
- returned binary configuration;
- externally rescored energy;
- relative error; and
- ground-state-hit indicator and numerical tolerance.

This record is sufficient to compare independent implementations without requiring either party to adopt the other party’s solver code.

### 11. Integrity verification

From the package root:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

Every distributed file should report `OK`.