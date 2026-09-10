# Solver-independent Wishart and Chook tile benchmark package

This package contains deterministic planted Ising instances for comparing independent implementations of QeFEM and LQA. The saved `.npz` files are solver-independent: they contain only problem data, a planted spin configuration, the corresponding ground-state energy, and generation metadata. They do not encode an optimizer, annealing schedule, replica count, device, or solver state.

The original LQA paper did not publish its exact matrices or random seeds. These banks follow the reported benchmark sizes and problem distributions, but they are not byte-for-byte copies of the paper's private instances.

## Package contents

```text
generators/
  wishart_generator.py       Existing Wishart generator; also contains LQA helpers
  chook_tile_generator.py    Existing C2/C3 tile generator
  local_lqa.py               Import dependency of wishart_generator.py
  coupling.py                Import dependency of both generator modules
  requirements-generation.txt

ready_to_use/
  wishart_n500/
    instance_00.npz ... instance_09.npz
    generation_settings.json
  chook_tile_l32/
    instance_00.npz ... instance_10.npz
    generation_settings.json

SHA256SUMS.txt                Integrity hashes for all distributed files
EMAIL_DRAFT.txt               Suggested covering email
```

`local_lqa.py` and `coupling.py` are included only because the existing generator modules import them. A recipient does not need either file to load or solve the already generated `.npz` instances. Keep the `generators` directory separate from the recipient's QeFEM source tree to avoid shadowing similarly named local modules.

## Environment for regeneration

Python 3.10 or newer is recommended. In a clean virtual environment:

```bash
python3 -m venv instance-env
source instance-env/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch
```

Only NumPy is needed to load and score the saved instance files. PyTorch is needed to import the existing generator modules because those files also contain optional solver adapters.

## Use the ready-to-use Wishart instances

The Wishart bank contains 10 fully connected instances with 500 spins. The requested hardness control is `alpha = M/N = 0.1, 0.2, ..., 1.0`. Instance seeds are `2026, ..., 2035`, and a random gauge hides the simple planted state.

Each file contains:

- `couplings`: dense `float64` symmetric matrix `K`, shape `(500, 500)`, with zero diagonal.
- `planted_spins`: `int8` vector in `{−1,+1}`, shape `(500,)`.
- `ground_energy`: exact planted ground energy `E0` for the stored convention.
- `alpha_requested`, `M`, `seed`, and `gauge`: generation metadata.

Load and verify one file with NumPy:

```python
import numpy as np

with np.load("ready_to_use/wishart_n500/instance_00.npz", allow_pickle=False) as data:
    K = data["couplings"].copy()
    planted = data["planted_spins"].copy()
    E0 = float(data["ground_energy"])
    alpha = float(data["alpha_requested"])

assert np.allclose(K, K.T)
assert np.allclose(np.diag(K), 0.0)
assert np.isclose(0.5 * planted @ K @ planted, E0)
```

The objective to minimize is

```text
E(s) = 0.5 * s.T @ K @ s = sum_(i<j) K[i,j] s[i] s[j],  s_i in {-1,+1}.
```

Pass `K` directly if the solver accepts a symmetric, zero-diagonal Ising matrix and internally applies the factor `0.5`. If the solver accepts each undirected interaction once, pass only the upper-triangle entries. Do not negate or rescale the stored couplings. If the recipient's solver defines its Hamiltonian with a leading minus sign, adapt its input boundary so that its reported score still equals the equation above.

## Generate a new Wishart bank

Do not execute `python wishart_generator.py` merely to generate matrices. That command-line entry point also launches the full LQA experiment. Instead, run the following as a Jupyter cell or Python input from inside `generators/`:

```python
from pathlib import Path
import json
from wishart_generator import generate_wishart

output = Path("../generated/wishart_n500_seed2026")
output.mkdir(parents=True, exist_ok=False)

alphas = [i / 10 for i in range(1, 11)]
seeds = [2026 + i for i in range(len(alphas))]

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

The output directory must not already exist. Change the first seed to create an independent bank. Keep the same `alpha` grid and spin count when comparing with this experiment design.

## Use the ready-to-use Chook C2/C3 tile instances

The tile bank contains 11 independent periodic `32 x 32` square lattices. Each has 1,024 spins, 2,048 undirected nearest-neighbour bonds, and 512 planted four-spin tiles. Requested `p(C3)` increases from `0.0` to `1.0` in steps of `0.1`; all remaining tiles are C2.

Each file contains:

- `edge_i`, `edge_j`: zero-based endpoint arrays; each undirected edge occurs once.
- `edge_w`: `float64` Ising weights aligned with the endpoint arrays.
- `planted_spins`: `int8` vector in `{−1,+1}`, shape `(1024,)`.
- `ground_energy`: exact planted ground energy `E0`.
- `c3_probability`, `c2_tiles`, `c3_tiles`, `lattice_size`, `seed`, and `gauge`: generation metadata.

Load and verify a sparse instance:

```python
import numpy as np

with np.load("ready_to_use/chook_tile_l32/instance_00.npz", allow_pickle=False) as data:
    edge_i = data["edge_i"].copy()
    edge_j = data["edge_j"].copy()
    edge_w = data["edge_w"].copy()
    planted = data["planted_spins"].copy()
    E0 = float(data["ground_energy"])
    p_c3 = float(data["c3_probability"])

planted_energy = np.sum(edge_w * planted[edge_i] * planted[edge_j])
assert np.isclose(planted_energy, E0)
```

The objective to minimize is

```text
E(s) = sum_e edge_w[e] * s[edge_i[e]] * s[edge_j[e]].
```

If the solver needs a dense symmetric matrix:

```python
n = planted.size
K = np.zeros((n, n), dtype=np.float64)
K[edge_i, edge_j] = edge_w
K[edge_j, edge_i] = edge_w
assert np.isclose(0.5 * planted @ K @ planted, E0)
```

## Generate a new Chook tile bank

From inside `generators/`, run:

```bash
python chook_tile_generator.py \
  --output ../generated/chook_tile_l32_seed32000 \
  --lattice-size 32 \
  --seed 32000
```

This produces 11 files for `p(C3) = 0.0, 0.1, ..., 1.0` plus `generation_settings.json`. The output directory must not already exist. Change `--seed` to make a new independent bank. The generator is a deterministic port of Chook's C2/C3 construction and gauge wrapper. It samples the same construction, but its NumPy `Generator` seeds are not expected to reproduce official Chook CLI output byte for byte.

## Connect either format to another QeFEM implementation

The recipient should adapt only the loading boundary:

1. Load one fixed `.npz` file.
2. Convert its dense matrix or edge list into the coupling type accepted by that QeFEM version.
3. Minimize the stored Ising objective using the solver's own code.
4. Convert every returned candidate to a binary spin vector in `{−1,+1}`.
5. Rescore candidates with the NumPy equations above, rather than trusting a solver-specific sign or normalization.
6. For a replica trial, select the final replica with the minimum rescored energy.
7. Report `abs(E_best - E0) / abs(E0)` and whether `E_best` reaches `E0` within a declared numerical tolerance.

The planted state provides a mandatory compatibility test. Before a real run, the recipient should load one file and confirm that its exact scorer returns the stored `ground_energy` for `planted_spins`. Failure means the solver boundary has a sign, factor-of-two, indexing, or duplicate-edge mismatch.

For the existing head-to-head protocol, use 500 fresh trials per fixed instance. LQA uses one fresh initialization and 500 updates per trial. QeFEM uses 128 fresh replicas and 500 steps per trial, returns the lowest-energy final replica, and starts the next trial from a fresh state. Do not continue a final state, replica population, or optimizer state into the next trial. Record direct wall-clock time per call separately from solution quality.

The two algorithms do not receive the same number of candidate states per trial because QeFEM is intrinsically replica based. The comparison should therefore show both solution quality and actual wall-clock time, with replica count stated explicitly.

## Integrity check

From the package root:

```bash
shasum -a 256 -c SHA256SUMS.txt
```

All lines should report `OK`. The hashes let both parties confirm that they used the same generator source and the same instance bank.