# QeFEM paper dojo

This directory is the self-contained working surface for the paper. The numerical kernels remain in the validated `qefem_prof` repository; this directory contains the manuscript, the reproducible experiment harness, and compact paper-facing evidence.

## Primary files

- `qefem_paper.tex` and `qefem_paper.pdf`: revised manuscript and compiled PDF.
- `paper_experiments.ipynb`: one-button, resumable experiment notebook.
- `paper_experiments.py`: tested orchestration used by the notebook. It imports the existing QeFEM implementation and does not duplicate the algorithm.
- `results/`: normalized endpoint, trajectory, configuration, and Markdown evidence.

## Running the experiments

1. Start Jupyter with the existing `ml+torch` Conda environment.
2. Open `paper_experiments.ipynb`.
3. Use **Run All**.

The canonical settings are at the top of the notebook. `SMOKE=True` performs a cheap end-to-end check; `SMOKE=False` runs the paper protocol. `OVERWRITE=False` resumes safely and skips only cases containing the required method/trial evidence. CPU is the exact portable default. Apple MPS can be selected as a new platform run, but floating-point differences can change final cuts.

The full paper suite is intentionally substantial: six K2000 replica populations, ten Wishart instances, eleven Chook instances, every G-set graph from G1 through G54, and every supplied G-set file above G54. Each case is committed as soon as it finishes.

## Results layout

- `results/k2000/replicas-*`: nested population-size runs; `results/k2000/lqa` stores the baseline.
- `results/wishart/alpha-*`: one folder for each alpha.
- `results/chook-tile/p-c3-*`: one folder for each requested tile fraction.
- `results/gset/G*`: all 54 instances from G1 through G54.
- `results/gset-big/G*`: all supplied instances with an index above 54.

Canonical case files are `configuration.json`, `endpoints.csv`, `trajectory.csv`, `trajectory_lqa.csv` when applicable, `report.md`, and `manifest.json`. Trajectories contain only the discrete objective, reference gap/error, and entropy.

Already-completed laptop runs were normalized into this tree. The 22 K2000/Wishart/Chook trajectories were exact endpoint replays. Older 128-replica Adam G-set results are retained with an `imported_local_*` prefix and explicitly labeled as historical baselines; they do not suppress the newer Feng-tuned G-set experiments. Raw candidate populations were not copied into this paper workspace.

## Reproducibility boundary

The harness pins `qefem_prof` revision `b14f73c53305b55242b418ef2013fba0ac3cce54` by default. Set `STRICT_REVISION=False` only when intentionally starting a new experiment series. Every completed configuration records seeds, source plan, input hash, device, and dtype.
