# Detailed QeFEM manuscript

This directory is a separate, editable working version of the paper. It uses
the first paper's RevTeX 4.2 `aps, prx, reprint` template and two-column main
text; the detailed appendices switch to one column for full-width derivations
and numerical tables. The original files in `Paper/` are not part of its
build. The manuscript develops
the variational free energy, Bloch-state derivatives, stationary equations,
annealing and discrete-neighbour dynamics, then interprets the saved G-set,
K2000, Wishart, and Chook evaluations. It records the K2000 gap and the
differences between comparison protocols explicitly.

The introduction states MaxCut as an optimization over vertex partitions.
The experiment section distinguishes the saved Wishart and Chook generation
schemes and fixed files. Result tables containing cut endpoints span the full
available text width.

## Files

- `qefem_detailed.tex`: LaTeX entry point.
- `sections/`: manuscript and appendices.
- `references.bib`: cited literature.
- `make_figures.py`: reads saved result files and regenerates all four figures,
  plot-ready CSVs, and appendix table rows. It does **not** run a solver.
- `figures/`: vector PDF figures and PNG previews.
- `plot_data/`: the exact data used to construct the figures, including an
  archived FEM/dSB K2000 maxima table for contextual auditing.
- `tables/`: generated LaTeX rows for the full result tables.
- `qefem_detailed.pdf`: compiled manuscript.

The figure data are read from `Paper/results/` and the G-set retained-result
file `qefem_prof/benchmarks/Gset/results/remaining_discrete/analysis/per_instance.csv`.
The archived FEM/dSB maxima were transcribed from saved output in
`FEM/benchmarks/maxcut/k2000_results.ipynb`; they are shown as context in
Appendix B, not as a matched-compute three-way result.

## Rebuild

From this directory, with the project Python environment and TeX Live
available:

```sh
python make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build qefem_detailed.tex
cp build/qefem_detailed.pdf qefem_detailed.pdf
```

The four figures have distinct evidentiary purposes: G-set attainment from
later search, K2000 population scaling, a coherent K2000 winning-replica
trajectory, and planted-instance error plus exact-hit frequency. Their
captions state each figure's unit of analysis and limits.
