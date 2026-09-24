"""Rebuild the detailed manuscript figures from saved, read-only results.

No solver is run here. Plot-ready CSV files are written alongside the figures.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
RESULTS = PAPER / "results"
GSET = (PAPER.parent / "qefem_prof" / "benchmarks" / "Gset" / "results"
        / "remaining_discrete" / "analysis" / "per_instance.csv")
FIGURES = HERE / "figures"
DATA = HERE / "plot_data"
TABLES = HERE / "tables"
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, GREY = "#2454A6", "#B84D20", "#237658", "#555555"


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    plt.close(fig)


# Five paired seeds at each nested QeFEM population size.
k = pd.read_csv(RESULTS / "k2000" / "endpoints.csv")
q = k[k.method == "qefem"].copy()
assert len(q) == 30 and q.groupby("replicas").size().eq(5).all()
population = q.groupby("replicas").cut.agg(["mean", "min", "max"]).reset_index()
population.to_csv(DATA / "k2000_population.csv", index=False)
lqa_cut = float(k.loc[k.method == "lqa", "cut"].mean())
fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.plot(population.replicas, population["mean"], "o-", color=BLUE, label="QeFEM trial mean")
ax.plot(population.replicas, population["max"], "s--", color=GREEN, label="QeFEM best of five")
ax.fill_between(population.replicas, population["min"], population["max"],
                color=BLUE, alpha=.11, label="range across five trials")
ax.axhline(lqa_cut, color=ORANGE, lw=1.4, ls="-.", label="LQA mean (one state per trial)")
ax.axhline(33337, color=GREY, lw=1.2, ls=":", label="stored reference: 33,337")
ax.set(xlabel="QeFEM replicas per trial", ylabel="Final binary cut", xscale="log")
ax.set_xticks(population.replicas, [f"{int(v):,}" for v in population.replicas])
ax.set_ylim(33235, 33347)
ax.grid(axis="y", alpha=.22)
ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=8)
save(fig, "k2000_population")


# One coherent QeFEM trajectory: the final winning replica from the best trial.
traj = pd.read_csv(RESULTS / "k2000" / "replicas-08192" / "trajectory.csv")
assert int(traj.step.max()) == 1000
traj.to_csv(DATA / "k2000_winner_trajectory.csv", index=False)
fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 5.5), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
a.semilogy(traj.step, traj.gap, color=BLUE, lw=1.35,
           label="reference gap of final winning replica")
a.axhline(44, color=GREEN, ls="--", lw=1.0, label="final gap: 44")
a.axvline(500, color=GREY, ls=":", lw=1.1, label="transverse field reaches zero")
a.set_ylabel("Stored-reference gap (log scale)")
a.legend(loc="upper right", fontsize=8)
a.grid(alpha=.2)
b.plot(traj.step, traj.entropy / 2000, color=ORANGE, lw=1.3)
b.axvline(500, color=GREY, ls=":", lw=1.1)
b.set(xlabel="Optimizer update", ylabel="Entropy per spin (nats)")
b.grid(alpha=.2)
save(fig, "k2000_winner_dynamics")


records = []
for family, stem in (("wishart", "alpha"), ("chook-tile", "p-c3")):
    for folder in sorted((RESULTS / family).glob(f"{stem}-*")):
        frame = pd.read_csv(folder / "endpoints.csv")
        parameter = float(folder.name.split("-")[-1])
        for method, part in frame.groupby("method"):
            records.append({"family": family, "parameter": parameter, "method": method,
                            "mean_error_percent": 100 * part.relative_error.mean(),
                            "mean_energy_excess": part.gap.mean(),
                            "ground_hits": int(part.ground_hit.sum()),
                            "trials": len(part)})
planted = pd.DataFrame(records).sort_values(["family", "parameter", "method"])
assert len(planted) == 42
planted.to_csv(DATA / "planted_instance_means.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), sharex="col",
                         gridspec_kw={"height_ratios": [2, 1]})
for column, (family, title, xlabel) in enumerate([
    ("wishart", "Wishart planted instances", r"Requested $\alpha$"),
    ("chook-tile", "Chook tile-planted instances", r"Requested $p(C_3)$"),
]):
    ax, hit_ax = axes[:, column]
    sub = planted[planted.family == family]
    for method, color, marker, label in (("qefem", BLUE, "o", "QeFEM"),
                                          ("lqa", ORANGE, "s", "LQA")):
        part = sub[sub.method == method]
        ax.plot(part.parameter, part.mean_error_percent, marker=marker,
                color=color, lw=1.6, ms=4, label=label)
        hit_ax.plot(part.parameter, part.ground_hits / part.trials, marker=marker,
                    color=color, lw=1.6, ms=4)
    ax.set(title=title, ylabel="Mean relative energy error (%)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    hit_ax.set(xlabel=xlabel, ylabel="Exact-hit fraction", ylim=(-.04, 1.04))
    hit_ax.set_yticks([0, .5, 1])
    hit_ax.grid(alpha=.2)
fig.tight_layout(h_pad=1.0)
save(fig, "planted_instance_errors")


gset = pd.read_csv(GSET)
assert len(gset) == 54 and int((gset.combined_gap == 0).sum()) == 33
gset[["instance", "target", "combined_best", "combined_gap", "status"]].to_csv(
    DATA / "gset_54_attainment.csv", index=False)
# The main G-set figure shows what the later search added, rather than only
# displaying the still-unmatched instances.
searched = gset[gset.previous_gap > 0].copy()
assert len(searched) == 31
assert int((searched.combined_gap < searched.previous_gap).sum()) == 27
fig, (a, b) = plt.subplots(1, 2, figsize=(10.0, 4.0),
                            gridspec_kw={"width_ratios": [3, 2]})
limits = [0, max(searched.previous_gap.max(), searched.combined_gap.max()) * 1.05]
a.plot(limits, limits, color=GREY, ls=":", lw=1.2, label="no change")
new_hits = searched.combined_gap.eq(0)
a.scatter(searched.loc[~new_hits, "previous_gap"],
          searched.loc[~new_hits, "combined_gap"],
          color=BLUE, s=37, label="remaining gaps")
a.scatter(searched.loc[new_hits, "previous_gap"],
          searched.loc[new_hits, "combined_gap"],
          color=GREEN, marker="D", s=49, label="new reference match")
a.set(xlabel="Gap before later search", ylabel="Retained gap after search",
      xlim=limits, ylim=(-.7, limits[1]))
a.grid(alpha=.2)
a.legend(fontsize=8, loc="upper left")
segments = [(23, BLUE, "23 earlier"), (8, GREEN, "8 fresh"),
            (2, "#78A98F", "2 search"), (21, "#DDDDDD", "21 open")]
left = 0
for count, color, label in segments:
    b.barh([0], [count], left=left, height=.42, color=color)
    if count > 2:
        b.text(left + count / 2, 0, str(count), ha="center", va="center",
               fontsize=10, color="white" if label != "21 open" else "#222222")
    else:
        b.annotate("2", xy=(left + count / 2, .22),
                   xytext=(left + count / 2, .42), ha="center", fontsize=9,
                   arrowprops={"arrowstyle": "-", "color": GREY, "lw": .7})
    left += count
b.set(xlim=(0, 54), ylim=(-.55, .85), xlabel="G1--G54 reference status")
b.legend(handles=[Patch(facecolor=color, label=label.split(" ", 1)[1])
                  for _, color, label in segments],
         loc="upper left", bbox_to_anchor=(0, .98), ncol=2,
         fontsize=7.5, frameon=False)
b.set_yticks([])
b.spines["left"].set_visible(False)
b.tick_params(axis="y", length=0)
fig.tight_layout()
save(fig, "gset_search_progress")


# Context only: exact maxima printed in the archived FEM/dSB notebook output.
# Keep these as an auditable table, not a plotted comparison with QeFEM:
# the saved QeFEM evaluation used a separate device and protocol.
context = pd.DataFrame([
    (100, 33133, 32993), (200, 33214, 33155),
    (400, 33275, 33248), (600, 33296, 33285),
    (800, 33296, 33272), (1000, 33306, 33304),
    (2000, 33332, 33333), (4000, 33337, 33337),
], columns=["steps", "fem_max", "dsb_max"])
context.to_csv(DATA / "k2000_archived_fem_dsb_maxima.csv", index=False)

def latex_rows(path, rows):
    path.write_text("\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
                    + "\n\\bottomrule\n")


trial_rows = []
k8 = q[q.replicas == 8192].sort_values("trial")
lqa = k[k.method == "lqa"].sort_values("trial")
for qr, lr in zip(k8.itertuples(), lqa.itertuples()):
    assert qr.seed == lr.seed
    trial_rows.append((qr.trial + 1, int(qr.seed), int(qr.cut), int(qr.gap),
                       int(lr.cut), int(lr.gap)))
latex_rows(TABLES / "k2000_trials.tex", trial_rows)

for family, name in (("wishart", "wishart"), ("chook-tile", "chook")):
    rows = []
    sub = planted[planted.family == family]
    for parameter, group in sub.groupby("parameter"):
        qr = group[group.method == "qefem"].iloc[0]
        lr = group[group.method == "lqa"].iloc[0]
        rows.append((f"{parameter:.1f}", f"{qr.mean_energy_excess:.2f}",
                     f"{lr.mean_energy_excess:.2f}",
                     f"{qr.mean_error_percent:.3f}", f"{lr.mean_error_percent:.3f}",
                     f"{qr.ground_hits}/{qr.trials}", f"{lr.ground_hits}/{lr.trials}"))
    latex_rows(TABLES / f"{name}_instances.tex", rows)

gset_rows = []
for row in gset.sort_values("instance", key=lambda s: s.str[1:].astype(int)).itertuples():
    status = {"previous": "prior", "confirmed": "fresh",
              "discovery_only": "search", "unresolved": "open"}[row.status]
    gset_rows.append((row.instance, int(row.target), int(row.combined_best),
                      int(row.combined_gap), status))
latex_rows(TABLES / "gset_54.tex", gset_rows)
assert len(gset_rows) == 54
latex_rows(TABLES / "gset_54_paired.tex",
           [gset_rows[i] + gset_rows[i + 27] for i in range(27)])

print("Wrote four figures, plot data, and result tables to", HERE)
