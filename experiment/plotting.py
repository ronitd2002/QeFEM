# experiments/plotting.py
# Pure visualization. Takes tensors, draws graphs, saves figures.
# No solver logic, no torch.optim, no problem definitions.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch


def build_nx_layout(W: torch.Tensor, layout_seed: int = 42):
    """Build a NetworkX graph and spring layout from adjacency matrix."""
    W_cpu = W.detach().cpu().numpy()
    N = W_cpu.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if W_cpu[i, j] > 0:
                G.add_edge(i, j, weight=float(W_cpu[i, j]))
    pos = nx.spring_layout(G, seed=layout_seed)
    return G, pos


def draw_graph_stage(ax, G, pos, p: torch.Tensor, title: str, best_spins=None):
    """
    Draw one stage panel.
    - Nodes coloured by marginal p_i (blue=0, red=1, white=0.5 undecided)
    - Cut edges drawn thick orange if best_spins provided
    """
    p_np = p.detach().cpu().numpy()
    cmap = plt.cm.RdBu_r

    node_colors = [cmap(float(p_np[i])) for i in range(len(p_np))]

    edge_colors, edge_widths = [], []
    for u, v in G.edges():
        if best_spins is not None:
            s = best_spins.detach().cpu()
            is_cut = (s[u] * s[v] < 0)
            edge_colors.append("darkorange" if is_cut else "#cccccc")
            edge_widths.append(2.5 if is_cut else 0.8)
        else:
            edge_colors.append("#cccccc")
            edge_widths.append(0.8)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=300,
                           edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(
        G, pos,
        labels={i: f"{p_np[i]:.2f}" for i in range(len(p_np))},
        ax=ax, font_size=5,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def plot_experiment_stages(
    G,
    pos,
    p_init: torch.Tensor,
    p_classical: torch.Tensor,
    p_quantum: torch.Tensor,
    spins_classical: torch.Tensor,
    spins_quantum: torch.Tensor,
    cut_classical: float,
    cut_quantum: float,
    N: int,
    edge_prob: float,
    graph_seed: int,
    exact_cut: float = None,
    acc_classical: float = None,
    acc_quantum: float = None,
    t_classical: float = None,
    t_quantum: float = None,
    save_path: str = "experiment_plot.png",
) -> plt.Figure:
    """
    Three-panel figure: init | classical | quantum.
    Returns the figure object (caller can close or display).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"MaxCut — Erdős-Rényi  N={N}, p={edge_prob}, seed={graph_seed}\n"
        "Orange edges = cut edges of best replica",
        fontsize=11,
    )

    draw_graph_stage(axes[0], G, pos, p=p_init,
                     title="Stage 0: Random init\n(marginals ≈ 0.5)")

    t_c_str = f"  [{t_classical:.2f}s]" if t_classical is not None else ""
    t_q_str = f"  [{t_quantum:.2f}s]" if t_quantum is not None else ""

    draw_graph_stage(axes[1], G, pos, p=p_classical,
                     title=f"Stage 1: Classical FEM{t_c_str}\ncut = {cut_classical:.4f}",
                     best_spins=spins_classical)

    draw_graph_stage(axes[2], G, pos, p=p_quantum,
                     title=f"Stage 2: Quantum MF{t_q_str}\ncut = {cut_quantum:.4f}",
                     best_spins=spins_quantum)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Node marginal p_i  (0=blue, 1=red)", fontsize=8)

    # Footer with accuracy
    if exact_cut is not None:
        footer = (
            f"Exact = {exact_cut:.4f}  |  "
            f"Classical {acc_classical*100:.1f}%  |  "
            f"Quantum {acc_quantum*100:.1f}%"
        )
        fig.text(0.5, 0.01, footer, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig