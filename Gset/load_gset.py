import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def resolve_instance_path(test_instance):
    path = Path(test_instance)
    if path.exists():
        return path

    path = Path("Gset") / test_instance
    if path.exists():
        return path

    raise FileNotFoundError(f"Could not find graph instance: {test_instance}")


def load_file_sparse(
    test_instance,
    plot=False,
    output_dir=None,
    plot_mode="combined",
    graph_type=None,
    edge_width=0.18,
):
    G = nx.Graph()
    path = resolve_instance_path(test_instance)

    with open(path, "r") as f:
        N, m = map(int, f.readline().split())

        G.add_nodes_from(range(N))
        sum_w = 0

        for _ in range(m):
            i, j, w = map(int, f.readline().split())
            G.add_edge(i - 1, j - 1, weight=-w)
            sum_w += w

        sum_w = 0.5 * sum_w

    if plot and plot_mode == "combined":
        plot_graph_diagnostics(G, path.name, output_dir=output_dir)
    elif plot and plot_mode == "full":
        plot_full_graph(G, path.name, graph_type=graph_type, output_dir=output_dir, edge_width=edge_width)

    J = build_sparse_tensor(G) if torch is not None else None
    return G, J, sum_w


def build_sparse_tensor(G):
    crow_indices = [0]
    col_indices = []
    values = []

    for node in G.nodes:
        for neighbor in G.neighbors(node):
            col_indices.append(neighbor)
            values.append(G.edges[node, neighbor]["weight"])
        crow_indices.append(len(values))

    crow_indices = torch.tensor(crow_indices, dtype=torch.int64)
    col_indices = torch.tensor(col_indices, dtype=torch.int64)
    values = torch.tensor(values, dtype=torch.float32)

    return torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(G.number_of_nodes(), G.number_of_nodes()),
        dtype=torch.float32,
    )


def plot_graph_diagnostics(G, name, output_dir=None, sample_size=2000):
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    degrees = np.array([d for _, d in G.degree()])
    print("Graph:", name)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Average degree:", degrees.mean())
    print("Minimum degree:", degrees.min())
    print("Maximum degree:", degrees.max())

    edge_list = list(G.edges())
    sampled_edges = random.Random(42).sample(edge_list, min(sample_size, len(edge_list)))

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(sampled_edges)

    pos = graph_layout(G, H)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    draw_edges(axes[0], H, pos, f"Sampled edges: {name}", edge_alpha=0.08)
    draw_edges(axes[1], G, pos, f"Full graph edges: {name}", edge_alpha=0.025)

    fig.tight_layout()
    if output_path:
        plt.savefig(output_path / f"{name}_sampled_and_full_edges.png", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def graph_layout(G, sampled_graph):
    if looks_toroidal(G):
        return grid_layout(G)
    return nx.spring_layout(sampled_graph, seed=42, iterations=80)


def looks_toroidal(G):
    edge_ratio = G.number_of_edges() / max(G.number_of_nodes(), 1)
    return 1.8 <= edge_ratio <= 2.2


def grid_layout(G):
    n = G.number_of_nodes()
    cols = math.ceil(math.sqrt(n))
    return {node: (node % cols, -(node // cols)) for node in G.nodes()}


def draw_edges(ax, G, pos, title, edge_alpha):
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=8, alpha=0.8, linewidths=0)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.25, alpha=edge_alpha)
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")


def plot_full_graph(G, name, graph_type=None, output_dir=None, edge_width=0.18):
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    graph_type = graph_type or "unknown"
    pos = typed_layout(G, graph_type)
    edge_alpha = full_edge_alpha(G.number_of_edges())
    node_size = full_node_size(G.number_of_nodes())
    title = f"{name} {graph_type} edge# {G.number_of_edges()} node# {G.number_of_nodes()}"

    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, alpha=edge_alpha, edge_color="#111827")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, alpha=0.9, linewidths=0, node_color="#2563eb")
    ax.set_title(title, pad=14)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")

    if output_path:
        plt.savefig(output_path / f"{name}_{graph_type}_full_edges.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def typed_layout(G, graph_type):
    graph_type = graph_type.lower()
    if graph_type == "toroidal":
        return rectangular_grid_layout(G)
    if graph_type == "planar":
        if G.number_of_edges() <= 3 * G.number_of_nodes() - 6:
            is_planar, _ = nx.check_planarity(G)
            if is_planar:
                return nx.planar_layout(G)
        print("Planar embedding unavailable; using spring layout for:", G.number_of_nodes(), G.number_of_edges())
    return nx.spring_layout(G, seed=42, iterations=80)


def rectangular_grid_layout(G):
    n = G.number_of_nodes()
    rows = int(math.sqrt(n))
    while rows > 1 and n % rows != 0:
        rows -= 1
    cols = math.ceil(n / rows)
    return {node: (idx % cols, -(idx // cols)) for idx, node in enumerate(sorted(G.nodes()))}


def full_edge_alpha(edge_count):
    if edge_count >= 20000:
        return 0.018
    if edge_count >= 10000:
        return 0.025
    if edge_count >= 5000:
        return 0.04
    return 0.08


def full_node_size(node_count):
    if node_count >= 3000:
        return 2
    if node_count >= 1500:
        return 3
    return 5


def parse_graph_types(entries):
    graph_types = {}
    for entry in entries:
        name, separator, graph_type = entry.partition("=")
        if not separator:
            raise ValueError(f"Graph type entry must look like G10=planar: {entry}")
        graph_types[name] = graph_type
    return graph_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot-mode", choices=("combined", "full"), default="combined")
    parser.add_argument("--graph-types", nargs="*", default=[])
    parser.add_argument("--edge-width", type=float, default=0.18)
    args = parser.parse_args()
    graph_types = parse_graph_types(args.graph_types)

    for instance in args.instances:
        name = resolve_instance_path(instance).name
        _, J, _ = load_file_sparse(
            instance,
            plot=True,
            output_dir=args.output_dir,
            plot_mode=args.plot_mode,
            graph_type=graph_types.get(name),
            edge_width=args.edge_width,
        )

        if args.device and J is None:
            raise RuntimeError("Cannot move sparse tensor to a device because torch is not installed.")
        if args.device:
            J = J.to(args.device)

        print(J.device if J is not None else "torch unavailable; skipped sparse tensor creation")


if __name__ == "__main__":
    main()
