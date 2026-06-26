import argparse
import gzip
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


def resolve_instance_path(instance):
    path = Path(instance)
    if path.exists():
        return path

    path = Path("data-gset") / instance
    if path.exists():
        return path

    path = Path("data-gz") / f"{instance}.gz"
    if path.exists():
        return path

    path = Path("data-gz") / instance
    if path.exists():
        return path

    raise FileNotFoundError(f"Could not find graph instance: {instance}")


def _open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def read_gset_edges(instance):
    """
    Read a G-set instance in the format:
        N M
        i j w
    Node labels are converted from 1-indexed to 0-indexed.
    """
    path = resolve_instance_path(instance)
    with _open_text(path) as f:
        num_nodes, num_edges = map(int, f.readline().split())
        edges = []
        for _ in range(num_edges):
            i, j, w = f.readline().split()
            edges.append((int(i) - 1, int(j) - 1, float(w)))
    return num_nodes, edges


def load_gset_dense(instance, device="cpu"):
    if torch is None:
        raise RuntimeError("PyTorch is required to build a dense G-set tensor.")

    num_nodes, edges = read_gset_edges(instance)
    W = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for i, j, w in edges:
        W[i, j] = w
        W[j, i] = w
    return W


def load_gset_sparse(instance, device="cpu"):
    if torch is None:
        raise RuntimeError("PyTorch is required to build a sparse G-set tensor.")

    num_nodes, edges = read_gset_edges(instance)
    indices = []
    values = []
    for i, j, w in edges:
        indices.append((i, j))
        indices.append((j, i))
        values.append(w)
        values.append(w)

    if indices:
        index_tensor = torch.tensor(indices, dtype=torch.int64, device=device).t()
        value_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    else:
        index_tensor = torch.empty((2, 0), dtype=torch.int64, device=device)
        value_tensor = torch.empty((0,), dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        index_tensor,
        value_tensor,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
        device=device,
    ).coalesce()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()

    for instance in args.instances:
        W = (
            load_gset_sparse(instance, device=args.device)
            if args.sparse
            else load_gset_dense(instance, device=args.device)
        )
        print(f"{instance}: shape={tuple(W.shape)} device={W.device}")


if __name__ == "__main__":
    main()
