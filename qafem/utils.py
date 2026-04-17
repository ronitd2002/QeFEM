import torch

def generate_erdos_renyi_graph(n, edge_prob, weight_mode="ones", device="cpu", seed=0):
    import numpy as np
    np.random.seed(seed)
    mask = np.triu((np.random.rand(n, n) < edge_prob).astype(np.float32), k=1)
    W = mask + mask.T
    W = torch.tensor(W, device=device)
    if weight_mode == "ones":
        return W
    if weight_mode == "uniform":
        np.random.seed(seed + 1)  # different seed for weights
        weights = 0.5 + np.random.rand(n, n)
        weights = np.triu(weights, k=1)
        weights = weights + weights.T
        weights = torch.tensor(weights, device=device)
        return W * weights
    raise ValueError(f"Unknown weight_mode: {weight_mode}. Use ones | uniform")


def get_device():
    # return "mps" if torch.backends.mps.is_available() else "cpu"
    return "cpu"  # to match notebook


def expected_cut(W, p):
    # p @ W gives each node's weighted neighbour sum; multiply by (1-p) for cut expectation
    return torch.einsum('bi,ij,bj->b', p, W, 1 - p)


def discrete_cut(W, s):
    # cut = 1/4 * (sum_ij W_ij - sum_ij W_ij s_i s_j)
    quad = torch.einsum('bi,ij,bj->b', s, W, s)
    return 0.25 * (W.sum() - quad)