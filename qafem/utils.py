import torch

def generate_erdos_renyi_graph(n, p, device="cpu", seed=0):
    torch.manual_seed(seed)
    mask = torch.triu((torch.rand(n, n, device=device) < p).float(), diagonal=1)
    W = mask + mask.T
    return W


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def expected_cut(W, p):
    # p @ W gives each node's weighted neighbour sum; multiply by (1-p) for cut expectation
    return torch.einsum('bi,ij,bj->b', p, W, 1 - p)


def discrete_cut(W, s):
    # cut = 1/4 * (sum_ij W_ij - sum_ij W_ij s_i s_j)
    quad = torch.einsum('bi,ij,bj->b', s, W, s)
    return 0.25 * (W.sum() - quad)