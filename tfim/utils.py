import torch

def generate_erdos_renyi_graph(n, p, device="cpu", seed=0):
    torch.manual_seed(seed)
    mask = torch.triu((torch.rand(n, n, device=device) < p).float(), diagonal=1)
    W = mask + mask.T
    return W


def expected_cut(W, p):
    return ((p @ W) * (1 - p)).sum(dim=1)


def discrete_cut(W, s):
    quad = ((s @ W) * s).sum(dim=1)
    return 0.25 * (W.sum() - quad)