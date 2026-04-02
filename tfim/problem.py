import torch
from .utils import expected_cut, discrete_cut

class MaxCutProblem:
    def __init__(self, W):
        self.W = W

    def expectation(self, p):
        return -expected_cut(self.W, p)

    def inference(self, p):
        spins = torch.where(p >= 0.5, torch.ones_like(p), -torch.ones_like(p))
        cut = discrete_cut(self.W, spins)
        return spins, cut