import torch
import torch.nn.functional as F
from .utils import expected_cut, discrete_cut

class MaxCutProblem:
    def __init__(self, W):
        self.W = W

    def expectation(self, p):
        return -expected_cut(self.W, p)

    def inference(self, p):
        config = (p > 0.5).long()
        s = F.one_hot(config, num_classes=2).float()
        cut = ((self.W @ s) * (1 - s)).sum((1, 2)) / 2
        spins = config.float()  # 0 or 1
        return spins, cut

    def quantum_energy(self, rz):
        """
        Mean-field Ising energy for the quantum stage.
        Derived from paper Eq. S6 with sigma_z magnetizations:
            U = 0.25 * sum_{ij} W_ij * rz_i * rz_j
        Minimising this anti-aligns rz across edges, maximising cut.
        rz: [R, N] -> [R]
        """
        return 0.25 * ((rz @ self.W) * rz).sum(dim=1)