import torch
from math import log


def entropy_binary(p):
    return - (p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum(1)


def entropy_grad_binary(p):
    return - (p * (1 - p) * (p.log() - (1 - p).log()))


class QAFEMSolver:
    def __init__(
        self,
        problem,
        num_trials=128,
        beta_min=1e-2,
        beta_max=40.0,
        beta_steps=300,
        gamma_max=1.5,
        gamma_min=0.0,
        gamma_steps=200,
        lr=0.04,
        device="cpu",
        seed=0
    ):
        self.problem = problem
        self.num_trials = num_trials
        self.device = device
        self.seed = seed
        self.lr = lr

        self.betas = torch.logspace(
            log(beta_min, 10),
            log(beta_max, 10),
            beta_steps,
            device=device
        )

        self.gammas = torch.linspace(
            gamma_max,
            gamma_min,
            gamma_steps,
            device=device
        )

    def classical_stage(self):
        torch.manual_seed(self.seed)

        N = self.problem.W.shape[0]

        h = 1e-3 * torch.randn(
            (self.num_trials, N),
            device=self.device,
            requires_grad=True
        )

        opt = torch.optim.Adam([h], lr=self.lr)

        for beta in self.betas:
            p = torch.sigmoid(h)

            F = self.problem.expectation(p) - entropy_binary(p) / beta

            opt.zero_grad()
            F.mean().backward()
            opt.step()

        return torch.sigmoid(h)

    def quantum_stage(self, p_init):
        eps = 1e-4

        mz = (2 * p_init - 1).clamp(-1 + eps, 1 - eps)

        a_raw = torch.log(mz.abs() / (1 - mz.abs()))
        a_raw = a_raw.detach().clone().requires_grad_(True)

        theta = torch.where(
            mz >= 0,
            torch.zeros_like(mz),
            torch.ones_like(mz) * torch.pi
        ).detach().clone().requires_grad_(True)

        opt = torch.optim.Adam([a_raw, theta], lr=self.lr)

        W = self.problem.W

        for gamma in self.gammas:
            a = torch.sigmoid(a_raw)
            rx = a * torch.sin(theta)
            rz = a * torch.cos(theta)

            # classical Ising-like energy for MaxCut
            U = 0.25 * ((rz @ W) * rz).sum(1)

            # transverse-field contribution
            T = -gamma * rx.sum(1)

            # von Neumann entropy of product qubit states
            lp = ((1 + a) / 2).clamp(1e-12, 1)
            lm = ((1 - a) / 2).clamp(1e-12, 1)
            S = -(lp * torch.log(lp) + lm * torch.log(lm)).sum(1)

            F = U + T - S / self.betas[-1]

            opt.zero_grad()
            F.mean().backward()
            opt.step()

        a = torch.sigmoid(a_raw)
        rz = a * torch.cos(theta)
        p_final = (1 + rz) / 2
        return p_final

    def solve(self):
        p_classical = self.classical_stage()
        p_quantum = self.quantum_stage(p_classical)

        spins_c, cut_c = self.problem.inference(p_classical)
        spins_q, cut_q = self.problem.inference(p_quantum)

        return {
            "classical_cut": cut_c.max().item(),
            "quantum_cut": cut_q.max().item(),
            "gain": (cut_q.max() - cut_c.max()).item()
        }