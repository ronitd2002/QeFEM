import torch
from math import log


def entropy_binary(p):
    return - (p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum(1)


def entropy_grad_binary(p):
    return - (p * (1 - p) * (p.log() - (1 - p).log()))


def make_beta_schedule(beta_min, beta_max, steps, kind, device):
    """
    kind:
      inverse    — beta = 1/T, T linearly spaced Tmax->Tmin (DEFAULT)
                   paper's primary 'inverse-proportional scheduling'
      geometric  — log-spaced beta (paper 'exponential scheduling')
    """
    if kind == "inverse":
        T = torch.linspace(1.0 / beta_min, 1.0 / beta_max, steps, device=device)
        return 1.0 / T
    if kind == "geometric":
        return torch.logspace(log(beta_min, 10), log(beta_max, 10), steps, device=device)
    raise ValueError(f"Unknown beta schedule: {kind}. Use inverse | geometric")


class QAFEMSolver:
    def __init__(
        self,
        problem,
        num_trials=128,
        beta_min=1e-2,
        beta_max=40.0,
        beta_steps=300,
        beta_schedule="inverse",     # inverse | geometric
        gamma_max=1.5,
        gamma_min=0.0,
        gamma_steps=200,
        beta_q_min=None,             # quantum stage beta; defaults to beta_max (fixed)
        beta_q_max=None,
        beta_q_schedule="fixed",     # fixed | geometric | inverse
        lr=0.04,
        device="cpu",
        seed=0
    ):
        self.problem = problem
        self.num_trials = num_trials
        self.device = device
        self.seed = seed
        self.lr = lr

        # Classical beta schedule (improvement 2a)
        self.betas = make_beta_schedule(beta_min, beta_max, beta_steps, beta_schedule, device)

        self.gammas = torch.linspace(gamma_max, gamma_min, gamma_steps, device=device)

        # Independent quantum beta schedule (improvement 2b)
        bq_min = beta_q_min if beta_q_min is not None else beta_max
        bq_max = beta_q_max if beta_q_max is not None else beta_max
        if beta_q_schedule == "fixed" or bq_min == bq_max:
            self.beta_qs = torch.full((gamma_steps,), bq_max, device=device)
        else:
            self.beta_qs = make_beta_schedule(bq_min, bq_max, gamma_steps, beta_q_schedule, device)

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

        # Improvement 3: track all three free energy components
        self.quantum_history = {"gamma": [], "cut_mean": [], "cut_max": [],
                                 "U_mean": [], "rx_mean": [], "S_mean": []}

        for gamma, beta_q in zip(self.gammas, self.beta_qs):
            a = torch.sigmoid(a_raw)
            rx = a * torch.sin(theta)
            rz = a * torch.cos(theta)

            # Improvement 1: energy from problem layer, not hardcoded here
            U = self.problem.quantum_energy(rz)

            T = -gamma * rx.sum(1)

            lp = ((1 + a) / 2).clamp(1e-12, 1)
            lm = ((1 - a) / 2).clamp(1e-12, 1)
            S = -(lp * torch.log(lp) + lm * torch.log(lm)).sum(1)

            F = U + T - S / beta_q   # improvement 2b: beta_q per step

            opt.zero_grad()
            F.mean().backward()
            opt.step()

            with torch.no_grad():
                p_q = ((1 + rz) / 2).clamp(0.0, 1.0)
                from .utils import expected_cut
                cut_exp = expected_cut(self.problem.W, p_q)
                self.quantum_history["gamma"].append(float(gamma.item()))
                self.quantum_history["cut_mean"].append(float(cut_exp.mean().item()))
                self.quantum_history["cut_max"].append(float(cut_exp.max().item()))
                self.quantum_history["U_mean"].append(float(U.mean().item()))
                self.quantum_history["rx_mean"].append(float(rx.abs().mean().item()))
                self.quantum_history["S_mean"].append(float(S.mean().item()))

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
            "gain": (cut_q.max() - cut_c.max()).item(),
            # Improvement 3: expose diagnostics so experiment layer can inspect them
            "quantum_history": self.quantum_history,
        }