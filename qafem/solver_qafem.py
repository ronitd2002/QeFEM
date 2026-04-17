import torch
from math import log, sqrt  # sqrt used in gamma sqrt schedule
from .utils import expected_cut


def entropy_potts(p):
    return - (p * p.log()).sum(2).sum(1)


def cut(W, p):
    # W: (n,n), p: (batch,n,q)
    return ((W @ p) * (1 - p)).sum((1, 2))


def entropy_grad_binary(p):
    return - (p * (1 - p) * (p.log() - (1 - p).log()))


def make_beta_schedule(beta_min, beta_max, steps, kind, device):
    """
    kind:
      inverse    — beta = 1/T, T linearly spaced Tmax->Tmin (DEFAULT)
                   paper's primary 'inverse-proportional scheduling'
      linear     — beta linearly spaced beta_min->beta_max
      geometric  — log-spaced beta (paper 'exponential scheduling')
    """
    if kind == "inverse":
        T = torch.linspace(1.0 / beta_min, 1.0 / beta_max, steps, device=device)
        return 1.0 / T
    if kind == "linear":
        return torch.linspace(beta_min, beta_max, steps, device=device)
    if kind == "geometric":
        return torch.logspace(log(beta_min, 10), log(beta_max, 10), steps, device=device)
    raise ValueError(f"Unknown beta schedule: {kind}. Use inverse | linear | geometric")


def make_gamma_schedule(gamma_max, gamma_min, steps, kind, device):
    """
    Gamma(t) schedules for the quantum stage transverse field.

    kind:
      linear  — linearly decays gamma_max -> gamma_min (DEFAULT)
                Simple and sufficient for a variational optimizer.
                The functional form between steps has no derivable
                physical consequence here since we are doing gradient
                descent, not solving the Schrodinger equation.

      sqrt    — c/sqrt(t) decay, inspired by Kadowaki & Nishimori 1998.
                Decays faster early, slower late. Try this if linear
                gives a quantum stage that collapses too quickly.
                Normalised so Gamma(steps) == gamma_min automatically.
    """
    if kind == "linear":
        return torch.linspace(gamma_max, gamma_min, steps, device=device)

    if kind == "sqrt":
        t = torch.arange(1, steps + 1, dtype=torch.float32, device=device)
        c = gamma_min * sqrt(steps)
        return (c / torch.sqrt(t)).clamp(gamma_min, gamma_max)

    raise ValueError(f"Unknown gamma schedule: {kind}. Use linear | sqrt")


class QAFEMSolver:
    def __init__(
        self,
        problem,
        num_trials=128,
        beta_min=1e-2,
        beta_max=40.0,
        beta_steps=300,
        beta_schedule="inverse",      # inverse | geometric
        gamma_max=1.5,
        gamma_min=0.0,
        gamma_steps=200,
        gamma_schedule="linear",      # linear (default) | sqrt
        beta_q_min=None,              # quantum stage beta; defaults to beta_max (fixed)
        beta_q_max=None,
        beta_q_schedule="fixed",      # fixed | geometric | inverse
        lr=0.04,
        lr_a=None,
        lr_theta=None,
        device="cpu",
        seed=0
    ):
        self.problem = problem
        self.num_trials = num_trials
        self.device = device
        self.seed = seed
        self.lr = lr
        self.lr_a = lr_a if lr_a is not None else lr
        self.lr_theta = lr_theta if lr_theta is not None else lr

        self.betas = make_beta_schedule(beta_min, beta_max, beta_steps, beta_schedule, device)

        # Gamma schedule — linear default, sqrt available for comparison
        self.gammas = make_gamma_schedule(
            gamma_max, gamma_min, gamma_steps, gamma_schedule, device
        )

        bq_min = beta_q_min if beta_q_min is not None else beta_max
        bq_max = beta_q_max if beta_q_max is not None else beta_max
        if beta_q_schedule == "fixed" or bq_min == bq_max:
            self.beta_qs = torch.full((gamma_steps,), bq_max, device=device)
        else:
            self.beta_qs = make_beta_schedule(bq_min, bq_max, gamma_steps, beta_q_schedule, device)

    def classical_stage(self):
        torch.manual_seed(self.seed)

        N = self.problem.W.shape[0]

        h = torch.rand(
            (self.num_trials, N, 2),
            device=self.device,
            dtype=torch.float32,
        ).requires_grad_()

        opt = torch.optim.Adam([h], lr=self.lr)

        for beta in self.betas:
            p_full = torch.softmax(h, dim=2)
            p = p_full[:, :, 1]  # prob of state 1
            F = -cut(self.problem.W, p_full) - entropy_potts(p_full) / beta

            opt.zero_grad()
            F.backward(torch.ones_like(F))
            opt.step()

        p_full = torch.softmax(h, dim=2)
        return p_full[:, :, 1]

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

        opt = torch.optim.Adam([
            {"params": [a_raw], "lr": self.lr_a},
            {"params": [theta], "lr": self.lr_theta},
        ])

        # Improvement 3: track all three free energy components
        self.quantum_history = {"gamma": [], "cut_mean": [], "cut_max": [],
                                 "U0_mean": [], "rx_mean": [], "S_vN_mean": []}

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
                cut_exp = expected_cut(self.problem.W, p_q)
                self.quantum_history["gamma"].append(float(gamma.item()))
                self.quantum_history["cut_mean"].append(float(cut_exp.mean().item()))
                self.quantum_history["cut_max"].append(float(cut_exp.max().item()))
                self.quantum_history["U0_mean"].append(float(U.mean().item()))
                self.quantum_history["rx_mean"].append(float(rx.abs().mean().item()))
                self.quantum_history["S_vN_mean"].append(float(S.mean().item()))

        a = torch.sigmoid(a_raw)
        rz = a * torch.cos(theta)
        p_final = (1 + rz) / 2
        return p_final

    def solve(self):
        p_classical = self.classical_stage()
        p_quantum = self.quantum_stage(p_classical)

        spins_c, cut_c = self.problem.inference(p_classical)
        spins_q, cut_q = self.problem.inference(p_quantum)

        best_classical_idx = int(cut_c.argmax().item())
        best_quantum_idx = int(cut_q.argmax().item())

        return {
            "p_classical": p_classical.detach(),
            "p_quantum": p_quantum.detach(),
            "best_classical_p": p_classical[best_classical_idx].detach(),
            "best_quantum_p": p_quantum[best_quantum_idx].detach(),
            "spins_classical": spins_c.detach(),
            "spins_quantum": spins_q.detach(),
            "best_spins_classical": spins_c[best_classical_idx].detach(),
            "best_spins_quantum": spins_q[best_quantum_idx].detach(),
            "cut_classical_all": cut_c.detach(),
            "cut_quantum_all": cut_q.detach(),
            "best_classical_cut": float(cut_c[best_classical_idx].item()),
            "best_quantum_cut": float(cut_q[best_quantum_idx].item()),
            "gain": float(cut_q[best_quantum_idx].item() - cut_c[best_classical_idx].item()),
            "quantum_history": self.quantum_history,
        }