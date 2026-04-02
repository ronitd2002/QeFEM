# double_stage_maxcut_linear.py
# Classical FEM stage + Quantum mean-field stage for MaxCut
# with linear transverse-field annealing.

import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch


# ---------------------------
# Device selection (MPS-aware)
# ---------------------------

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------
# Utilities
# ---------------------------

def read_graph(file: str, index_start: int = 1, device: str = "cpu") -> torch.Tensor:
    """
    Read an undirected weighted graph into a symmetric adjacency matrix W.
    File format:
        n m
        i j [w]
    If w is absent, weight 1.0 is assumed.
    """
    with open(file, "r") as f:
        n, m = map(int, f.readline().strip().split())
        W = torch.zeros((n, n), dtype=torch.float32, device=device)
        for _ in range(m):
            parts = f.readline().strip().split()
            if len(parts) < 2:
                continue
            i, j = int(parts[0]) - index_start, int(parts[1]) - index_start
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            W[i, j] = w
            W[j, i] = w
    return W


def generate_erdos_renyi_graph(
    n: int,
    edge_prob: float,
    weight_mode: str = "ones",
    weight_low: float = 0.5,
    weight_high: float = 1.5,
    seed: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a symmetric weighted Erdős-Rényi graph.
    weight_mode: "ones" | "uniform" | "signed"
    """
    torch.manual_seed(seed)
    upper_mask = torch.triu(
        (torch.rand((n, n), device=device) < edge_prob).float(), diagonal=1
    )
    if weight_mode == "ones":
        weights = torch.ones((n, n), device=device)
    elif weight_mode == "uniform":
        weights = weight_low + (weight_high - weight_low) * torch.rand((n, n), device=device)
    elif weight_mode == "signed":
        weights = torch.where(
            torch.rand((n, n), device=device) < 0.5,
            -torch.ones((n, n), device=device),
            torch.ones((n, n), device=device),
        )
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    W = (weights * upper_mask)
    W = W + W.T
    W.fill_diagonal_(0.0)
    return W


def expected_cut_from_probs(W: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Expected cut for Bernoulli marginals. W:[N,N], p:[R,N] -> [R]"""
    return ((p @ W) * (1.0 - p)).sum(dim=-1)


def shannon_entropy_binary(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Binary entropy summed over nodes. p:[R,N] -> [R]"""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).sum(dim=-1)


def discrete_cut_from_spins(W: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Cut value from Ising spins s in {-1,+1}. W:[N,N], s:[R,N] -> [R]"""
    quad = ((s @ W) * s).sum(dim=-1)
    return 0.25 * (W.sum() - quad)


def make_beta_schedule(
    beta_min: float, beta_max: float, steps: int,
    kind: str = "geometric", device: str = "cpu"
) -> torch.Tensor:
    if kind == "geometric":
        return torch.logspace(math.log10(beta_min), math.log10(beta_max), steps, device=device)
    if kind == "linear":
        return torch.linspace(beta_min, beta_max, steps, device=device)
    raise ValueError(f"Unknown beta schedule: {kind}")


# ---------------------------
# Stage 1: Classical FEM
# ---------------------------

@dataclass
class ClassicalStageResult:
    h: torch.Tensor
    p: torch.Tensor
    cut_discrete: torch.Tensor
    spins: torch.Tensor
    best_replica: int
    history: Dict[str, list]


def run_classical_fem_stage(
    W: torch.Tensor,
    replicas: int = 128,
    beta_min: float = 1e-2,
    beta_max: float = 40.0,
    beta_steps: int = 300,
    beta_schedule: str = "geometric",
    lr: float = 0.04,
    seed: int = 0,
    device: Optional[str] = None,
) -> ClassicalStageResult:
    """
    Classical FEM stage for MaxCut (q=2, one Bernoulli field per node).
    Minimizes: F_cl = -E_cut[p] - S[p]/beta
    """
    if device is None:
        device = get_device()
    torch.manual_seed(seed)
    W = W.to(device)
    N = W.shape[0]

    h = 1e-3 * torch.randn((replicas, N), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([h], lr=lr)
    beta_range = make_beta_schedule(beta_min, beta_max, beta_steps, beta_schedule, device)

    history = {"beta": [], "cut_mean": [], "cut_max": []}

    for beta in beta_range:
        p = torch.sigmoid(h)
        cut_exp = expected_cut_from_probs(W, p)
        F = -cut_exp - shannon_entropy_binary(p) / beta

        optimizer.zero_grad()
        F.mean().backward()
        optimizer.step()

        with torch.no_grad():
            history["beta"].append(float(beta.item()))
            history["cut_mean"].append(float(cut_exp.mean().item()))
            history["cut_max"].append(float(cut_exp.max().item()))

    with torch.no_grad():
        p = torch.sigmoid(h)
        spins = torch.where(p >= 0.5, torch.ones_like(p), -torch.ones_like(p))
        cut_disc = discrete_cut_from_spins(W, spins)
        best = int(torch.argmax(cut_disc).item())

    return ClassicalStageResult(
        h=h.detach(), p=p.detach(),
        cut_discrete=cut_disc, spins=spins,
        best_replica=best, history=history,
    )


# ---------------------------
# Stage 2: Quantum mean-field
# ---------------------------

@dataclass
class QuantumStageResult:
    rx: torch.Tensor
    rz: torch.Tensor
    rho_prob: torch.Tensor
    cut_discrete: torch.Tensor
    spins: torch.Tensor
    best_replica: int
    history: Dict[str, list]


def von_neumann_entropy(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Von Neumann entropy for qubits with Bloch radius a. a:[R,N] -> [R]"""
    lp = ((1.0 + a) * 0.5).clamp(eps, 1.0)
    lm = ((1.0 - a) * 0.5).clamp(eps, 1.0)
    return -(lp * torch.log(lp) + lm * torch.log(lm)).sum(dim=-1)


def run_quantum_linear_stage(
    W: torch.Tensor,
    p_init: torch.Tensor,
    gamma_max: float = 1.5,
    gamma_min: float = 0.0,
    gamma_steps: int = 200,
    beta_q: float = 60.0,
    lr_a: float = 0.02,
    lr_theta: float = 0.02,
    device: Optional[str] = None,
) -> QuantumStageResult:
    """
    Quantum mean-field stage. Linear Gamma schedule.
    Qubit ansatz: rho_i = 1/2 (I + rx sigma_x + rz sigma_z)
    Minimizes: F_q = U0(rz) - Gamma * sum_i rx_i - S_vN(a)/beta_q
    """
    if device is None:
        device = get_device()
    W, p_init = W.to(device), p_init.to(device)
    eps = 1e-4

    # Init from classical solution
    mz = (2.0 * p_init - 1.0).clamp(-1 + eps, 1 - eps)
    a0 = mz.abs().clamp(eps, 1 - eps)
    theta0 = torch.where(mz >= 0, torch.zeros_like(mz), math.pi * torch.ones_like(mz))
    a_raw = torch.log(a0 / (1.0 - a0)).detach().clone().requires_grad_(True)
    theta = theta0.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([
        {"params": [a_raw], "lr": lr_a},
        {"params": [theta], "lr": lr_theta},
    ])
    gamma_range = torch.linspace(gamma_max, gamma_min, gamma_steps, device=device)

    history = {"gamma": [], "cut_mean": [], "cut_max": []}

    for gamma in gamma_range:
        a = torch.sigmoid(a_raw)
        rx = a * torch.sin(theta)
        rz = a * torch.cos(theta)

        U0 = 0.25 * ((rz @ W) * rz).sum(dim=-1)
        Fq = U0 - gamma * rx.sum(dim=-1) - von_neumann_entropy(a) / beta_q

        optimizer.zero_grad()
        Fq.mean().backward()
        optimizer.step()

        with torch.no_grad():
            p_q = ((1.0 + rz) * 0.5).clamp(0.0, 1.0)
            cut_exp = expected_cut_from_probs(W, p_q)
            history["gamma"].append(float(gamma.item()))
            history["cut_mean"].append(float(cut_exp.mean().item()))
            history["cut_max"].append(float(cut_exp.max().item()))

    with torch.no_grad():
        a = torch.sigmoid(a_raw)
        rx = a * torch.sin(theta)
        rz = a * torch.cos(theta)
        spins = torch.where(rz >= 0, torch.ones_like(rz), -torch.ones_like(rz))
        cut_disc = discrete_cut_from_spins(W, spins)
        best = int(torch.argmax(cut_disc).item())
        p_q = ((1.0 + rz) * 0.5).clamp(0.0, 1.0)

    return QuantumStageResult(
        rx=rx.detach(),
        rz=rz.detach(),
        rho_prob=p_q.detach(),
        cut_discrete=cut_disc.detach(),
        spins=spins.detach(),
        best_replica=best,
        history=history,
    )


# ---------------------------
# Full pipeline
# ---------------------------

@dataclass
class DoubleStageResult:
    classical: ClassicalStageResult
    quantum: QuantumStageResult
    best_cut_classical: float
    best_cut_quantum: float
    best_spins_classical: torch.Tensor
    best_spins_quantum: torch.Tensor


def run_double_stage(
    W: torch.Tensor,
    replicas: int = 128,
    beta_min: float = 1e-2,
    beta_max: float = 40.0,
    beta_steps: int = 300,
    beta_schedule: str = "geometric",
    gamma_max: float = 1.5,
    gamma_min: float = 0.0,
    gamma_steps: int = 200,
    beta_q: Optional[float] = None,
    lr_classical: float = 0.04,
    lr_a: float = 0.02,
    lr_theta: float = 0.02,
    seed: int = 0,
    device: Optional[str] = None,
) -> DoubleStageResult:
    if device is None:
        device = get_device()
    W = W.to(device)
    if beta_q is None:
        beta_q = beta_max

    classical = run_classical_fem_stage(
        W=W, replicas=replicas,
        beta_min=beta_min, beta_max=beta_max,
        beta_steps=beta_steps, beta_schedule=beta_schedule,
        lr=lr_classical, seed=seed, device=device,
    )
    quantum = run_quantum_linear_stage(
        W=W, p_init=classical.p,
        gamma_max=gamma_max, gamma_min=gamma_min,
        gamma_steps=gamma_steps, beta_q=beta_q,
        lr_a=lr_a, lr_theta=lr_theta, device=device,
    )

    bc, bq = classical.best_replica, quantum.best_replica
    return DoubleStageResult(
        classical=classical, quantum=quantum,
        best_cut_classical=float(classical.cut_discrete[bc].item()),
        best_cut_quantum=float(quantum.cut_discrete[bq].item()),
        best_spins_classical=classical.spins[bc].cpu(),
        best_spins_quantum=quantum.spins[bq].cpu(),
    )


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    N = 100
    W = generate_erdos_renyi_graph(n=N, edge_prob=0.15, weight_mode="ones", seed=7, device=device)

    result = run_double_stage(
        W=W,
        replicas=128,
        beta_min=1e-2,
        beta_max=40.0,
        beta_steps=300,
        beta_schedule="geometric",
        gamma_max=1.5,
        gamma_min=0.0,
        gamma_steps=200,
        beta_q=60.0,
        lr_classical=0.04,
        lr_a=0.02,
        lr_theta=0.02,
        seed=123,
        device=device,
    )

    print("\n=== Results ===")
    print(f"Best classical FEM cut : {result.best_cut_classical:.4f}")
    print(f"Best quantum-stage cut : {result.best_cut_quantum:.4f}")
    print(f"Gain from stage 2      : {result.best_cut_quantum - result.best_cut_classical:.4f}")

    print("\nClassical stage (final):")
    print(f"  mean cut : {result.classical.history['cut_mean'][-1]:.4f}")
    print(f"  max cut  : {result.classical.history['cut_max'][-1]:.4f}")

    print("\nQuantum stage (final):")
    print(f"  mean cut : {result.quantum.history['cut_mean'][-1]:.4f}")
    print(f"  max cut  : {result.quantum.history['cut_max'][-1]:.4f}")

    if N <= 30:
        print("\nBest classical spins:", result.best_spins_classical.numpy())
        print("Best quantum spins  :", result.best_spins_quantum.numpy())
    else:
        print(f"\nSpin configs shape: {tuple(result.best_spins_classical.shape)} (N={N}, not printing)")