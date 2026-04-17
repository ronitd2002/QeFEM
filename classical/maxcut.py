import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# GRAPH UTILITIES
def random_binary_undirected_graph(n_nodes=20, edge_prob=0.2, seed=1): # generate random undirected graph
    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_prob, seed=seed, directed=False)
    return G
def draw_graph(G, seed=42):
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(G, pos, with_labels=True, node_color="lightblue")
    plt.show()
def graph_to_binary_adjacency(G):
    n = G.number_of_nodes()
    W = np.zeros((n, n), dtype=np.float32)
    for u, v in G.edges():
        W[u, v] = 1.0
        W[v, u] = 1.0
    return W

def draw_cut(G, x, seed=42):
    pos = nx.spring_layout(G, seed=seed)

    # node colors: 0 -> one color, 1 -> another
    node_colors = ["tab:blue" if x[i] == 0 else "tab:orange" for i in G.nodes()]

    # edge colors: cut edges in red, others in light gray
    edge_colors = []
    for u, v in G.edges():
        edge_colors.append("tab:red" if x[u] != x[v] else "lightgray")

    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, width=2)
    plt.show()

def cut(W, p):
    # W: (n,n), p: (batch,n,q)
    return ((W @ p) * (1 - p)).sum((1, 2))

def S(p, eps=1e-12):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(2).sum(1)

def argmax_cut(W, p):
    config = p.argmax(dim=2)  # (batch,n)
    s = F.one_hot(config, num_classes=p.shape[2]).float()
    return config, cut(W, s) / 2

def solve_maxcut(W, batch=64, lr=0.01, steps=200):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    W = torch.tensor(W, dtype=torch.float32).to(device)
    n = W.shape[0]
    q = 2
    h = torch.nn.Parameter(torch.rand(batch, n, q, device=device))
    optimizer = torch.optim.Adam([h], lr=lr)
    beta_range = torch.linspace(0.5, 20.0, steps=steps).to(device)

    for beta in beta_range:
        p = torch.softmax(h, dim=2)
        Fval = -cut(W, p) - S(p) / beta

        optimizer.zero_grad()
        Fval.backward(torch.ones_like(Fval))
        optimizer.step()

    p_final = torch.softmax(h, dim=2)
    config, cutvals = argmax_cut(W, p_final)
    return config, cutvals, p_final, h.detach()

G = random_binary_undirected_graph(n_nodes=20, edge_prob=0.2, seed=1)
W = graph_to_binary_adjacency(G)

# --- device ---
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- graph ---
W = torch.tensor(W, dtype=torch.float32).to(device)

# --- hyperparameters ---
batch = 64
q = 2
n = W.shape[0]
beta_range = torch.linspace(0.5, 20.0, steps=200).to(device)

# --- learnable fields ---
h = torch.nn.Parameter(torch.rand(batch, n, q, device=device))

# solve
config, cutvals, p, h_final = solve_maxcut(W, batch=64, steps=500)

# pick best replica r*
best = cutvals.argmax().item()
p_best = p[best]
h_best = h_final[best]

W_cpu = W.detach().cpu() if W.device.type != "cpu" else W
p_best_cpu = p_best.detach().cpu()
h_best_cpu = h_best.detach().cpu()


# local scores: (n,)
score0 = (W_cpu @ p_best_cpu[:, 1])  # sum_j W_ij * p_j(1)
score1 = (W_cpu @ p_best_cpu[:, 0])  # sum_j W_ij * p_j(0)
local_argmax = (score1 > score0).int()  # 1 if score1 bigger else 0

best_cut = cutvals[best].item()
best_config = local_argmax

print("Best cut value:", best_cut)
print("h_best:\n", h_best_cpu)
print("p_best:\n", p_best_cpu)
#print("score0:", score0)
#print("score1:", score1)
print("local_argmax partition:", local_argmax.numpy())

x = best_config.cpu().numpy()
import pandas as pd
def maxcut_marginals_dataframe(h_best, p_best):
    import pandas as pd
    h_best_cpu = h_best.detach().cpu()
    p_best_cpu = p_best.detach().cpu()

    df = pd.DataFrame({
        "logit_0": h_best_cpu[:, 0],
        "logit_1": h_best_cpu[:, 1],
        "marginal_p(0)": p_best_cpu[:, 0],
        "marginal_p(1)": p_best_cpu[:, 1],
    })
    return df

# ---- usage ----
df_maxcut = maxcut_marginals_dataframe(h_best, p_best)
draw_cut(G,x)
df_maxcut