import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
def load_graph_data(file_path):
    # Load the graph data from a file (e.g., edge list)
    G = nx.read_edgelist(file_path, nodetype=int)
    
    # Create node features (for simplicity, using degree as a feature)
    node_features = np.array([G.degree(node) for node in G.nodes()])
    
    # Create edge index
    edge_index = np.array(list(G.edges())).T  # Shape: [2, num_edges]
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)  # Shape: [num_nodes, num_features]
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # Shape: [2, num_edges]
    
    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data