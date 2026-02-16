import torch

def build_edge_index(num_nodes):
    row, col = [], []
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src != dst:
                row.append(dst)   # target
                col.append(src)   # source
    return torch.tensor([row, col], dtype=torch.long)