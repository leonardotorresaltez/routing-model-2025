import torch

def build_edge_index(num_nodes, device='cpu'):
    # Generate all indices from 0 to num_nodes-1
    indices = torch.arange(num_nodes, device=device)
    
    # Create a grid of all possible pairs (src, dst)
    grid = torch.meshgrid(indices, indices, indexing='ij')
    src = grid[0].reshape(-1)
    dst = grid[1].reshape(-1)
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    
    return edge_index