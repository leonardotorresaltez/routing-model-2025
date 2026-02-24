import torch

def build_edge_index(num_nodes, device='cpu'):
    indices = torch.arange(num_nodes, device=device)
    src, dst = torch.meshgrid(indices, indices, indexing='ij')
    src = src.reshape(-1)
    dst = dst.reshape(-1)
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    
    return edge_index