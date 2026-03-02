import torch

def build_edge_index(coords, k=10, device="cpu"):
    """
    coords: [N, 2] lat/lon
    k: number of nearest neighbors
    """
    num_nodes = coords.shape[0]

    # Pairwise distances
    dist = torch.cdist(coords, coords)

    # K nearest neighbors (skip itself)
    knn = dist.topk(k + 1, largest=False).indices[:, 1:]  # [N, k]

    # Directed edges i -> j
    src = torch.arange(num_nodes).unsqueeze(1).repeat(1, k).reshape(-1)
    dst = knn.reshape(-1)

    edge_index = torch.stack([src, dst], dim=0)

    # Add reverse edges j -> i
    edge_index_rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

    # Add self-loops
    self_loops = torch.arange(num_nodes)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    return edge_index.to(device)