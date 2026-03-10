import gymnasium as gym
import numpy as np

class NormalizedRewards:
    def __init__(self, cfg, time_matrix, knn_k=15):

        self.cfg = cfg
        self.time_matrix = time_matrix
        mask = time_matrix > 0
        self.global_mean = time_matrix[mask].mean().item()
        self.global_std = time_matrix[mask].std().item()
        self.global_max = time_matrix[mask].max().item()
        self.global_min = time_matrix[mask].min().item()

        print(f"Global time matrix stats: mean={   self.global_mean:.2f}, std={self.global_std:.2f}, min={self.global_min:.2f}, max={self. global_max:.2f}")

        # Precompute KNN neighbor sets for zone reward (same k as policy)
        tm = time_matrix.numpy() if hasattr(time_matrix, 'numpy') else np.array(time_matrix)
        n = tm.shape[0]
        self._knn_neighbors = []
        for i in range(n):
            dists = tm[i].copy()
            dists[i] = np.inf  # exclude self
            knn = np.argpartition(dists, knn_k)[:knn_k]
            self._knn_neighbors.append(set(knn.tolist()))

 
    
    def getRewardVisitBonus(self):
        base_reward = (0 - (-self.global_mean)) / self.global_std # aprox 0.87
        if (self.cfg.data_dir == "data_version_2" ):
            return 0.87
        if (self.cfg.data_dir == "data_version_1" ):    
            return 0.72
        else:
            return self.global_mean / self.global_std
        
    def getRewardNonOP(self):
        base_reward = (13.77 - (-self.global_mean)) / self.global_std # aprox 0.87
        if (self.cfg.data_dir == "data_version_2" ):
            return - 1.5  # reduced from -3.0: trucks can now retire when remaining visits are inefficient
        if (self.cfg.data_dir == "data_version_1" ):    
            return - 2.5
        else:
            return -self.global_mean
        
    def getRewardTotalFleetTime(self, total_fleet_time):
        # Linear penalty, 3x stronger than the original /86.5 version.
        # At 1000h: -17.2  →  redistributed to all steps ≈ 26% of mean return (~65).
        # At  750h:  -8.6  →  ≈ 13% weight.  At 500h: 0.
        efficiency_reward = -(total_fleet_time - 500) / 29.0
        return float(np.clip(efficiency_reward, -20.0, 5.0))
            

    def getRewardCoverage(self, n_unvisited):
        """
        Terminal penalty for each customer left unvisited.
        Normalized with the same base as visit_bonus (global_mean / global_std ≈ 0.87),
        so each unvisited customer costs exactly -1 normalized unit (same scale as other rewards).
        """
        penalty_per_customer = self.global_mean / self.global_std  # ≈ 0.87, same scale as visit bonus
        return -n_unvisited * penalty_per_customer

    def getRewardZoneBonus(self, prev_node, selected_node):
        """Bonus if the truck stays within the KNN zone of its previous node."""
        if selected_node in self._knn_neighbors[prev_node]:
            return 0.15  # ~17% of visit_bonus (0.87), same normalized scale
        return 0.0

    def getRewardDistance(self, prev_node, selected_node):
        """
        'reward' es el valor crudo que sale de tu env (ej: -time_matrix[u,v]).
        """
        reward = self.time_matrix[prev_node, selected_node].item()  

        norm_reward =  (reward - self.global_mean) / self.global_std

        return - norm_reward