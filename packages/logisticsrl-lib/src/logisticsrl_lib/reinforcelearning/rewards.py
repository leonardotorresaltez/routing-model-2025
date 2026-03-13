import gymnasium as gym
import numpy as np

class NormalizedRewards:
    def __init__(self, cfg, time_matrix):
        self.cfg = cfg
        self.time_matrix = time_matrix
        mask = time_matrix > 0
        if mask.any():
            self.global_mean = time_matrix[mask].mean().item()
            self.global_std = time_matrix[mask].std().item() 
            self.global_max = time_matrix[mask].max().item()
            self.global_min = time_matrix[mask].min().item()
        else:
            self.global_mean = 0.0
            self.global_std = 1.0
            self.global_max = 0.0
            self.global_min = 0.0

        if self.global_std == 0:
            self.global_std = 1.0

        print(f"Global time matrix stats: mean={self.global_mean:.2f}, std={self.global_std:.2f}, min={self.global_min:.2f}, max={self.global_max:.2f}")

    def getRewardVisitBonus(self):
        """
        Reward for visiting a customer. 
        Scaled to the average move cost (global_mean / global_std).
        """
        return self.global_mean / self.global_std
        
    def getRewardNonOP(self):
        """
        Small penalty for a truck being retired (NO-OP).
        Relative to the visit bonus scale.
        """
        base_unit = self.global_mean / self.global_std
        return -0.2 * base_unit
        
    def getRewardTotalFleetTime(self, total_fleet_time):
        """
        Terminal penalty for total time spent by all trucks.
        Normalized by global_std to stay in the same scale as other rewards.
        """
        return -0.1 * (total_fleet_time / self.global_std)

    def getRewardCoverage(self, n_unvisited):
        """
        Terminal penalty for each customer left unvisited.
        Stronger scale than the visit bonus to ensure customers are prioritized.
        """
        penalty_per_customer = 1.5 * (self.global_mean / self.global_std)
        return -n_unvisited * penalty_per_customer

    def getRewardDistance(self, prev_node, selected_node):
        """
        Cost of moving between two nodes.
        Normalized to ensure the average move has a cost around -mean/std.
        """
        dist = self.time_matrix[prev_node, selected_node].item()  
        return -(dist / self.global_std)