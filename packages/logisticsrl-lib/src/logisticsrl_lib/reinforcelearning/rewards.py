import gymnasium as gym
import numpy as np

class NormalizedRewards:
    def __init__(self, cfg,time_matrix):
       
        self.cfg = cfg
        self.time_matrix = time_matrix
        mask = time_matrix > 0
        self.global_mean = time_matrix[mask].mean().item()
        self.global_std = time_matrix[mask].std().item() 
        self.global_max = time_matrix[mask].max().item()
        self.global_min = time_matrix[mask].min().item()

        print(f"Global time matrix stats: mean={   self.global_mean:.2f}, std={self.global_std:.2f}, min={self.global_min:.2f}, max={self. global_max:.2f}")

 
    
    def getRewardVisitBonus(self):
        base_reward = (0 - (-self.global_mean)) / self.global_std # aprox 0.87
        if (self.cfg.data_dir == "data_version_2" ):
            return 0.87
        if (self.cfg.data_dir == "data_version_1" ):    
            return 0.72
        else:
            return self.global_mean / self.global_std
        
    def getRewardNonOP(self):
        if (self.cfg.data_dir == "data_version_2" ):
            return -1.5   # trucks can retire when remaining visits are inefficient
        if (self.cfg.data_dir == "data_version_1" ):
            # With 12h limit, trucks are often FORCED to NO-OP (masked nodes).
            # Low penalty avoids punishing unavoidable retirements and allows
            # the policy to skip very distant nodes (trip > 1.22 std above mean).
            return -0.5
        else:
            return -self.global_mean

    def getRewardTotalFleetTime(self, total_fleet_time):
        if (self.cfg.data_dir == "data_version_2"):
            # Linear: 3x stronger than original /86.5. At 1000h: -17.2 (~26% of mean return).
            efficiency_reward = -(total_fleet_time - 500) / 29.0
            return float(np.clip(efficiency_reward, -20.0, 5.0))
        else:
            # data_version_1: 5 trucks × 12h max = 60h capacity. User target: <35h.
            # Divisor 2.5 (2x stronger than before):
            # At 52h: -6.8  |  At 35h: 0  |  At 25h: +4
            # At 52h terminal ≈ -6.8 - coverage = ~-9.6 vs mean returns ~10 → 96% swing.
            efficiency_reward = -(total_fleet_time - 35) / 2.5
            return float(np.clip(efficiency_reward, -15.0, 5.0))
            

    def getRewardCoverage(self, n_unvisited):
        """
        Terminal penalty for each customer left unvisited.
        Normalized with the same base as visit_bonus (global_mean / global_std ≈ 0.87),
        so each unvisited customer costs exactly -1 normalized unit (same scale as other rewards).
        """
        penalty_per_customer = self.global_mean / self.global_std  # ≈ 0.87, same scale as visit bonus
        return -n_unvisited * penalty_per_customer

    def getRewardDistance(self, prev_node, selected_node):
        """
        'reward' es el valor crudo que sale de tu env (ej: -time_matrix[u,v]).
        """
        reward = self.time_matrix[prev_node, selected_node].item()  

        norm_reward =  (reward - self.global_mean) / self.global_std

        return - norm_reward