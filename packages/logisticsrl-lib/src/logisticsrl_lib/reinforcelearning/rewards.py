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
        base_reward = (13.77 - (-self.global_mean)) / self.global_std # aprox 0.87
        if (self.cfg.data_dir == "data_version_2" ):
            return - 3.0
        if (self.cfg.data_dir == "data_version_1" ):    
            return - 2.5
        else:
            return -self.global_mean
        

            

    def getRewardDistance(self, prev_node, selected_node):
        """
        'reward' es el valor crudo que sale de tu env (ej: -time_matrix[u,v]).
        """
        reward = self.time_matrix[prev_node, selected_node].item()  

        norm_reward =  (reward - self.global_mean) / self.global_std

        return - norm_reward