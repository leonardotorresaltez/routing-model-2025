import numpy as np

class NormalizedRewards:
    def __init__(self, cfg, time_matrix=None):
        self.cfg = cfg
        self.time_matrix = time_matrix 
        # UNIVERSAL CONSTANTS
        self.VISIT_BONUS = +2.0            
        self.NO_OP_PENALTY = -0.5       
        self.COVERAGE_PENALTY = -2.0       

        self.DISTANCE_SCALE = 0.01      

        # Fleet time penalty
        self.FLEET_TIME_SCALE = 0.02      

    #Visit Bonus  
    def getRewardVisitBonus(self):
        return self.VISIT_BONUS
    # NO-OP Penalty
    def getRewardNonOP(self):
        return self.NO_OP_PENALTY
    # Distance Penalty
    def getRewardDistance(self, prev_node, selected_node):
        travel_time = self.time_matrix[prev_node, selected_node].item()
        return -self.DISTANCE_SCALE * travel_time
    #Fleet Time Penalty 
    def getRewardTotalFleetTime(self, total_fleet_time):
        return -self.FLEET_TIME_SCALE * total_fleet_time
    #Coverage Penalty 
    def getRewardCoverage(self, n_unvisited):
        return self.COVERAGE_PENALTY * n_unvisited
    def getRewardDistance(self, prev_node, selected_node):
        travel_time = self.time_matrix[prev_node, selected_node].item()
        return -self.DISTANCE_SCALE * travel_time