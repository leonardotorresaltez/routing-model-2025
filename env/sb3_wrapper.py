"""
Wrapper for SimpleFleetRoutingEnv to work with Stable Baselines 3.

Handles action masking and observation conversion for PPO/A2C training.
"""

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.routing_env_simple import SimpleFleetRoutingEnv


class FleetRoutingSB3Wrapper(gym.Wrapper):
    """
    Wrapper for SimpleFleetRoutingEnv to enable stable-baselines3 training.
    
    Key features:
    - Converts Dict observation to flat vector for neural network input
    - Stores feasibility mask for action masking during policy sampling
    - Implements action validation before environment step
    """
    
    def __init__(self, env: SimpleFleetRoutingEnv):
        """Initialize wrapper."""
        super().__init__(env)
        
        num_trucks = env.num_trucks
        num_customers = env.num_customers
        
        self.num_trucks = num_trucks
        self.num_customers = num_customers
        
        self._last_feasibility_mask = None
        self.observation_space = self._build_observation_space()
        self.action_space = spaces.MultiDiscrete([num_customers + 1] * num_trucks) #  C choices x T trucks, choices per truck
           
    def _build_observation_space(self) -> spaces.Box:
        """
        Build a flat Box observation space for neural network input.
        
        Observation vector format (concatenated):
        - Truck positions: (num_trucks, 2)
        - Truck loads: (num_trucks,)
        - Truck capacities: (num_trucks,)
        - Truck utilization: (num_trucks,) [computed]
        - Customer locations: (num_customers, 2)
        - Customer weights: (num_customers,)
        - Unvisited mask: (num_customers,)
        
        Total size: 2*T + T + T + T + 2*N + N + N = 5*T + 4*N
        """
        T = self.num_trucks
        N = self.num_customers
        
        obs_size = 5 * T + 4 * N
        
        return spaces.Box(
            low=-10.0,
            high=27000.0,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert Dict observation to flat vector."""
        components = []
        
        truck_positions = obs_dict["truck_positions"].flatten()
        truck_loads = obs_dict["truck_loads"]
        truck_capacities = obs_dict["truck_capacities"]
        
        truck_utilization = truck_loads / (truck_capacities + 1e-8)
        
        customer_locations = obs_dict["customer_locations"].flatten()
        customer_weights = obs_dict["customer_weights"]
        unvisited_mask = obs_dict["unvisited_mask"].astype(np.float32)
        
        components.extend([
            truck_positions,
            truck_loads,
            truck_capacities,
            truck_utilization,
            customer_locations,
            customer_weights,
            unvisited_mask
        ])
        
        return np.concatenate(components).astype(np.float32)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return flat observation."""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        obs = self._flatten_observation(obs_dict)
        
        self._last_feasibility_mask = obs_dict["feasibility_mask"].copy()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute step with action masking validation.
        
        Args:
            action: MultiDiscrete action array
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=int)
        
        if self._last_feasibility_mask is not None:
            for truck_idx, action_val in enumerate(action):
                if not self._is_action_feasible(truck_idx, action_val):
                    feasible_actions = np.where(self._last_feasibility_mask[truck_idx] == 1)[0]
                    if len(feasible_actions) > 0:
                        action[truck_idx] = np.random.choice(feasible_actions)
                    else:
                        action[truck_idx] = self.num_customers
        
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        obs = self._flatten_observation(obs_dict)
        
        self._last_feasibility_mask = obs_dict["feasibility_mask"].copy()
        
        info["feasibility_mask"] = self._last_feasibility_mask
        
        return obs, reward, terminated, truncated, info
    
    def _is_action_feasible(self, truck_idx: int, action_val: int) -> bool:
        """Check if action is feasible according to last known mask."""
        if self._last_feasibility_mask is None:
            return True
        
        if truck_idx >= len(self._last_feasibility_mask):
            return False
        
        if action_val >= len(self._last_feasibility_mask[truck_idx]):
            return False
        
        return self._last_feasibility_mask[truck_idx, action_val] == 1
    
    def get_feasibility_mask(self) -> np.ndarray:
        """Get current feasibility mask for action masking."""
        if self._last_feasibility_mask is None:
            return np.ones((self.num_trucks, self.num_customers + 1), dtype=np.int8)
        
        return self._last_feasibility_mask.copy()
