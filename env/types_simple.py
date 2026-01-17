"""
Data types for simplified fleet routing problem.

This module defines the core data structures:
- Customer: delivery location and weight
- Truck: capacity and home depot
- Depot: loading point location
- RouteState: current truck state during episode
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Customer:
    """
    Represents a customer/delivery location.
    
    Attributes:
        id: unique customer identifier (0 to N-1)
        x, y: 2D location coordinates
        weight: delivery weight in kg
    """
    id: int
    x: float
    y: float
    weight: float
    
    def location(self) -> Tuple[float, float]:
        """Return location as (x, y) tuple."""
        return (self.x, self.y)


@dataclass
class Truck:
    """
    Represents a delivery truck.
    
    Attributes:
        id: unique truck identifier (0 to T-1)
        depot_id: home depot ID (0 to D-1)
        max_capacity: maximum weight truck can carry (kg)
    """
    id: int
    depot_id: int
    max_capacity: float


@dataclass
class Depot:
    """
    Represents a loading/depot location.
    
    Attributes:
        id: unique depot identifier (0 to D-1)
        x, y: 2D location coordinates
    """
    id: int
    x: float
    y: float
    
    def location(self) -> Tuple[float, float]:
        """Return location as (x, y) tuple."""
        return (self.x, self.y)


@dataclass
class TruckState:
    """
    Represents the state of a single truck during an episode.
    
    Attributes:
        truck: Truck object
        current_location: (x, y) coordinates (changes as truck moves)
        current_load: total weight of assigned customers (kg)
        visited_customers: list of customer IDs already delivered
        unvisited_customers: list of customer IDs not yet assigned
    """
    truck: Truck
    current_location: Tuple[float, float]
    current_load: float = 0.0
    visited_customers: List[int] = field(default_factory=list)
    
    def utilization(self) -> float:
        """Return current load utilization as percentage (0.0 to 1.0)."""
        return self.current_load / self.truck.max_capacity if self.truck.max_capacity > 0 else 0.0
    
    def remaining_capacity(self) -> float:
        """Return remaining weight capacity."""
        return self.truck.max_capacity - self.current_load
