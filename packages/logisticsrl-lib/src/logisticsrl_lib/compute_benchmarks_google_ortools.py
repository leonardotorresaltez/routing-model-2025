import os
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from logisticsrl_lib.configs.config import parse_args
from loader_lib.data_loader import FleetStatus, MDVRPDataLoader, TruckState
from common_lib.evaluation_utils import evaluate_solution
from common_lib.visualization_utils_plotly import create_routing_graph, visualize_routing_solution

# We need to scale them because OR-Tools expects integer costs, and we want to preserve some precision.
TIME_SCALE = 100
SKIP_PENALTY = 10000


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def solve_with_ortools(time_matrix, truck_starts, max_time, time_limit=60*5):

    num_nodes = time_matrix.shape[0]
    num_vehicles = len(truck_starts)

    starts = truck_starts
    ends = truck_starts

    manager = pywrapcp.RoutingIndexManager(
        num_nodes,
        num_vehicles,
        starts,
        ends
    )

    routing = pywrapcp.RoutingModel(manager)

    # ----------------------------
    # Travel time callback
    # ----------------------------
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return int(time_matrix[from_node, to_node].item())

    transit_callback = routing.RegisterTransitCallback(time_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    # ----------------------------
    # Time constraint
    # ----------------------------
    routing.AddDimension(
        transit_callback,
        0,
        int(max_time),
        True,
        "Time"
    )

    time_dimension = routing.GetDimensionOrDie("Time")

    # ----------------------------
    # Allow skipping customers
    # ----------------------------
    

    for node in range(num_nodes):
        if node in truck_starts:
            continue

        routing.AddDisjunction(
            [manager.NodeToIndex(node)],
            SKIP_PENALTY
        )

    # ----------------------------
    # Search parameters
    # ----------------------------
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(search_parameters)

    tours = [[] for _ in range(num_vehicles)]

    if solution:
        for vehicle_id in range(num_vehicles):

            index = routing.Start(vehicle_id)
            route = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                index = solution.Value(routing.NextVar(index))

            if len(route) > 1:
                tours[vehicle_id] = route
            else:
                tours[vehicle_id] = [truck_starts[vehicle_id]]

    return tours


def train():

    cfg = parse_args()
    cfg.max_daily_delivery_time_each_truck = int(cfg.max_daily_delivery_time_each_truck * TIME_SCALE)

    set_seed(cfg.seed)

    os.makedirs("checkpoints", exist_ok=True)

    loader = MDVRPDataLoader(data_dir=cfg.data_dir)
    data = loader.load_data()

    node_to_idx = data["node_to_idx"]
    idx_to_node = data["idx_to_node"]

    cfg.num_nodes = data["num_nodes"]

    truck_starts = [truck.depot_idx for truck in data["trucks"]]

    print(f"Truck start positions (indices): {truck_starts}")

    time_matrix = data["time_matrix"]
    

    time_matrix = (time_matrix * TIME_SCALE).round().long()

    unique_truck_starts = list(set(truck_starts))

    time_matrix_from_truck_starts = time_matrix[unique_truck_starts, :]
    time_matrix_to_truck_starts = time_matrix[:, unique_truck_starts].T

    go_and_return_time_matrix = (
        time_matrix_from_truck_starts + time_matrix_to_truck_starts
    )

    time_wrt_closest_truck_start = go_and_return_time_matrix.min(dim=0).values

    unfeasible_nodes = (
        time_wrt_closest_truck_start > cfg.max_daily_delivery_time_each_truck
    ).nonzero(as_tuple=True)[0].tolist()

    print(
        f"Unfeasible nodes (time to closest truck start > {cfg.max_daily_delivery_time_each_truck}): {unfeasible_nodes}"
    )

    customers_df = data["customers_df"]

    customers_df["idx"] = customers_df["id_customer"].map(node_to_idx)

    customers_df = customers_df.set_index("idx")

    customers_df["feasible"] = True
    customers_df.loc[unfeasible_nodes, "feasible"] = False

    customers = data["customers"]
    depots = data["depots"]

    # ----------------------------
    # Solve with OR-Tools
    # ----------------------------

    tours = solve_with_ortools(
        time_matrix,
        truck_starts,
        cfg.max_daily_delivery_time_each_truck
    )

    print("Number of tours:", len(tours))

    # ----------------------------
    # Evaluate solution
    # ----------------------------

    total_destinations_visited, total_time, pct_intersections = evaluate_solution(
        tours,
        data,
        truck_starts,
        cfg
    )

    print(
        f"Total destinations visited: {total_destinations_visited}, "
        f"Total time: {total_time:.2f}, "
        f"Percentage of intersections: {pct_intersections:.2%}"
    )

    # ----------------------------
    # Mark delivered customers
    # ----------------------------

    visited_nodes = set(node for tour in tours for node in tour)

    for customer in customers:

        if customer.idx in visited_nodes:
            customer.delivered = True
        else:
            customer.delivered = False

    # ----------------------------
    # Visualization
    # ----------------------------

    G = create_routing_graph(
        depots,
        customers,
        tours,
        truck_starts
    )

    visualize_routing_solution(
        G,
        step=0,
        title_suffix="",
        save_path=f"checkpoints/viz_ortools_{cfg.data_dir}.html"
    )


if __name__ == "__main__":
    train()