# Routing Model Data

This folder contains the datasets used by the routing model, generated from `initial_data_info.xlsx` using the `data_generator.ipynb` notebook.

## Contents

- **`data_version_1/`**: Scenario with 50 customers, 1 depot, and 5 trucks. 24h limit. 
- **`data_version_2/`**: Scenario with 500 customers, 5 depots, and 50 trucks. 12h limit. 
- **`data_version_3/`**: Scenario with 10 customers, 1 depots, and 2 trucks. 12h limit. 
- **`initial_data_info.xlsx`**: Source file containing raw coordinates and node information.
- **`data_generator.ipynb`**: Logic for selecting nodes and calculating distance/time matrices.

## Generated Files (per version)

- `selected_customers.csv`: List of delivery locations chosen for the scenario.
- `selected_depot.csv`: The location of the starting depots.
- `selected_trucks.csv`: List of trucks assigned to the scenario.
- `time_between_nodes_X.csv`: Chunks of the distance and time matrix between all nodes.

## Benchmark results

- **`data_version_1/`**: Total destinations visited: 47, Total time: 37.57, Percentage of intersections: 2.47%
- **`data_version_2/`**: Total destinations visited: 493, Total time: 415.41, Percentage of intersections: 1.46%
- **`data_version_3/`**: Total destinations visited: 4, Total time: 15.00, Percentage of intersections: 0.00%