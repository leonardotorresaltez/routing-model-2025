# Routing Model Data

This folder contains the datasets used by the routing model, generated from `initial_data_info.xlsx` using the `data_generator.ipynb` notebook.

## Contents

- **`data_version_1/`**: Scenario with 50 customers, 1 depot, and 5 trucks.
- **`data_version_2/`**: Scenario with 500 customers, 5 depots, and 50 trucks.
- **`initial_data_info.xlsx`**: Source file containing raw coordinates and node information.
- **`data_generator.ipynb`**: Logic for selecting nodes and calculating distance/time matrices.

## Generated Files (per version)

- `selected_customers.csv`: List of delivery locations chosen for the scenario.
- `selected_depot.csv`: The location of the starting depots.
- `selected_trucks.csv`: List of trucks assigned to the scenario.
- `time_between_nodes_X.csv`: Chunks of the distance and time matrix between all nodes.