# Routing Model Data

*Version 1*

This folder contains the datasets used by the routing model, generated from `initial_data_info.xlsx` using the `data_generator.ipynb` notebook.

## Contents

- **`distances_to_depot_version_50_deliveries/`**: Data for simulations with 50 delivery points.
- **`distances_to_depot_version_500_deliveries/`**: Data for simulations with 500 delivery points.
- **`initial_data_info.xlsx`**: Source file containing raw coordinates and node information.
- **`data_generator.ipynb`**: Logic for selecting nodes and calculating distance matrices.

## Generated Files (per version)

- `selected_customers.csv`: List of delivery locations chosen for the scenario.
- `selected_depot.csv`: The location of the starting depots.
- `distances_depot_costumers_X.csv`: Chunks of the distance and time matrix between nodes.