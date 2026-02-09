# Logistics RL: Routing Optimization POC

This repository contains a modular Reinforcement Learning framework designed to solve Routing Problems (like TSP and VRP) using an Attention-based Pointer Network.

## Project Structure
```
routing-model-2025/
├── checkpoints/ # Pesos de modelos entrenados
├── data/ # Datos de entrada y salida
├── notebooks/ # Jupyter notebooks de experimentación
├── packages/
│ ├── logisticsrl-lib/
│ │ └── src/logisticsrl_lib/
│ │ ├── main.py # Script principal de entrenamiento
│ │ ├── configs/
│ │ │ └── config.py # Configuración y argumentos
│ │ └── reinforcelearning/
│ │ ├── agent.py # Lógica del agente RL
│ │ ├── policy.py # Arquitectura de la policy
│ │ └── tsp_env.py # Entorno Gymnasium
│ ├── loader-lib/
│ │ └── src/loader_lib/
│ │ └── data_loader.py # Carga y procesamiento de datos
│ └── common-lib/
│ └── src/common_lib/
│ ├── evaluation_utils.py # Utilidades de evaluación
│ └── visualization_utils_plotly.py # Visualización de rutas
├── run_experiments.py # Orquestador de experimentos/grid search
├── requirements.txt # Dependencias para entorno base
├── pyproject.toml # Configuración de Poetry y scripts
├── README.md # Documentación principal
└── wandb/ # Logs de experimentos W&B
```
---

## Installation & Setup


0. **Install Poetry**
   Poetry is the recommended dependency manager for this project. You can install it by following the official documentation:
   - **Mac/Linux:**
     ```bash
     curl -sSL https://install.python-poetry.org | python3 -
     ```
   - **Windows (PowerShell):**
     ```powershell
     (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
     ```
   More details and alternative methods: https://python-poetry.org/docs/#installation

1. **Install Dependencies**
   Use the provided requirements file to set up your environment:
   ```bash
cd packages/logisticsrl-lib && poetry install
cd ../loader-lib && poetry install
cd ../common-lib && poetry install
   ```

2. **Initialize Weights & Biases (Optional)**
   This project is integrated with W&B for real-time experiment tracking.
   ```bash
   wandb login
   ```

3. **Update dependencies after adding a new package**
   If you add a new dependency to any of the Poetry-managed packages, run:
   ```bash
   poetry update
   ```
   inside the corresponding package directory (e.g., `packages/logisticsrl-lib`). This will update the lock file and install the new dependency.

4. **Run the training script with Poetry**
   To execute the main training script using Poetry's script system, run:
   ```bash
   poetry run train
   ```
   This will call the `train`  function defined in `main.py` of the `logisticsrl-lib` package, ensuring the correct environment and dependencies are used.   

5. **Poetry, virtual environments, and Visual Studio Code**
   Poetry automatically creates and manages a virtual environment (venv) for each project. To use this venv in Visual Studio Code:
   - Run `poetry env info --path` inside your package directory to get the path to the Poetry-managed venv.
   - In VS Code, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), search for "Python: Select Interpreter", and choose the interpreter that matches the path shown by Poetry.
   - This ensures that scripts and notebooks use the correct environment with all dependencies installed by Poetry.


---
