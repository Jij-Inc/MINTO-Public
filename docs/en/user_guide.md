# MINTO User Guide: From Beginner to Expert

## Getting Started

### Installation and Setup

```bash
pip install minto
```

MINTO automatically handles dependencies including:
- OMMX (Open Mathematical Modeling Exchange)
- JijModeling (Mathematical modeling framework)
- Pandas (Data analysis)
- NumPy (Numerical computing)

### Your First Experiment

```python
import minto

# Create an experiment with automatic environment tracking
experiment = minto.Experiment(
    name="my_first_optimization",
    collect_environment=True  # Default: captures system info
)

# Log experiment-level parameters (shared across all runs)
experiment.log_global_parameter("algorithm", "greedy")
experiment.log_global_parameter("max_iterations", 1000)

# Create and run an optimization iteration
run = experiment.run()
with run:
    run.log_parameter("iteration", 1)
    run.log_parameter("objective_value", 42.5)
    run.log_parameter("solve_time", 0.123)

# Save and view results
experiment.save()
print(experiment.get_run_table())
```

## API Design Philosophy

MINTO uses an explicit separation between experiment-level and run-level data:

- **Experiment-level data**: Shared across all runs (problems, instances, global parameters)
- **Run-level data**: Specific to each run iteration (solutions, run parameters)

### Experiment-Level Methods
These methods store data that is shared across all runs:

- `experiment.log_global_parameter()` - For experiment-wide parameters
- `experiment.log_global_problem()` - For problem definitions
- `experiment.log_global_instance()` - For problem instances
- `experiment.log_global_config()` - For configuration objects

### Run-Level Methods
These methods store data specific to individual runs:

```python
# Create an explicit run object
run = experiment.run()
with run:
    run.log_parameter("iteration", 1)
    run.log_solution("sol", solution)
    run.log_sampleset("samples", sampleset)
```

This explicit separation makes it clear where data is stored and eliminates confusion about storage context.

## Core Concepts in Detail

### Understanding the Experiment Object

The `Experiment` class is your primary interface to MINTO. It manages:

```python
# Basic experiment creation
experiment = minto.Experiment(
    name="optimization_study",           # Experiment identifier
    savedir="./my_experiments",         # Storage directory
    auto_saving=True,                   # Automatic persistence
    collect_environment=True            # Environment metadata
)

# Key properties
print(f"Experiment name: {experiment.name}")
print(f"Timestamp: {experiment.timestamp}")
print(f"Running: {experiment._running}")
print(f"Current run ID: {experiment._run_id}")
```

### The Data Hierarchy

MINTO uses an explicit separation between experiment-level and run-level data:
- **Experiment-level data**: Shared across all runs (problems, instances, global parameters)
- **Run-level data**: Specific to each run iteration (solutions, run parameters)

This explicit separation replaces the previous implicit with-clause behavior, making it clearer where data is stored.

#### Experiment-Level Data (Shared)

```python
# Problems: Mathematical formulations
import jijmodeling as jm

problem = jm.Problem("traveling_salesman")
n = jm.Placeholder("n")
x = jm.BinaryVar("x", shape=(n, n))
# ... define objective and constraints ...

experiment.log_global_problem("tsp", problem)

# Instances: Concrete problem data
cities = load_tsp_cities("berlin52.tsp")
instance = create_tsp_instance(cities)
experiment.log_global_instance("berlin52", instance)

# Config objects: Use the dedicated config method
algorithm_config = {
    "population_size": 100,
    "elite_ratio": 0.1,
    "crossover_operators": ["ox", "pmx"]
}
experiment.log_global_config("genetic_config", algorithm_config)
```

#### Run-Level Data (Per Iteration)

```python
# Multiple runs with different parameters
temperatures = [100, 500, 1000, 2000]

for temp in temperatures:
    run = experiment.run()  # Create explicit run object
    with run:  # Context manager for automatic cleanup
        # Log run-specific parameters
        run.log_parameter("temperature", temp)
        run.log_parameter("cooling_rate", 0.95)
        
        # Solve and log results
        solution = simulated_annealing(problem, temp)
        run.log_solution("sa_solution", solution)
        
        # Log performance metrics
        run.log_parameter("objective", solution.objective)
        run.log_parameter("feasible", solution.is_feasible)
        run.log_parameter("runtime", solution.elapsed_time)
```

### Data Types and Storage

#### Simple Parameters

MINTO handles various data types automatically:

```python
# Experiment-level parameters (shared across runs)
experiment.log_global_parameter("learning_rate", 0.001)      # float
experiment.log_global_parameter("population_size", 100)      # int
experiment.log_global_parameter("algorithm", "genetic")      # str

# Collections
experiment.log_global_parameter("layer_sizes", [64, 128, 64])           # list
experiment.log_global_parameter("hyperparams", {"lr": 0.01, "decay": 0.9})  # dict

# NumPy arrays
import numpy as np
experiment.log_global_parameter("weights", np.array([0.1, 0.5, 0.4]))   # ndarray

# Run-specific parameters (different for each iteration)
run = experiment.run()
with run:
    run.log_parameter("iteration", 1)
    run.log_parameter("current_loss", 0.234)
    run.log_parameter("batch_size", 32)
```

#### Complex Objects

For optimization-specific objects, MINTO provides specialized methods:

```python
# Experiment-level objects (shared data)
experiment.log_global_problem("knapsack", knapsack_problem)
experiment.log_global_instance("test_case", problem_instance)

# Run-specific objects (per iteration)
run = experiment.run()
with run:
    run.log_solution("optimal_solution", solution)
    
    # Samplesets (OMMX/JijModeling)
    run.log_sampleset("samples", sample_collection)
```

## Advanced Features

### Automatic Solver Integration

The `log_solver` decorator captures solver behavior automatically:

```python
# Decorator approach
@experiment.log_solver
def my_genetic_algorithm(problem, population_size=100, generations=1000):
    """
    All parameters (population_size, generations) are logged automatically.
    Problem objects are captured and stored appropriately.
    Return values are analyzed and logged based on type.
    """
    # Implementation here
    return best_solution

# Usage automatically logs everything
result = my_genetic_algorithm(tsp_problem, population_size=50)

# Explicit approach with parameter exclusion
solver = experiment.log_solver(
    "genetic_algorithm", 
    my_genetic_algorithm,
    exclude_params=["debug_mode"]  # Don't log debug parameters
)
result = solver(tsp_problem, population_size=50, debug_mode=True)
```

### Environment Metadata

MINTO automatically captures comprehensive environment information:

```python
# View captured environment data
experiment.print_environment_summary()
# Output:
# Environment Summary:
# OS: macOS 14.1.1
# Python: 3.11.5
# CPU: Apple M2 Pro (12 cores)
# Memory: 32.0 GB
# Virtual Environment: /opt/conda/envs/minto
# Key Packages:
#   - minto: 1.0.0
#   - jijmodeling: 1.7.0
#   - ommx: 0.2.0

# Access detailed environment info
env_info = experiment.get_environment_info()
print(env_info["hardware"]["cpu_count"])  # 12
print(env_info["packages"]["numpy"])       # "1.24.3"
```

### Data Persistence

#### Directory-Based Storage

```python
# Automatic saving (default)
experiment = minto.Experiment("study", auto_saving=True)
# Data saved after each log_* call

# Manual saving
experiment = minto.Experiment("study", auto_saving=False)
# ... do work ...
experiment.save()  # Explicit save

# Custom save location
experiment.save("/path/to/custom/directory")

# Loading
loaded_experiment = minto.Experiment.load_from_dir(
    "/path/to/saved/experiment"
)
```

#### OMMX Archive Format

```python
# Save as portable archive
artifact = experiment.save_as_ommx_archive("study.ommx")

# Load from archive
experiment = minto.Experiment.load_from_ommx_archive("study.ommx")

# Share via GitHub (requires setup)
experiment.push_github(
    org="my-organization",
    repo="optimization-studies",
    name="parameter_sweep_v1"
)
```

### Data Analysis and Visualization

#### Table Generation

```python
# Run-level results table
results = experiment.get_run_table()
print(results.head())
#   run_id algorithm  temperature  objective  solve_time
# 0      0        sa          100      -1234       0.45
# 1      1        sa          500       -987       0.32
# 2      2        sa         1000       -856       0.28

# Experiment-level tables
tables = experiment.get_experiment_tables()
print(tables["problems"].head())      # Problem definitions
print(tables["instances"].head())     # Instance data
print(tables["parameters"].head())    # All parameters
```

#### Data Analysis Patterns

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Performance analysis
results = experiment.get_run_table()

# Temperature vs. objective value
plt.figure(figsize=(10, 6))
plt.plot(results["parameter"]["temperature"], 
         results["parameter"]["objective"], 'o-')
plt.xlabel("Temperature")
plt.ylabel("Objective Value")
plt.title("Simulated Annealing: Temperature Sensitivity")
plt.show()

# Algorithm comparison (if multiple algorithms)
if "algorithm" in results["parameter"].columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results, x=("parameter", "algorithm"), 
                y=("parameter", "objective"))
    plt.title("Algorithm Performance Comparison")
    plt.show()

# Convergence analysis
convergence_data = results[["parameter"]].copy()
convergence_data["iteration"] = range(len(results))
plt.plot(convergence_data["iteration"], 
         convergence_data[("parameter", "objective")])
plt.xlabel("Iteration")
plt.ylabel("Best Objective")
plt.title("Convergence History")
plt.show()
```

## Best Practices and Patterns

### Experiment Design

#### 1. Focused Experiments

```python
# Good: Clear, focused scope
experiment_temp = minto.Experiment("sa_temperature_analysis")
experiment_crossover = minto.Experiment("ga_crossover_comparison")

# Avoid: Mixed, unfocused experiments
experiment_everything = minto.Experiment("all_optimization_work")
```

#### 2. Systematic Parameter Sweeps

```python
# Structured parameter exploration
experiment = minto.Experiment("hyperparameter_optimization")

# Define parameter space
param_space = {
    "population_size": [50, 100, 200, 400],
    "crossover_rate": [0.6, 0.7, 0.8, 0.9],
    "mutation_rate": [0.01, 0.05, 0.1, 0.2]
}

# Systematic exploration
from itertools import product

for pop_size, cx_rate, mut_rate in product(*param_space.values()):
    run = experiment.run()
    with run:
        # Log parameter combination
        run.log_parameter("population_size", pop_size)
        run.log_parameter("crossover_rate", cx_rate)
        run.log_parameter("mutation_rate", mut_rate)
        
        # Run optimization
        solution = genetic_algorithm(
            problem=tsp_problem,
            population_size=pop_size,
            crossover_rate=cx_rate,
            mutation_rate=mut_rate
        )
        
        # Log results
        run.log_solution("ga_solution", solution)
        run.log_parameter("objective", solution.objective)
        run.log_parameter("generations", solution.generations)
```

#### 3. Comprehensive Logging

```python
run = experiment.run()
with run:
    start_time = time.time()
    
    # Algorithm configuration (run-specific)
    run.log_parameter("algorithm", "genetic_algorithm")
    run.log_parameter("population_size", 100)
    run.log_parameter("max_generations", 1000)
    
    # Problem characteristics (could be experiment-level if shared)
    run.log_parameter("problem_size", len(cities))
    run.log_parameter("problem_type", "symmetric_tsp")
    
    # Run optimization
    solution = optimize()
    
    # Solution quality
    run.log_parameter("objective_value", solution.objective)
    run.log_parameter("feasible", solution.is_feasible)
    run.log_parameter("optimality_gap", solution.gap)
    
    # Performance metrics
    run.log_parameter("solve_time", time.time() - start_time)
    run.log_parameter("iterations", solution.iterations)
    run.log_parameter("evaluations", solution.function_evaluations)
    
    # Solution itself
    run.log_solution("best_solution", solution)
```

### Data Organization

#### Naming Conventions

```python
# Use descriptive, searchable names
experiment.log_global_parameter("simulated_annealing_temperature", 1000)
experiment.log_global_parameter("genetic_algorithm_population_size", 100)

# Run-level solution logging
run = experiment.run()
with run:
    run.log_solution("clarke_wright_routes", cw_solution)

# Include units when relevant
experiment.log_global_parameter("time_limit_seconds", 300)
experiment.log_global_parameter("memory_limit_mb", 2048)

# Use consistent prefixes for related parameters
experiment.log_global_parameter("sa_temperature", 1000)
experiment.log_global_parameter("sa_cooling_rate", 0.95)
experiment.log_global_parameter("sa_min_temperature", 0.01)
```

#### Experiment Comparison

```python
# Combine related experiments for analysis
experiments = [
    minto.Experiment.load_from_dir("genetic_algorithm_study"),
    minto.Experiment.load_from_dir("simulated_annealing_study"),
    minto.Experiment.load_from_dir("tabu_search_study")
]

# Create combined analysis
combined = minto.Experiment.concat(
    experiments, 
    name="algorithm_comparison_meta_study"
)

# Analyze across all algorithms
results = combined.get_run_table()
algorithm_performance = results.groupby(("parameter", "algorithm")).agg({
    ("parameter", "objective"): ["mean", "std", "min", "max"],
    ("parameter", "solve_time"): ["mean", "std"]
})
print(algorithm_performance)
```

## Troubleshooting

### Common Issues

#### Data Not Persisting

```python
# Problem: auto_saving=False but no manual save
experiment = minto.Experiment("study", auto_saving=False)
# ... do work ...
# Data lost if no explicit save!

# Solution: Enable auto_saving or call save()
experiment = minto.Experiment("study", auto_saving=True)
# OR
experiment.save()  # Manual save
```

#### Parameter Type Errors

```python
# Problem: Non-serializable objects as parameters
class CustomObject:
    pass

experiment.log_parameter("custom", CustomObject())  # Raises ValueError

# Solution: Use log_object for complex data
experiment.log_object("custom_config", {
    "type": "CustomObject",
    "parameters": {"value": 42}
})
```

#### Run Context Confusion

```python
# OLD: Confusing implicit context behavior (DEPRECATED)
experiment.log_parameter("global_param", "value")  # Was experiment level

with experiment.run():
    experiment.log_parameter("run_param", "value")  # Was run level due to context

# NEW: Explicit and clear separation
experiment.log_global_parameter("global_param", "value")  # Always experiment level

run = experiment.run()
with run:
    run.log_parameter("run_param", "value")  # Clearly run level
```

### Performance Optimization

#### Large Datasets

```python
# For large solution objects, consider storing summaries
run = experiment.run()
with run:
    large_solution = solve_large_problem()
    
    # Store summary instead of full solution
    run.log_parameter("objective", large_solution.objective)
    run.log_parameter("solution_size", len(large_solution.variables))
    run.log_parameter("nonzero_count", large_solution.nonzeros)
    
    # Store full solution selectively
    if large_solution.is_optimal:
        run.log_solution("optimal_solution", large_solution)
```

#### Batch Processing

```python
# Efficient parameter logging
parameter_batch = {
    "algorithm": "genetic",
    "population_size": 100,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1
}

run = experiment.run()
with run:
    run.log_params(parameter_batch)  # Single call for multiple params
```

## Integration Examples

### With Popular Optimization Libraries

#### OR-Tools Integration

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

experiment = minto.Experiment("ortools_vrp_study")

@experiment.log_solver
def solve_vrp_ortools(distance_matrix, vehicle_count=1, depot=0):
    # OR-Tools VRP solving
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), vehicle_count, depot)
    routing = pywrapcp.RoutingModel(manager)
    
    # Define cost callback and add cost dimension
    # ... OR-Tools specific code ...
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    return extract_solution(solution, manager, routing)

# Usage automatically logs parameters and results
vrp_solution = solve_vrp_ortools(
    distance_matrix=cities_distances,
    vehicle_count=3,
    depot=0
)
```

#### CPLEX Integration (via OMMX)

```python
import ommx_cplex_adapter as cplex_ad

experiment = minto.Experiment("cplex_milp_study")

# Log problem and instance
experiment.log_global_problem("facility_location", milp_problem)
experiment.log_global_instance("northeast_facilities", problem_instance)

time_limits = [60, 300, 1800]  # 1min, 5min, 30min

for time_limit in time_limits:
    run = experiment.run()
    with run:
        run.log_parameter("time_limit_seconds", time_limit)
        run.log_parameter("solver", "cplex")
        
        # Solve with CPLEX
        adapter = cplex_ad.OMMXCPLEXAdapter(problem_instance)
        adapter.set_time_limit(time_limit)
        
        solution = adapter.solve()
        
        run.log_solution("cplex_solution", solution)
        run.log_parameter("objective", solution.objective)
        run.log_parameter("solve_status", str(solution.status))
        run.log_parameter("gap", solution.mip_gap)
```

### Jupyter Notebook Integration

```python
# Notebook-friendly experiment management
%matplotlib inline
import matplotlib.pyplot as plt

# Create experiment
experiment = minto.Experiment("notebook_optimization_study")

# Interactive parameter exploration
from ipywidgets import interact, FloatSlider

@interact(temperature=FloatSlider(min=1, max=1000, step=10, value=100))
def run_experiment(temperature):
    run = experiment.run()
    with run:
        run.log_parameter("temperature", temperature)
        
        solution = simulated_annealing(problem, temperature)
        run.log_solution("sa_solution", solution)
        run.log_parameter("objective", solution.objective)
        
        # Real-time visualization
        results = experiment.get_run_table()
        plt.figure(figsize=(8, 4))
        plt.plot(results["parameter"]["temperature"], 
                results["parameter"]["objective"], 'o-')
        plt.xlabel("Temperature")
        plt.ylabel("Objective")
        plt.title("SA Performance vs Temperature")
        plt.grid(True)
        plt.show()
        
        print(f"Latest result: {solution.objective:.2f}")
```

This comprehensive user guide provides practical knowledge for effectively using MINTO in real optimization research and development workflows. By following these patterns and best practices, users can build robust, reproducible optimization experiments.
