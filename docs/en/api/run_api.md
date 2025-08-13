# Run API Reference

The `Run` class manages individual experiment runs within a MINTO experiment. Each run represents a single iteration of an optimization experiment with its own parameters, solutions, and other run-specific data.

## Class Overview

```python
from minto import Experiment

# Create an experiment
experiment = Experiment(name="optimization_study")

# Create a run explicitly
run = experiment.run()

# Use the run in a context manager
with run:
    run.log_parameter("iteration", 1)
    run.log_solution("result", solution)
```

## Methods

### `log_parameter(name: str, value: float | int | str | list | dict | np.ndarray)`

Log a parameter specific to this run.

**Parameters:**
- `name` (str): Name of the parameter
- `value`: Parameter value (can be scalar or complex data structure)

**Example:**
```python
run = experiment.run()
with run:
    run.log_parameter("temperature", 1000)
    run.log_parameter("cooling_rate", 0.95)
    run.log_parameter("hyperparameters", {"lr": 0.01, "momentum": 0.9})
```

### `log_params(params: dict[str, float | int | str])`

Log multiple parameters at once.

**Parameters:**
- `params` (dict): Dictionary of parameter names and values

**Example:**
```python
run = experiment.run()
with run:
    run.log_params({
        "population_size": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8
    })
```

### `log_solution(name: str, solution: ommx_v1.Solution)`

Log an optimization solution to this run.

**Parameters:**
- `name` (str): Name identifier for the solution
- `solution` (ommx_v1.Solution): OMMX solution object

**Example:**
```python
run = experiment.run()
with run:
    # Solve optimization problem
    solution = solver.solve(problem)
    run.log_solution("optimal_solution", solution)
```

### `log_sampleset(name: str, sampleset: jm.SampleSet)`

Log a JijModeling sampleset to this run.

**Parameters:**
- `name` (str): Name identifier for the sampleset
- `sampleset` (jm.SampleSet): JijModeling sampleset object

**Example:**
```python
run = experiment.run()
with run:
    # Generate samples
    sampleset = sampler.sample(problem)
    run.log_sampleset("samples", sampleset)
```

### `log_object(name: str, obj: typ.Any)`

Log any arbitrary object to this run.

**Parameters:**
- `name` (str): Name identifier for the object
- `obj`: Any Python object (must be JSON-serializable)

**Example:**
```python
run = experiment.run()
with run:
    results = {
        "best_fitness": 0.95,
        "convergence_history": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        "final_population": population_array
    }
    run.log_object("ga_results", results)
```

### `log_config(name: str, value: dict[str, typ.Any])`

Log a configuration object to this run. This is the preferred method for logging configuration dictionaries.

**Parameters:**
- `name` (str): Name of the configuration
- `value` (dict): Configuration dictionary

**Example:**
```python
run = experiment.run()
with run:
    solver_config = {
        "algorithm": "simulated_annealing",
        "max_iterations": 1000,
        "early_stopping": True
    }
    run.log_config("solver_config", solver_config)
```

## Context Manager Usage

The `Run` class should be used as a context manager to ensure proper resource management:

```python
# Correct usage
run = experiment.run()
with run:
    # All run-specific logging happens here
    run.log_parameter("seed", 42)
    # ... perform optimization ...
    run.log_solution("result", solution)
# Run is automatically closed after the with block

# Attempting to log after closing will raise an error
run.log_parameter("late_param", 123)  # RuntimeError!
```

## Complete Example

```python
import minto
import numpy as np

# Create experiment
experiment = minto.Experiment(
    name="algorithm_comparison",
    auto_saving=True
)

# Log experiment-level data
experiment.log_global_parameter("problem_size", 100)
experiment.log_global_problem("tsp", tsp_problem)

# Run multiple iterations
for seed in [42, 123, 456]:
    run = experiment.run()
    with run:
        # Log run-specific parameters
        run.log_parameter("random_seed", seed)
        run.log_parameter("start_time", datetime.now().isoformat())
        
        # Configure and run solver
        np.random.seed(seed)
        solution = genetic_algorithm(
            problem=tsp_problem,
            population_size=50,
            generations=100
        )
        
        # Log results
        run.log_solution("ga_solution", solution)
        run.log_parameter("objective_value", solution.objective)
        run.log_parameter("computation_time", solution.elapsed_time)
        
        # Log additional analysis
        run.log_config("final_stats", {
            "best_fitness": solution.objective,
            "feasible": solution.is_feasible,
            "iterations": solution.iterations
        })

# Analyze results
results_df = experiment.get_run_table()
print(f"Average objective: {results_df['parameter']['objective_value'].mean()}")
```

## Migration from Old API

If you're migrating from the old implicit context behavior:

```python
# Old approach (deprecated)
with experiment.run():
    experiment.log_parameter("param", value)
    experiment.log_solution("sol", solution)

# New approach (explicit)
run = experiment.run()
with run:
    run.log_parameter("param", value)
    run.log_solution("sol", solution)
```

The new explicit approach makes it clear which data belongs to the run level versus the experiment level, improving code clarity and preventing confusion.