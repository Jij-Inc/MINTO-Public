# Understanding MINTO: Mental Model and Design Philosophy

## What is MINTO?

MINTO is the "MLflow for optimization" - a Python framework designed to systematically manage, track, and analyze mathematical optimization experiments. Just as MLflow revolutionized machine learning experiment tracking, MINTO brings structured experiment management to the optimization domain.

## Core Mental Model

### The Two-Level Storage Architecture

At its heart, MINTO operates on a **two-level storage architecture** that mirrors how optimization research is naturally organized:

1. **Experiment Level** - Shared, reusable data:
   - **Problems**: Mathematical formulations that define optimization objectives and constraints
   - **Instances**: Concrete data that parameterizes problems (e.g., graph structures, matrices)
   - **Objects**: Custom data structures and configurations
   - **Environment Metadata**: System information for reproducibility

2. **Run Level** - Iteration-specific data:
   - **Parameters**: Algorithm settings, hyperparameters, and control values
   - **Solutions**: Optimization results and solution vectors
   - **Samplesets**: Collections of solution samples (for stochastic solvers)
   - **Performance Metrics**: Timing, objective values, and quality measures

This separation reflects the natural workflow where researchers define a problem once but run many variations with different parameters, algorithms, or configurations.

### The Explicit Run Creation Pattern

MINTO uses explicit run creation to clearly separate experiment-wide and run-specific data:

```python
import minto

# Create experiment
experiment = minto.Experiment("my_optimization_study")

# Log experiment-wide data (shared across all runs)
experiment.log_problem("tsp", traveling_salesman_problem)
experiment.log_instance("berlin52", berlin_52_instance)

# Log run-specific data (unique to each execution)
for temperature in [100, 500, 1000]:
    run = experiment.run()  # Explicitly create a new run
    with run:  # Use run context for automatic cleanup
        run.log_parameter("temperature", temperature)
        solution = solve_with_simulated_annealing(temperature)
        run.log_solution("sa_result", solution)
```

This pattern ensures:
- **Clear separation of concerns**: Experiment vs run data is explicit
- **No implicit behavior**: Where data is stored is always obvious
- **Better mental model**: Matches natural optimization workflow
- Systematic data organization
- Easy analysis across runs

## Key Design Principles

### 1. Reproducibility by Default

MINTO automatically captures environment metadata including:
- Operating system and version
- Hardware specifications (CPU, memory)
- Python version and virtual environment
- Package versions for optimization libraries
- Execution timestamps

This ensures experiments can be reproduced across different environments and systems.

### 2. Flexible Data Types

MINTO supports both simple and complex data structures:

**Simple Types** (stored as parameters):
- Scalars: `int`, `float`, `str`
- Basic collections: `list`, `dict`
- Arrays: `numpy.ndarray`

**Complex Types** (stored as objects):
- Optimization problems (`jijmodeling.Problem`)
- Problem instances (`ommx.v1.Instance`)
- Solutions (`ommx.v1.Solution`)
- Samplesets (`ommx.v1.SampleSet`, `jijmodeling.SampleSet`)

### 3. Multiple Storage Formats

MINTO provides flexibility in how experiments are persisted:

- **Directory-based storage**: Human-readable, version-control friendly
- **OMMX Archives**: Standardized, portable binary format
- **GitHub integration**: Direct sharing and collaboration support

### 4. Automatic Solver Integration

The `log_solver` method can be used at both experiment and run levels:

```python
# For experiment-level solver logging (problem/instance registration)
@experiment.log_solver
def setup_problem(problem_data):
    # Problem setup logic
    return problem_instance

# For run-level solver logging (algorithm execution)
run = experiment.run()
with run:
    @run.log_solver
    def my_optimization_solver(problem, temperature=1000, iterations=10000):
        # Solver implementation
        return solution
    
    # All parameters and results are automatically logged to this run
    result = my_optimization_solver(tsp_problem, temperature=500)
```

## Understanding the Data Flow

### Experiment Lifecycle

1. **Initialization**: Create experiment with automatic environment capture
2. **Setup**: Log problems, instances, and shared objects
3. **Execution**: Run optimization iterations within context managers
4. **Analysis**: Generate tables and visualizations from logged data
5. **Persistence**: Save to disk or share via archives

### Data Relationships

```text
Experiment
├── Environment Metadata (automatic)
├── Problems (shared across runs)
├── Instances (shared across runs)
├── Objects (shared configurations)
└── Runs
    ├── Run 0
    │   ├── Parameters (algorithm settings)
    │   ├── Solutions (optimization results)
    │   └── Metadata (run-specific info)
    ├── Run 1
    │   ├── Parameters
    │   ├── Solutions
    │   └── Metadata
    └── ...
```

### Table Generation

MINTO automatically generates structured tables for analysis:

```python
# Get run-level results table
results = experiment.get_run_table()
print(results)
#   run_id  temperature  objective_value  solve_time
# 0      0          100            -1234        0.45
# 1      1          500             -987        0.32
# 2      2         1000             -856        0.28

# Access detailed experiment information
tables = experiment.get_experiment_tables()
# Returns: problems, instances, objects, parameters, solutions tables
```

## Best Practices and Usage Patterns

### 1. Experiment Organization

**Structure your experiments by research question:**
```python
# Good: Focused experiment scope
experiment = minto.Experiment("temperature_sensitivity_analysis")

# Avoid: Mixing unrelated studies
experiment = minto.Experiment("all_my_optimization_work")
```

**Use meaningful names:**
```python
# Good: Descriptive and searchable
experiment.log_parameter("simulated_annealing_temperature", 1000)
experiment.log_solution("best_tour", optimal_solution)

# Avoid: Cryptic abbreviations
experiment.log_parameter("sa_temp", 1000)
experiment.log_solution("sol", optimal_solution)
```

### 2. Data Logging Strategy

**Separate concerns by storage level:**
```python
# Experiment level: Problem definition and shared data
experiment.log_problem("vehicle_routing", vrp_problem)
experiment.log_instance("customer_locations", berlin_customers)

# Run level: Algorithm parameters and results
with experiment.run():
    experiment.log_parameter("vehicle_capacity", 100)
    experiment.log_parameter("algorithm", "clarke_wright")
    experiment.log_solution("routes", best_routes)
```

**Log comprehensive metadata:**
```python
with experiment.run():
    # Algorithm configuration
    experiment.log_parameter("population_size", 100)
    experiment.log_parameter("crossover_rate", 0.8)
    experiment.log_parameter("mutation_rate", 0.1)
    
    # Performance metrics
    experiment.log_parameter("objective_value", solution.objective)
    experiment.log_parameter("solve_time", elapsed_time)
    experiment.log_parameter("iterations", total_iterations)
    
    # Solution quality
    experiment.log_parameter("feasible", solution.is_feasible)
    experiment.log_parameter("optimality_gap", gap_percentage)
```

### 3. Solver Integration

**Use the log_solver decorator for automatic capture:**
```python
# Automatic parameter and result logging
@experiment.log_solver
def genetic_algorithm(problem, population_size=100, generations=1000):
    # Implementation
    return best_solution

# Or explicit solver logging
solver = experiment.log_solver("genetic_algorithm", genetic_algorithm)
result = solver(tsp_problem, population_size=50)
```

**Handle complex solver outputs:**
```python
with experiment.run():
    result = complex_solver(problem)
    
    # Log multiple solution components
    experiment.log_solution("primary_solution", result.best_solution)
    experiment.log_parameter("convergence_history", result.objective_history)
    experiment.log_parameter("computation_stats", result.statistics)
```

### 4. Analysis and Visualization

**Generate comparative analysis:**
```python
# Load and combine experiments
experiments = [
    minto.Experiment.load_from_dir("exp_genetic_algorithm"),
    minto.Experiment.load_from_dir("exp_simulated_annealing"),
    minto.Experiment.load_from_dir("exp_tabu_search")
]

combined = minto.Experiment.concat(experiments, name="algorithm_comparison")
results = combined.get_run_table()

# Analyze performance across algorithms
import matplotlib.pyplot as plt
results.groupby("algorithm")["objective_value"].mean().plot(kind="bar")
plt.title("Algorithm Performance Comparison")
plt.show()
```

## Common Usage Patterns

### 1. Parameter Sweeps

```python
experiment = minto.Experiment("parameter_sensitivity")
experiment.log_problem("quadratic_assignment", qap_problem)

for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        with experiment.run():
            experiment.log_parameter("penalty_alpha", alpha)
            experiment.log_parameter("penalty_beta", beta)
            
            solution = solve_with_penalties(alpha, beta)
            experiment.log_solution("penalized_solution", solution)
            experiment.log_parameter("objective", solution.objective)
```

### 2. Algorithm Benchmarking

```python
experiment = minto.Experiment("solver_benchmark")

# Load standard benchmark instances
for instance_name in ["kroA100", "kroB100", "kroC100"]:
    instance = load_tsplib_instance(instance_name)
    experiment.log_instance(instance_name, instance)

algorithms = ["genetic", "simulated_annealing", "ant_colony"]

for algorithm in algorithms:
    for instance_name in experiment.dataspace.experiment_datastore.instances:
        with experiment.run():
            experiment.log_parameter("algorithm", algorithm)
            experiment.log_parameter("instance", instance_name)
            
            solver = get_solver(algorithm)
            solution = solver.solve(instance)
            
            experiment.log_solution("result", solution)
            experiment.log_parameter("objective", solution.objective)
            experiment.log_parameter("solve_time", solution.runtime)
```

### 3. Hyperparameter Optimization

```python
experiment = minto.Experiment("hyperparameter_tuning")
experiment.log_problem("scheduling", job_shop_problem)

from itertools import product

# Define parameter grid
param_grid = {
    "population_size": [50, 100, 200],
    "crossover_rate": [0.7, 0.8, 0.9],
    "mutation_rate": [0.01, 0.05, 0.1]
}

# Grid search
for params in product(*param_grid.values()):
    pop_size, crossover, mutation = params
    
    with experiment.run():
        experiment.log_parameter("population_size", pop_size)
        experiment.log_parameter("crossover_rate", crossover)
        experiment.log_parameter("mutation_rate", mutation)
        
        solution = genetic_algorithm(
            problem=job_shop_problem,
            population_size=pop_size,
            crossover_rate=crossover,
            mutation_rate=mutation
        )
        
        experiment.log_solution("optimized_schedule", solution)
        experiment.log_parameter("makespan", solution.makespan)
        experiment.log_parameter("tardiness", solution.total_tardiness)
```

## Integration with the Optimization Ecosystem

### OMMX Compatibility

MINTO is built around the Open Mathematical Modeling Exchange (OMMX) standard:

- **Native OMMX support**: Problems, instances, and solutions work seamlessly
- **Standardized format**: Ensures interoperability across tools
- **Archive compatibility**: Direct import/export with OMMX archives

### JijModeling Integration

MINTO provides first-class support for JijModeling problems:

```python
import jijmodeling as jm

# Define optimization problem
problem = jm.Problem("knapsack")
x = jm.BinaryVar("x", shape=(n,))
problem += jm.sum(i, values[i] * x[i])  # Objective
problem += jm.Constraint("capacity", jm.sum(i, weights[i] * x[i]) <= capacity)

# Automatic conversion and logging
experiment.log_problem("knapsack", problem)
```

### Solver Ecosystem

MINTO works with various optimization solvers:

- **Commercial solvers**: CPLEX, Gurobi (via OMMX adapters)
- **Open source solvers**: SCIP, OR-Tools (via OMMX adapters)
- **Metaheuristics**: OpenJij, custom implementations
- **Cloud solvers**: JijZept, D-Wave systems

## Conclusion

MINTO's mental model centers on systematic, reproducible optimization research. By understanding the two-level storage architecture, context manager pattern, and automatic metadata capture, users can leverage MINTO to:

- **Streamline research workflows**: Reduce boilerplate and focus on optimization problems
- **Ensure reproducibility**: Automatic environment capture and systematic data organization
- **Enable collaboration**: Standardized formats and sharing mechanisms
- **Accelerate discovery**: Structured analysis and visualization tools

The framework abstracts away experiment management complexity while maintaining flexibility for diverse optimization use cases. Whether conducting academic research, industrial optimization, or algorithm development, MINTO provides the foundation for rigorous, systematic experimentation.
