# MINTO Frequently Asked Questions (FAQ)

## General Questions

### What is MINTO and how is it different from MLflow?

**MINTO** (Management and Insight Tool for Optimization) is the "MLflow for optimization" - a specialized experiment tracking framework designed specifically for mathematical optimization research and development.

**Key Differences from MLflow:**

| Feature | MLflow | MINTO |
|---------|--------|-------|
| **Primary Domain** | Machine Learning | Mathematical Optimization |
| **Data Types** | Models, metrics, artifacts | Problems, instances, solutions, samplesets |
| **Storage Format** | MLflow format | OMMX archives + directories |
| **Problem Structure** | Training/validation sets | Problem formulation + instances |
| **Reproducibility** | Model versioning | Environment + solver metadata |
| **Integration** | ML frameworks (sklearn, pytorch) | Optimization solvers (CPLEX, OR-Tools, JijZept) |

MINTO provides optimization-specific features like automatic problem/instance separation, solver parameter capture, and solution quality tracking that MLflow doesn't address.

### When should I use MINTO vs. other experiment tracking tools?

**Use MINTO when you're working on:**
- Mathematical optimization problems (TSP, VRP, scheduling, etc.)
- Algorithm development and benchmarking
- Hyperparameter tuning for optimization solvers
- Comparing different optimization approaches
- Research requiring reproducible optimization experiments

**Consider other tools for:**
- Pure machine learning workflows (use MLflow)
- Simple parameter logging (use Weights & Biases)
- Large-scale model training (use TensorBoard)

### Is MINTO suitable for both research and industry?

Yes! MINTO is designed for both contexts:

**Research Use Cases:**
- Academic paper experiments
- Algorithm development and analysis
- PhD/Masters thesis research
- Conference benchmark studies

**Industry Use Cases:**
- Production algorithm tuning
- A/B testing optimization strategies
- Operational research projects
- Supply chain optimization

## Installation and Setup

### How do I install MINTO?

```bash
# Basic installation
pip install minto

# With optional dependencies for specific solvers
pip install minto[cplex]  # For CPLEX integration
pip install minto[gurobi] # For Gurobi integration
pip install minto[all]    # All optional dependencies
```

### What are the system requirements?

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 1GB disk space

**Recommended:**
- Python 3.10+
- 8GB+ RAM (for large optimization problems)
- SSD storage (for better I/O performance)
- Multi-core CPU (for parallel experiments)

### Can I use MINTO without OMMX?

While MINTO is built around OMMX (Open Mathematical Modeling Exchange), you can use many features without deep OMMX knowledge:

```python
import minto

# Basic usage without explicit OMMX
experiment = minto.Experiment("simple_study")

with experiment.run():
    experiment.log_parameter("algorithm", "genetic")
    experiment.log_parameter("objective_value", 42.5)
    # MINTO handles OMMX conversion automatically
```

However, for advanced features like solver integration and standardized problem formats, OMMX knowledge is beneficial.

## Core Concepts

### What's the difference between experiment-level and run-level data?

**Experiment-level data** is shared across all runs within an experiment:
- **Problems**: Mathematical formulations
- **Instances**: Problem data (graphs, matrices, etc.)
- **Objects**: Shared configurations
- **Environment**: System metadata

**Run-level data** is specific to each optimization iteration:
- **Parameters**: Algorithm settings, hyperparameters
- **Solutions**: Optimization results
- **Metrics**: Performance measurements

```python
experiment = minto.Experiment("tsp_study")

# Experiment-level: shared across all runs
experiment.log_problem("tsp", traveling_salesman_problem)
experiment.log_instance("berlin52", berlin52_data)

# Run-level: unique per iteration
for temperature in [100, 500, 1000]:
    run = experiment.run()  # Create explicit run
    with run:  # Run context
        run.log_parameter("temperature", temperature)
        solution = solve(temperature)
        run.log_solution("result", solution)
```

### When should I use `log_parameter` vs `log_object`?

**Use `log_parameter` for:**
- Simple values: numbers, strings, small lists
- Algorithm hyperparameters
- Performance metrics
- Configuration flags

**Use `log_object` for:**
- Complex data structures
- Custom configurations
- Large datasets
- Non-serializable objects

```python
# Parameters: simple values
experiment.log_parameter("population_size", 100)
experiment.log_parameter("crossover_rate", 0.8)

# Objects: complex structures
experiment.log_object("algorithm_config", {
    "operators": ["ox", "pmx", "cx"],
    "selection": "tournament",
    "elitism": True
})
```

### How does automatic environment collection work?

MINTO automatically captures:

```python
# Automatic collection (enabled by default)
experiment = minto.Experiment("study", collect_environment=True)

# What's captured:
# - OS: macOS 14.1.1, Windows 11, Ubuntu 22.04
# - Hardware: CPU cores, memory, architecture
# - Python: version, virtual environment path
# - Packages: versions of optimization libraries
# - Execution: timestamp, working directory

# View captured info
experiment.print_environment_summary()
```

This ensures experiments are reproducible across different systems and environments.

## Data Management

### Where does MINTO store my experiment data?

**Default Storage:**
- Directory: `.minto_experiments/` in current working directory
- Structure: Organized by experiment name and data type

**Customizable Storage:**
```python
# Custom directory
experiment = minto.Experiment("study", savedir="/path/to/experiments")

# OMMX archive
experiment.save_as_ommx_archive("study.ommx")

# Cloud storage (via custom integration)
upload_to_cloud(experiment.save_as_ommx_archive())
```

### How do I share experiments with collaborators?

**Method 1: OMMX Archives**
```python
# Save as portable archive
artifact = experiment.save_as_ommx_archive("study.ommx")

# Share file via email, cloud storage, etc.
# Collaborator loads:
shared_experiment = minto.Experiment.load_from_ommx_archive("study.ommx")
```

**Method 2: Directory Sharing**
```python
# Save to shared directory
experiment = minto.Experiment("study", savedir="/shared/network/drive")

# Collaborator loads:
shared_experiment = minto.Experiment.load_from_dir("/shared/network/drive/study")
```

**Method 3: GitHub Integration**
```python
# Push to GitHub repository
experiment.push_github(
    org="optimization-lab",
    repo="experiments",
    name="tsp_study_v1"
)

# Collaborator pulls:
experiment = minto.Experiment.load_from_registry("optimization-lab/experiments:tsp_study_v1")
```

### Can I combine experiments from different sources?

Yes! MINTO provides powerful experiment combination:

```python
# Load experiments from different sources
exp1 = minto.Experiment.load_from_dir("genetic_algorithm_study")
exp2 = minto.Experiment.load_from_ommx_archive("simulated_annealing.ommx")
exp3 = minto.Experiment.load_from_registry("lab/repo:tabu_search")

# Combine for meta-analysis
combined = minto.Experiment.concat(
    [exp1, exp2, exp3],
    name="algorithm_comparison_meta_study"
)

# Analyze across all experiments
results = combined.get_run_table()
print(f"Total runs: {len(results)}")
```

### How do I handle large solution objects?

**Strategy 1: Store summaries**
```python
with experiment.run():
    large_solution = solve_large_problem()
    
    # Store key metrics instead of full solution
    experiment.log_parameter("objective", large_solution.objective)
    experiment.log_parameter("solution_size", len(large_solution.variables))
    experiment.log_parameter("runtime", large_solution.time)
    
    # Store full solution only if optimal
    if large_solution.is_optimal:
        experiment.log_solution("optimal_solution", large_solution)
```

**Strategy 2: External storage**
```python
with experiment.run():
    large_solution = solve_large_problem()
    
    # Save solution externally
    solution_path = f"/data/solutions/{experiment.name}_{experiment._run_id}.sol"
    large_solution.save(solution_path)
    
    # Log reference in MINTO
    experiment.log_parameter("solution_file", solution_path)
    experiment.log_parameter("objective", large_solution.objective)
```

## Solver Integration

### How do I integrate my custom solver with MINTO?

**Method 1: Decorator Approach**
```python
@experiment.log_solver
def my_custom_solver(problem, param1=10, param2=0.5):
    """Your solver implementation."""
    # MINTO automatically logs:
    # - Function name as solver name
    # - All parameters (param1, param2)
    # - Problem objects
    # - Return values (if solutions/samplesets)
    
    solution = your_optimization_logic(problem, param1, param2)
    return solution

# Usage automatically logs everything
result = my_custom_solver(tsp_problem, param1=20)
```

**Method 2: Manual Integration**
```python
def my_custom_solver(problem, param1=10, param2=0.5):
    """Your solver implementation."""
    
    with experiment.run():
        # Manual logging
        experiment.log_parameter("solver_name", "my_custom_solver")
        experiment.log_parameter("param1", param1)
        experiment.log_parameter("param2", param2)
        experiment.log_problem("input_problem", problem)
        
        solution = your_optimization_logic(problem, param1, param2)
        
        experiment.log_solution("solver_result", solution)
        experiment.log_parameter("objective", solution.objective)
        
    return solution
```

### Can MINTO work with commercial solvers like CPLEX or Gurobi?

Yes! MINTO integrates with commercial solvers through OMMX adapters:

**CPLEX Integration:**
```python
import ommx_cplex_adapter as cplex_ad

experiment = minto.Experiment("cplex_study")

@experiment.log_solver
def solve_with_cplex(instance, time_limit=300):
    adapter = cplex_ad.OMMXCPLEXAdapter(instance)
    adapter.set_time_limit(time_limit)
    
    solution = adapter.solve()
    return solution

# Usage
solution = solve_with_cplex(milp_instance, time_limit=600)
```

**Gurobi Integration:**
```python
import ommx_gurobi_adapter as gurobi_ad

@experiment.log_solver
def solve_with_gurobi(instance, mip_gap=0.01):
    adapter = gurobi_ad.OMMXGurobiAdapter(instance)
    adapter.set_mip_gap(mip_gap)
    
    solution = adapter.solve()
    return solution
```

### How do I handle solver failures or timeouts?

```python
@experiment.log_solver
def robust_solver(problem, time_limit=300):
    try:
        solution = your_solver(problem, time_limit)
        return solution
        
    except TimeoutError:
        # Log timeout information
        experiment.log_parameter("status", "timeout")
        experiment.log_parameter("time_limit_reached", True)
        return None
        
    except Exception as e:
        # Log error information
        experiment.log_parameter("status", "error")
        experiment.log_parameter("error_message", str(e))
        experiment.log_parameter("error_type", type(e).__name__)
        return None

# Usage with error handling
with experiment.run():
    result = robust_solver(difficult_problem)
    
    if result is not None:
        experiment.log_parameter("solve_successful", True)
        experiment.log_solution("result", result)
    else:
        experiment.log_parameter("solve_successful", False)
```

## Performance and Optimization

### How can I speed up experiment logging?

**1. Use Auto-saving Efficiently**
```python
# For long experiments, enable auto-saving
experiment = minto.Experiment("study", auto_saving=True)

# For fast experiments, batch operations
experiment = minto.Experiment("study", auto_saving=False)
# ... run many iterations ...
experiment.save()  # Single save at end
```

**2. Batch Parameter Logging**
```python
# Instead of multiple calls
experiment.log_parameter("param1", value1)
experiment.log_parameter("param2", value2)
experiment.log_parameter("param3", value3)

# Use batch logging
experiment.log_params({
    "param1": value1,
    "param2": value2,
    "param3": value3
})
```

**3. Selective Data Logging**
```python
# Log only essential data during runs
with experiment.run():
    experiment.log_parameter("key_metric", essential_value)
    # Skip logging large intermediate results
    
    if run_is_important:  # e.g., best result so far
        experiment.log_solution("detailed_solution", full_solution)
```

### Can I run multiple experiments in parallel?

Yes! MINTO supports parallel experiments:

**Approach 1: Separate Experiment Objects**
```python
import multiprocessing as mp

def run_parallel_experiment(experiment_id, parameters):
    # Each process creates its own experiment
    experiment = minto.Experiment(
        name=f"parallel_study_{experiment_id}",
        auto_saving=True
    )
    
    with experiment.run():
        for key, value in parameters.items():
            experiment.log_parameter(key, value)
        
        solution = solve_optimization_problem(parameters)
        experiment.log_solution("result", solution)
    
    return experiment.name

# Run experiments in parallel
if __name__ == "__main__":
    parameter_sets = [
        {"algorithm": "genetic", "pop_size": 50},
        {"algorithm": "genetic", "pop_size": 100},
        {"algorithm": "sa", "temperature": 1000},
        {"algorithm": "sa", "temperature": 2000}
    ]
    
    with mp.Pool(processes=4) as pool:
        experiment_names = pool.starmap(
            run_parallel_experiment,
            [(i, params) for i, params in enumerate(parameter_sets)]
        )
    
    # Combine results
    experiments = [
        minto.Experiment.load_from_dir(f".minto_experiments/{name}")
        for name in experiment_names
    ]
    
    combined = minto.Experiment.concat(experiments, name="parallel_combined")
```

**Approach 2: Using Dask (see Integration Guide)**

### How much disk space do experiments use?

**Typical Space Usage:**

| Component | Small Experiment | Medium Experiment | Large Experiment |
|-----------|------------------|-------------------|------------------|
| **Parameters** | < 1 MB | 1-10 MB | 10-100 MB |
| **Problems** | < 1 MB | 1-50 MB | 50-500 MB |
| **Instances** | 1-10 MB | 10-100 MB | 100MB-1GB |
| **Solutions** | 1-10 MB | 10-100 MB | 100MB-10GB |
| **Environment** | < 1 MB | < 1 MB | < 1 MB |
| **Total** | **< 20 MB** | **50-300 MB** | **1-50 GB** |

**Space-Saving Tips:**
```python
# 1. Store solution summaries
experiment.log_parameter("objective", solution.objective)
experiment.log_parameter("variables_count", len(solution.variables))

# 2. Use OMMX archives (compressed)
experiment.save_as_ommx_archive("compressed_study.ommx")

# 3. Clean up intermediate experiments
old_experiment_dir.rmdir()  # Remove after combining
```

## Troubleshooting

### My experiment data isn't being saved. What's wrong?

**Check 1: Auto-saving Setting**
```python
# Make sure auto_saving is enabled
experiment = minto.Experiment("study", auto_saving=True)

# OR manually save
experiment = minto.Experiment("study", auto_saving=False)
# ... do work ...
experiment.save()  # Don't forget this!
```

**Check 2: Directory Permissions**
```python
# Ensure write permissions to save directory
import pathlib
save_dir = pathlib.Path(".minto_experiments")
save_dir.mkdir(parents=True, exist_ok=True)

experiment = minto.Experiment("study", savedir=save_dir)
```

**Check 3: Disk Space**
```python
import shutil
free_space = shutil.disk_usage(".").free
print(f"Free disk space: {free_space / (1024**3):.1f} GB")
```

### I'm getting serialization errors with complex objects. How do I fix this?

**Problem:** Custom objects can't be JSON serialized
```python
class CustomOptimizer:
    def __init__(self, param):
        self.param = param

# This will fail:
experiment.log_parameter("optimizer", CustomOptimizer(42))  # SerializationError
```

**Solution 1: Store object properties**
```python
optimizer = CustomOptimizer(42)

# Store serializable properties
experiment.log_parameter("optimizer_type", "CustomOptimizer")
experiment.log_parameter("optimizer_param", optimizer.param)
```

**Solution 2: Use log_object for complex data**
```python
experiment.log_object("optimizer_config", {
    "class_name": "CustomOptimizer",
    "parameters": {"param": 42},
    "module": "my_optimizers"
})
```

**Solution 3: Custom serialization**
```python
def serialize_optimizer(optimizer):
    return {
        "type": type(optimizer).__name__,
        "state": optimizer.__dict__
    }

experiment.log_parameter("optimizer", serialize_optimizer(optimizer))
```

### How do I debug slow experiment loading?

**Diagnostic Steps:**

```python
import time

start_time = time.time()

# Time the loading process
experiment = minto.Experiment.load_from_dir("large_experiment")
load_time = time.time() - start_time

print(f"Loading took {load_time:.2f} seconds")

# Check experiment size
tables = experiment.get_experiment_tables()
for table_name, table in tables.items():
    print(f"{table_name}: {len(table)} rows, {table.memory_usage().sum() / 1024**2:.1f} MB")
```

**Common Solutions:**

1. **Large Solutions**: Store summaries instead of full solutions
2. **Many Runs**: Use `Experiment.concat()` to combine smaller experiments
3. **Storage Location**: Use SSD instead of HDD for better I/O performance

### Can I recover from a corrupted experiment?

**Recovery Strategies:**

**1. Partial Recovery (if auto_saving was enabled)**
```python
# Try loading individual runs
experiment_dir = pathlib.Path(".minto_experiments/corrupted_study")

# Check what data is available
print("Available data:")
for subdir in experiment_dir.iterdir():
    if subdir.is_dir():
        print(f"  {subdir.name}: {len(list(subdir.glob('*')))} files")

# Try loading partial data
try:
    experiment = minto.Experiment.load_from_dir(experiment_dir)
    print(f"Recovered {len(experiment.runs)} runs")
except Exception as e:
    print(f"Full recovery failed: {e}")
    # Implement custom recovery logic
```

**2. OMMX Archive Recovery**
```python
# OMMX archives are more robust
try:
    experiment = minto.Experiment.load_from_ommx_archive("backup.ommx")
    print("Successfully recovered from OMMX archive")
except Exception as e:
    print(f"Archive recovery failed: {e}")
```

**3. Prevention: Regular Backups**
```python
# Backup strategy
def backup_experiment(experiment, backup_dir="./backups"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{backup_dir}/{experiment.name}_{timestamp}.ommx"
    
    experiment.save_as_ommx_archive(backup_path)
    print(f"Backup saved: {backup_path}")

# Use regularly during long experiments
experiment = minto.Experiment("long_study")

for i in range(1000):
    with experiment.run():
        # ... optimization work ...
        pass
    
    # Backup every 100 runs
    if i % 100 == 0:
        backup_experiment(experiment)
```

## Advanced Usage

### How do I implement custom analysis workflows?

```python
class OptimizationAnalyzer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.results = experiment.get_run_table()
    
    def convergence_analysis(self):
        """Analyze convergence patterns."""
        import matplotlib.pyplot as plt
        
        # Group by algorithm
        for algorithm in self.results["parameter"]["algorithm"].unique():
            algo_data = self.results[
                self.results["parameter"]["algorithm"] == algorithm
            ]
            
            plt.plot(
                algo_data["run_id"],
                algo_data["parameter"]["objective"],
                label=algorithm,
                marker='o'
            )
        
        plt.xlabel("Run ID")
        plt.ylabel("Objective Value")
        plt.title("Convergence Comparison")
        plt.legend()
        plt.show()
    
    def statistical_analysis(self):
        """Perform statistical analysis."""
        import scipy.stats as stats
        
        # Compare algorithms statistically
        algorithms = self.results["parameter"]["algorithm"].unique()
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                data1 = self.results[
                    self.results["parameter"]["algorithm"] == algo1
                ]["parameter"]["objective"]
                
                data2 = self.results[
                    self.results["parameter"]["algorithm"] == algo2
                ]["parameter"]["objective"]
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                print(f"{algo1} vs {algo2}: p-value = {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  Significant difference detected!")
    
    def parameter_sensitivity(self, parameter_name):
        """Analyze parameter sensitivity."""
        import seaborn as sns
        
        if parameter_name in self.results["parameter"].columns:
            plt.figure(figsize=(10, 6))
            
            sns.scatterplot(
                data=self.results,
                x=("parameter", parameter_name),
                y=("parameter", "objective"),
                hue=("parameter", "algorithm")
            )
            
            plt.title(f"Sensitivity Analysis: {parameter_name}")
            plt.show()
        else:
            print(f"Parameter {parameter_name} not found in results")

# Usage
experiment = minto.Experiment.load_from_dir("comprehensive_study")
analyzer = OptimizationAnalyzer(experiment)

analyzer.convergence_analysis()
analyzer.statistical_analysis()
analyzer.parameter_sensitivity("population_size")
```

### Can I extend MINTO with custom plugins?

While MINTO doesn't have a formal plugin system, you can extend it:

```python
class MINTOExtension:
    def __init__(self, experiment):
        self.experiment = experiment
    
    def log_optimization_trace(self, trace_data):
        """Log detailed optimization trace."""
        trace_summary = {
            "total_iterations": len(trace_data),
            "best_objective": min(point["objective"] for point in trace_data),
            "convergence_rate": self._calculate_convergence_rate(trace_data)
        }
        
        self.experiment.log_object("optimization_trace", {
            "summary": trace_summary,
            "full_trace": trace_data
        })
    
    def log_algorithm_comparison(self, algorithm_results):
        """Log results from algorithm comparison."""
        comparison_data = {
            "algorithms_tested": list(algorithm_results.keys()),
            "best_algorithm": min(algorithm_results.items(), 
                                 key=lambda x: x[1]["objective"])[0],
            "performance_matrix": algorithm_results
        }
        
        self.experiment.log_object("algorithm_comparison", comparison_data)
    
    def _calculate_convergence_rate(self, trace_data):
        # Custom convergence calculation
        objectives = [point["objective"] for point in trace_data]
        improvements = [objectives[i-1] - objectives[i] 
                       for i in range(1, len(objectives)) 
                       if objectives[i-1] > objectives[i]]
        return sum(improvements) / len(improvements) if improvements else 0

# Usage
experiment = minto.Experiment("extended_study")
extension = MINTOExtension(experiment)

with experiment.run():
    # Standard MINTO logging
    experiment.log_parameter("algorithm", "genetic")
    
    # Extended logging
    extension.log_optimization_trace(detailed_trace)
    extension.log_algorithm_comparison(comparison_results)
```

This FAQ covers the most common questions and scenarios users encounter when working with MINTO. For additional help, consult the API documentation or reach out to the MINTO community.
