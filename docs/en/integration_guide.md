# MINTO Integration and Migration Guide

## Migration from Other Experiment Tracking Tools

### From MLflow

If you're coming from MLflow for machine learning experiments, MINTO will feel familiar while being optimized for optimization problems.

#### Concept Mapping

| MLflow | MINTO | Notes |
|--------|-------|-------|
| `mlflow.start_run()` | `experiment.run()` | Explicit run creation |
| `mlflow.log_param()` | `run.log_parameter()` | Run-level parameters |
| `mlflow.log_metric()` | `run.log_parameter()` | Performance metrics |
| `mlflow.log_artifact()` | `run.log_solution()` | Complex objects |
| - | `experiment.log_problem()` | Optimization-specific |
| - | `experiment.log_instance()` | Optimization-specific |
| Tracking Server | Directory/OMMX | Local or archive storage |

#### Code Migration Example

**MLflow Style:**
```python
import mlflow

mlflow.set_experiment("optimization_study")

with mlflow.start_run():
    mlflow.log_param("algorithm", "genetic")
    mlflow.log_param("population_size", 100)
    
    solution = optimize()
    
    mlflow.log_metric("objective", solution.objective)
    mlflow.log_metric("runtime", solution.time)
    mlflow.log_artifact("solution.json")
```

**MINTO Style:**
```python
import minto

experiment = minto.Experiment("optimization_study")

run = experiment.run()
with run:
    run.log_parameter("algorithm", "genetic")
    run.log_parameter("population_size", 100)
    
    solution = optimize()
    
    run.log_parameter("objective", solution.objective)
    run.log_parameter("runtime", solution.time)
    run.log_solution("optimal_solution", solution)
```

### From Manual Experiment Tracking

#### Before: Manual CSV/Excel Tracking

```python
# Old approach: Manual tracking
import pandas as pd
import csv

results = []

for temperature in [100, 500, 1000]:
    for alpha in [0.1, 0.5, 1.0]:
        solution = simulated_annealing(problem, temperature, alpha)
        
        # Manual result collection
        results.append({
            "temperature": temperature,
            "alpha": alpha,
            "objective": solution.objective,
            "runtime": solution.time,
            "feasible": solution.is_feasible
        })

# Manual saving
df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)
```

#### After: MINTO Approach

```python
# MINTO approach: Automated tracking
import minto

experiment = minto.Experiment("sa_parameter_study")
experiment.log_problem("tsp", tsp_problem)

for temperature in [100, 500, 1000]:
    for alpha in [0.1, 0.5, 1.0]:
        run = experiment.run()
        with run:
            run.log_parameter("temperature", temperature)
            run.log_parameter("alpha", alpha)
            
            solution = simulated_annealing(problem, temperature, alpha)
            
            run.log_solution("sa_solution", solution)
            run.log_parameter("objective", solution.objective)
            run.log_parameter("runtime", solution.time)
            run.log_parameter("feasible", solution.is_feasible)

# Automatic analysis
results = experiment.get_run_table()
print(results.head())

# Automatic saving (if auto_saving=True)
experiment.save()
```

## Integrating MINTO into Existing Workflows

### Research Laboratory Integration

#### Setting Up Shared Experiments

```python
# Shared base directory for lab experiments
import pathlib

LAB_EXPERIMENTS_DIR = pathlib.Path("/shared/optimization_lab/experiments")

def create_lab_experiment(researcher_name, study_name):
    """Standard experiment creation for lab members."""
    experiment_name = f"{researcher_name}_{study_name}_{datetime.now().strftime('%Y%m%d')}"
    
    experiment = minto.Experiment(
        name=experiment_name,
        savedir=LAB_EXPERIMENTS_DIR,
        auto_saving=True,
        collect_environment=True
    )
    
    # Log researcher information
    experiment.log_parameter("researcher", researcher_name)
    experiment.log_parameter("institution", "Optimization Research Lab")
    
    return experiment

# Usage by lab members
experiment = create_lab_experiment("alice", "genetic_algorithm_study")
```

#### Standardized Problem Library

```python
# Laboratory problem registry
class LabProblemRegistry:
    def __init__(self, base_dir):
        self.base_dir = pathlib.Path(base_dir)
        self.problems = {}
        self.instances = {}
        
    def register_problem(self, name, problem, description=""):
        """Register a standard problem for lab use."""
        self.problems[name] = {
            "problem": problem,
            "description": description,
            "added_by": "lab_admin",
            "added_date": datetime.now()
        }
        
        # Save to shared location
        problem_path = self.base_dir / "problems" / f"{name}.pkl"
        problem_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(problem_path, "wb") as f:
            pickle.dump(problem, f)
    
    def get_problem(self, name):
        """Get a standard problem."""
        if name in self.problems:
            return self.problems[name]["problem"]
        
        # Load from disk if not in memory
        problem_path = self.base_dir / "problems" / f"{name}.pkl"
        if problem_path.exists():
            with open(problem_path, "rb") as f:
                return pickle.load(f)
        
        raise ValueError(f"Problem {name} not found")

# Lab usage
registry = LabProblemRegistry("/shared/optimization_lab/registry")

# Standard problems available to all researchers
experiment = create_lab_experiment("bob", "metaheuristics_comparison")
experiment.log_problem("tsp_50", registry.get_problem("tsp_50_cities"))
experiment.log_problem("vrp_100", registry.get_problem("vrp_100_customers"))
```

### CI/CD Integration

#### Automated Benchmarking Pipeline

```python
# benchmark_pipeline.py
import minto
import argparse
import json
from pathlib import Path

def run_benchmark_suite():
    """Automated benchmark execution for CI/CD."""
    
    # Create timestamp-based experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = minto.Experiment(
        name=f"benchmark_suite_{timestamp}",
        savedir=Path("./benchmark_results"),
        auto_saving=True
    )
    
    # Load standard benchmark problems
    benchmark_problems = load_benchmark_suite()
    
    for problem_name, problem_data in benchmark_problems.items():
        experiment.log_problem(problem_name, problem_data["problem"])
        experiment.log_instance(f"{problem_name}_instance", problem_data["instance"])
        
        # Run all algorithms in the suite
        algorithms = ["genetic", "simulated_annealing", "tabu_search"]
        
        for algorithm in algorithms:
            with experiment.run():
                experiment.log_parameter("problem", problem_name)
                experiment.log_parameter("algorithm", algorithm)
                experiment.log_parameter("ci_commit", get_git_commit_hash())
                
                solver = get_algorithm_solver(algorithm)
                solution = solver.solve(problem_data["instance"])
                
                experiment.log_solution(f"{algorithm}_solution", solution)
                experiment.log_parameter("objective", solution.objective)
                experiment.log_parameter("runtime", solution.runtime)
                experiment.log_parameter("feasible", solution.is_feasible)
    
    # Generate performance report
    results = experiment.get_run_table()
    
    # Compare with baseline (previous runs)
    baseline_results = load_baseline_results()
    performance_comparison = compare_with_baseline(results, baseline_results)
    
    # Save comparison report
    with open("benchmark_report.json", "w") as f:
        json.dump(performance_comparison, f, indent=2)
    
    # Return success/failure for CI
    return performance_comparison["overall_status"] == "PASS"

if __name__ == "__main__":
    success = run_benchmark_suite()
    exit(0 if success else 1)
```

#### GitHub Actions Integration

```yaml
# .github/workflows/optimization_benchmark.yml
name: Optimization Algorithm Benchmarks

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 0'  # Weekly benchmarks

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install minto
        pip install -r requirements.txt
    
    - name: Run benchmark suite
      run: |
        python benchmark_pipeline.py
        
    - name: Upload benchmark results
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: |
          benchmark_results/
          benchmark_report.json
    
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      run: |
        python scripts/post_benchmark_comment.py \
          --results benchmark_report.json \
          --pr-number ${{ github.event.number }}
```

### Docker Integration

#### Containerized Experiments

```dockerfile
# Dockerfile.optimization
FROM python:3.11-slim

# Install optimization solvers
RUN apt-get update && apt-get install -y \
    build-essential \
    coinor-cbc \
    coinor-clp \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MINTO
RUN pip install minto

# Copy experiment code
COPY experiments/ /experiments/
WORKDIR /experiments

# Default command
CMD ["python", "run_experiment.py"]
```

```python
# run_experiment.py (containerized experiment)
import minto
import os
import sys

def main():
    # Get configuration from environment variables
    experiment_name = os.getenv("EXPERIMENT_NAME", "containerized_study")
    problem_type = os.getenv("PROBLEM_TYPE", "tsp")
    algorithm = os.getenv("ALGORITHM", "genetic")
    
    # Create experiment with container metadata
    experiment = minto.Experiment(
        name=f"{experiment_name}_{algorithm}",
        savedir="/results",  # Mount point for results
        auto_saving=True
    )
    
    # Log container information
    experiment.log_parameter("container_id", os.getenv("HOSTNAME", "unknown"))
    experiment.log_parameter("docker_image", os.getenv("DOCKER_IMAGE", "unknown"))
    
    # Load problem based on type
    problem, instance = load_problem(problem_type)
    experiment.log_problem(problem_type, problem)
    experiment.log_instance(f"{problem_type}_instance", instance)
    
    # Run optimization
    with experiment.run():
        experiment.log_parameter("algorithm", algorithm)
        
        solver = get_solver(algorithm)
        solution = solver.solve(instance)
        
        experiment.log_solution("result", solution)
        experiment.log_parameter("objective", solution.objective)
        experiment.log_parameter("runtime", solution.runtime)
    
    # Save results
    experiment.save()
    print(f"Experiment {experiment.name} completed successfully")

if __name__ == "__main__":
    main()
```

```bash
# Run containerized experiment
docker run -v $(pwd)/results:/results \
           -e EXPERIMENT_NAME="docker_optimization" \
           -e PROBLEM_TYPE="vrp" \
           -e ALGORITHM="genetic" \
           optimization:latest
```

### Cloud Integration

#### AWS S3 Storage

```python
import boto3
import minto
from pathlib import Path

class S3ExperimentManager:
    def __init__(self, bucket_name, region="us-east-1"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3", region_name=region)
    
    def save_experiment_to_s3(self, experiment, key_prefix="experiments/"):
        """Save MINTO experiment to S3."""
        
        # Save as OMMX archive locally first
        local_path = f"/tmp/{experiment.name}.ommx"
        artifact = experiment.save_as_ommx_archive(local_path)
        
        # Upload to S3
        s3_key = f"{key_prefix}{experiment.name}.ommx"
        self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
        
        # Clean up local file
        Path(local_path).unlink()
        
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def load_experiment_from_s3(self, experiment_name, key_prefix="experiments/"):
        """Load MINTO experiment from S3."""
        
        # Download from S3
        s3_key = f"{key_prefix}{experiment_name}.ommx"
        local_path = f"/tmp/{experiment_name}.ommx"
        
        self.s3_client.download_file(self.bucket_name, s3_key, local_path)
        
        # Load experiment
        experiment = minto.Experiment.load_from_ommx_archive(local_path)
        
        # Clean up local file
        Path(local_path).unlink()
        
        return experiment

# Usage
s3_manager = S3ExperimentManager("my-optimization-experiments")

# Save experiment to cloud
experiment = minto.Experiment("cloud_optimization_study")
# ... run experiment ...
s3_url = s3_manager.save_experiment_to_s3(experiment)
print(f"Experiment saved to: {s3_url}")

# Load experiment from cloud
loaded_experiment = s3_manager.load_experiment_from_s3("cloud_optimization_study")
results = loaded_experiment.get_run_table()
```

#### Azure Blob Storage

```python
from azure.storage.blob import BlobServiceClient
import minto

class AzureBlobExperimentManager:
    def __init__(self, connection_string, container_name):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
    
    def save_experiment_to_blob(self, experiment, blob_prefix="experiments/"):
        """Save MINTO experiment to Azure Blob Storage."""
        
        # Save as OMMX archive
        local_path = f"/tmp/{experiment.name}.ommx"
        experiment.save_as_ommx_archive(local_path)
        
        # Upload to blob storage
        blob_name = f"{blob_prefix}{experiment.name}.ommx"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # Cleanup
        Path(local_path).unlink()
        
        return f"https://{blob_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
    
    def load_experiment_from_blob(self, experiment_name, blob_prefix="experiments/"):
        """Load MINTO experiment from Azure Blob Storage."""
        
        blob_name = f"{blob_prefix}{experiment_name}.ommx"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        
        # Download to local temp file
        local_path = f"/tmp/{experiment_name}.ommx"
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        # Load experiment
        experiment = minto.Experiment.load_from_ommx_archive(local_path)
        
        # Cleanup
        Path(local_path).unlink()
        
        return experiment
```

## Advanced Integration Patterns

### Distributed Computing Integration

#### Dask Integration

```python
import minto
import dask
from dask.distributed import Client, as_completed
from dask import delayed

def parallel_optimization_study():
    """Run parallel optimization experiments using Dask."""
    
    # Connect to Dask cluster
    client = Client("scheduler-address:8786")
    
    # Create base experiment
    base_experiment = minto.Experiment("parallel_optimization_study")
    
    # Define parameter space
    parameter_combinations = [
        {"algorithm": "genetic", "pop_size": 50, "generations": 100},
        {"algorithm": "genetic", "pop_size": 100, "generations": 100},
        {"algorithm": "sa", "temperature": 1000, "cooling": 0.95},
        {"algorithm": "sa", "temperature": 2000, "cooling": 0.99},
        # ... more combinations
    ]
    
    @delayed
    def run_single_experiment(params):
        """Run a single optimization experiment."""
        
        # Create worker-specific experiment
        worker_experiment = minto.Experiment(
            name=f"worker_{params['algorithm']}_{hash(str(params))}",
            savedir=f"/shared/experiments/worker_results",
            auto_saving=True
        )
        
        with worker_experiment.run():
            # Log parameters
            for key, value in params.items():
                worker_experiment.log_parameter(key, value)
            
            # Run optimization
            solution = optimize_with_params(params)
            
            # Log results
            worker_experiment.log_solution("result", solution)
            worker_experiment.log_parameter("objective", solution.objective)
            worker_experiment.log_parameter("runtime", solution.runtime)
        
        return worker_experiment.name
    
    # Submit all experiments
    futures = [run_single_experiment(params) for params in parameter_combinations]
    
    # Collect results as they complete
    completed_experiments = []
    for future in as_completed(futures):
        experiment_name = future.result()
        completed_experiments.append(experiment_name)
        print(f"Completed: {experiment_name}")
    
    # Combine all worker experiments
    worker_experiments = [
        minto.Experiment.load_from_dir(f"/shared/experiments/worker_results/{name}")
        for name in completed_experiments
    ]
    
    combined_experiment = minto.Experiment.concat(
        worker_experiments,
        name="parallel_optimization_combined"
    )
    
    return combined_experiment

# Usage
combined_results = parallel_optimization_study()
print(combined_results.get_run_table())
```

#### Ray Integration

```python
import ray
import minto

@ray.remote
class OptimizationWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.experiment = minto.Experiment(
            name=f"ray_worker_{worker_id}",
            auto_saving=True
        )
    
    def run_optimization(self, problem, algorithm_params):
        """Run optimization on this worker."""
        
        with self.experiment.run():
            # Log worker and parameters
            self.experiment.log_parameter("worker_id", self.worker_id)
            
            for key, value in algorithm_params.items():
                self.experiment.log_parameter(key, value)
            
            # Run optimization
            solution = solve_problem(problem, algorithm_params)
            
            # Log results
            self.experiment.log_solution("worker_solution", solution)
            self.experiment.log_parameter("objective", solution.objective)
        
        return {
            "worker_id": self.worker_id,
            "objective": solution.objective,
            "experiment_name": self.experiment.name
        }

def distributed_hyperparameter_search():
    """Distributed hyperparameter search using Ray."""
    
    # Initialize Ray
    ray.init()
    
    # Create worker pool
    num_workers = 4
    workers = [OptimizationWorker.remote(i) for i in range(num_workers)]
    
    # Define search space
    search_space = generate_parameter_combinations()
    
    # Distribute work
    futures = []
    for i, params in enumerate(search_space):
        worker = workers[i % num_workers]
        future = worker.run_optimization.remote(tsp_problem, params)
        futures.append(future)
    
    # Collect results
    results = ray.get(futures)
    
    # Find best result
    best_result = min(results, key=lambda x: x["objective"])
    
    # Load best experiment for detailed analysis
    best_experiment = minto.Experiment.load_from_dir(
        f"./ray_worker_{best_result['worker_id']}"
    )
    
    ray.shutdown()
    
    return best_result, best_experiment
```

### Database Integration

#### PostgreSQL Integration

```python
import psycopg2
import pandas as pd
import minto
from sqlalchemy import create_engine

class PostgreSQLExperimentTracker:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        self.setup_tables()
    
    def setup_tables(self):
        """Create tables for experiment tracking."""
        
        create_experiments_table = """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            environment_info JSONB,
            description TEXT
        );
        """
        
        runs_table = """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id SERIAL PRIMARY KEY,
            experiment_id INTEGER REFERENCES experiments(experiment_id),
            run_index INTEGER NOT NULL,
            parameters JSONB NOT NULL,
            results JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self.engine.connect() as conn:
            conn.execute(create_experiments_table)
            conn.execute(runs_table)
            conn.commit()
    
    def sync_minto_experiment(self, experiment):
        """Sync MINTO experiment to PostgreSQL."""
        
        # Insert or update experiment record
        experiment_data = {
            "name": experiment.name,
            "environment_info": experiment.get_environment_info(),
            "description": f"MINTO experiment with {len(experiment.runs)} runs"
        }
        
        # Insert experiment
        insert_experiment_sql = """
        INSERT INTO experiments (name, environment_info, description)
        VALUES (%(name)s, %(environment_info)s, %(description)s)
        ON CONFLICT (name) DO UPDATE SET
            environment_info = EXCLUDED.environment_info,
            description = EXCLUDED.description
        RETURNING experiment_id;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(insert_experiment_sql, experiment_data)
            experiment_id = result.fetchone()[0]
        
        # Insert runs
        runs_table = experiment.get_run_table()
        
        for idx, run in runs_table.iterrows():
            run_data = {
                "experiment_id": experiment_id,
                "run_index": run["run_id"],
                "parameters": run["parameter"].to_dict(),
                "results": {
                    col: run[col].to_dict() if hasattr(run[col], "to_dict") else str(run[col])
                    for col in runs_table.columns if col != "parameter"
                }
            }
            
            insert_run_sql = """
            INSERT INTO experiment_runs (experiment_id, run_index, parameters, results)
            VALUES (%(experiment_id)s, %(run_index)s, %(parameters)s, %(results)s)
            ON CONFLICT DO NOTHING;
            """
            
            with self.engine.connect() as conn:
                conn.execute(insert_run_sql, run_data)
                conn.commit()
    
    def query_experiments(self, algorithm=None, min_objective=None):
        """Query experiments from database."""
        
        query = """
        SELECT e.name, e.created_at, COUNT(r.run_id) as num_runs,
               AVG((r.results->>'objective')::float) as avg_objective
        FROM experiments e
        LEFT JOIN experiment_runs r ON e.experiment_id = r.experiment_id
        WHERE 1=1
        """
        
        params = {}
        
        if algorithm:
            query += " AND r.parameters->>'algorithm' = %(algorithm)s"
            params["algorithm"] = algorithm
            
        if min_objective:
            query += " AND (r.results->>'objective')::float >= %(min_objective)s"
            params["min_objective"] = min_objective
        
        query += " GROUP BY e.experiment_id, e.name, e.created_at ORDER BY e.created_at DESC"
        
        return pd.read_sql(query, self.engine, params=params)

# Usage
db_tracker = PostgreSQLExperimentTracker("postgresql://user:pass@localhost/experiments")

# Sync MINTO experiment to database
experiment = minto.Experiment("database_tracked_study")
# ... run experiments ...
db_tracker.sync_minto_experiment(experiment)

# Query experiments
genetic_experiments = db_tracker.query_experiments(algorithm="genetic")
print(genetic_experiments)
```

This comprehensive integration guide provides practical patterns for incorporating MINTO into existing research and development workflows, from individual research projects to large-scale distributed computing environments.
