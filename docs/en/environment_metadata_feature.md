# MINTO Environment Metadata Auto-Collection Feature

## Overview

We have implemented an automatic environment metadata collection feature in the MINTO library. This feature significantly improves the reproducibility of benchmark experiments and makes it easier to compare experimental results across different environments.

## New Features

### 1. Automatic Environment Metadata Collection

By specifying `collect_environment=True` (default) when creating an experiment, the following information is automatically collected:

#### Information Collected
- **OS Information**: Operating system name and version
- **Hardware Information**: CPU information, core count, memory capacity, architecture
- **Python Environment**: Python version, execution path, virtual environment
- **Package Versions**: Versions of major optimization-related libraries
- **Execution Information**: Timestamp

### 2. Persistence Support

Environment metadata can be saved and loaded in the following formats:
- Directory format save/load
- OMMX archive format save/load

### 3. Convenient Methods

#### `get_environment_info()`
Get experiment environment metadata in dictionary format

#### `print_environment_summary()`
Display environment information summary in a readable format

## Usage Examples

### Basic Usage

```{code-cell} python
import minto

# Experiment with environment metadata collection enabled (default)
experiment = minto.Experiment(
    name="my_benchmark",
    collect_environment=True
)

# Run experiment
with experiment:
    experiment.log_parameter("algorithm", "my_algorithm")
    experiment.log_parameter("result", 42.0)

# Display environment information
experiment.print_environment_summary()

# Display experimental results
results = experiment.get_run_table()
print(results)

# Save experiment (environment information is automatically included)
experiment.save()
```



### Disabling Environment Metadata

```python
# Disable environment metadata collection
experiment = minto.Experiment(
    name="simple_experiment",
    collect_environment=False
)
```

### Loading Saved Experiments

```python
# Load from directory
loaded_exp = minto.Experiment.load_from_dir("path/to/experiment")

# Load from OMMX archive
loaded_exp = minto.Experiment.load_from_ommx_archive("experiment.ommx")

# Check environment information
env_info = loaded_exp.get_environment_info()
if env_info:
    print(f"Experiment OS: {env_info['os_name']}")
    print(f"Python Version: {env_info['python_version']}")
```

## Applications in Benchmark Experiments

The environment metadata feature provides the following benefits:

1. **Ensuring Reproducibility**: Detailed experiment environment information is automatically recorded
2. **Environment Comparison**: Proper comparison of experimental results across different machines
3. **Debugging Support**: Easy identification of issues caused by environment differences
4. **Research Reporting**: Automatic acquisition of environment information needed for papers and reports

## Implementation Details

### Architecture

- `minto/environment.py`: Core functionality for environment information collection
- `minto/experiment.py`: Integration with the Experiment class
- Auto-collection is executed only once during experiment creation (minimizing overhead)
- Error tolerance: Experiments continue even if environment information collection fails

### Dependencies

- `psutil`: Detailed hardware information acquisition (optional)
- Basic functionality works with standard library only

### Testing

Comprehensive tests implemented in `tests/test_environment_metadata.py`:
- Environment metadata collection tests
- Disable functionality tests
- Persistence (save/load) tests
- OMMX archive persistence tests
- Method operation tests

## Future Extensibility

1. GPU information collection
2. Network environment information
3. Docker/container environment detection
4. Custom environment information addition
5. Automatic environment difference analysis functionality

## Summary

With this implementation, MINTO provides an automatic environment metadata collection feature that significantly improves the reproducibility and traceability of optimization experiments. Researchers and developers no longer need to manually record experiment environment details, enabling more efficient execution of reliable benchmark experiments.

