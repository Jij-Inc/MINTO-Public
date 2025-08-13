# Logging API Reference

## Module Overview

The Minto library's logging functionality consists of the following modules:

- `minto.logging_config`: Log configuration and formatting functionality
- `minto.logger`: Main logging interface

## minto.logging_config

### LogLevel

An enumeration that defines log levels.

```python
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
```

**Usage example:**
```python
from minto.logging_config import LogLevel
config = LogConfig(level=LogLevel.DEBUG)
```

### LogFormat

An enumeration that defines log formats.

```python
class LogFormat(Enum):
    SIMPLE = "simple"
    DETAILED = "detailed"
    MINIMAL = "minimal"
    COMPACT = "compact"
```

**Usage example:**
```python
from minto.logging_config import LogFormat
config = LogConfig(format=LogFormat.DETAILED)
```

### LogConfig

A dataclass that manages log configuration.

```python
@dataclass
class LogConfig:
    enabled: bool = True
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.DETAILED
    show_timestamps: bool = True
    show_icons: bool = True
    show_colors: bool = True
    show_details: bool = True
    max_value_length: Optional[int] = 200
```

**Parameters:**
- `enabled`: Enable/disable logging functionality
- `level`: Log level to output
- `format`: Log display format
- `show_timestamps`: Display timestamps
- `show_icons`: Display icons
- `show_colors`: Display colors
- `show_details`: Display detailed information
- `max_value_length`: Maximum display length for values (None for no limit)

**Methods:**

#### should_log(level: LogLevel) -> bool
Determines whether logs at the specified level should be output.

```python
config = LogConfig(level=LogLevel.WARNING)
print(config.should_log(LogLevel.INFO))    # False
print(config.should_log(LogLevel.ERROR))   # True
```

### LogFormatter

A class that formats log messages.

```python
class LogFormatter:
    def __init__(self, config: LogConfig):
        self.config = config
        self._indent_level = 0
```

**Methods:**

#### format_message(level: LogLevel, message: str, **kwargs) -> str
Formats a log message.

**Parameters:**
- `level`: Log level
- `message`: Message content
- `**kwargs`: Additional formatting information

#### set_indent_level(level: int)
Sets the indent level.

#### increment_indent()
Increases the indent level by one.

#### decrement_indent()
Decreases the indent level by one.

#### truncate_value(value: Any, max_length: Optional[int] = None) -> str
Truncates a value to the specified length.

## minto.logger

### MintoLogger

The main class providing logging functionality.

```python
class MintoLogger:
    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self.formatter = LogFormatter(self.config)
```

**Experiment Lifecycle Methods:**

#### log_experiment_start(name: str)
Logs the start of an experiment.

```python
logger.log_experiment_start("my_experiment")
```

#### log_experiment_end(name: str, duration: float, num_runs: int)
Logs the end of an experiment.

**Parameters:**
- `name`: Experiment name
- `duration`: Execution time (seconds)
- `num_runs`: Number of runs executed

#### log_environment_info(env_info: Dict[str, Any])
Logs environment information.

**Run Lifecycle Methods:**

#### log_run_start(run_id: int)
Logs the start of a run.

#### log_run_end(run_id: int, duration: float)
Logs the end of a run.

**Data Logging Methods:**

#### log_parameter(key: str, value: Any)
Logs a parameter.

```python
logger.log_parameter("temperature", 1.0)
logger.log_parameter("solver_type", "OpenJij")
```

#### log_object(key: str, obj: Any, description: Optional[str] = None)
Logs an object.

```python
logger.log_object("problem_data", data_dict, "QUBO problem instance")
```

#### log_solution(key: str, solution: Any)
Logs a solution.

```python
logger.log_solution("best_solution", [1, 0, 1, 0, 1])
```

#### log_sampleset(key: str, num_samples: int, best_energy: Optional[float] = None)
Logs a sample set.

```python
logger.log_sampleset("results", 1000, -42.5)
```

#### log_solver(name: str, execution_time: Optional[float] = None)
Logs solver execution.

**Diagnostic Methods:**

#### log_debug(message: str)
Logs a DEBUG level message.

#### log_info(message: str)
Logs an INFO level message.

#### log_warning(message: str)
Logs a WARNING level message.

#### log_error(message: str)
Logs an ERROR level message.

#### log_critical(message: str)
Logs a CRITICAL level message.

### Global Functions

#### configure_logging(**kwargs)
Configures global logging settings.

```python
from minto.logger import configure_logging
from minto.logging_config import LogLevel

configure_logging(
    enabled=True,
    level=LogLevel.DEBUG,
    show_timestamps=True,
    show_colors=False
)
```

**Parameters:** Accepts the same parameters as the `LogConfig` class.

#### get_logger() -> MintoLogger
Gets the globally configured logger instance.

```python
from minto.logger import get_logger

logger = get_logger()
logger.log_experiment_start("global_experiment")
```

## Experiment Class Extensions

### New Parameters

#### verbose_logging: bool = False
Controls whether logging functionality is enabled/disabled.

#### log_config: Optional[LogConfig] = None
Specifies custom log configuration. Uses global configuration when not specified.

### New Methods

#### finish_experiment()
Ends the experiment and logs statistics.

```python
exp = Experiment(name="test", verbose_logging=True)
# ... run experiment ...
exp.finish_experiment()
```

## Run Class Extensions

### New Methods

#### log_solver(name: str, solver_func: Callable, exclude_params: Optional[List[str]] = None) -> Callable
Wraps a solver function to automatically log parameters and execution time.

```python
def my_solver(param1, param2, secret_key):
    return {"result": "success"}

with run:
    wrapped_solver = run.log_solver(
        "my_solver", 
        my_solver,
        exclude_params=["secret_key"]
    )
    result = wrapped_solver(param1=10, param2="test", secret_key="hidden")
```

**Parameters:**
- `name`: Solver name
- `solver_func`: Solver function to wrap
- `exclude_params`: List of parameter names to exclude from logging

**Return:** Wrapped solver function

## Usage Examples

### Basic Usage

```python
from minto import Experiment

exp = Experiment(name="api_example", verbose_logging=True)
run = exp.run()

with run:
    run.log_parameter("method", "QAOA")
    run.log_parameter("layers", 3)
    run.log_solution("result", [1, 0, 1])

exp.finish_experiment()
```

### Custom Configuration

```python
from minto import Experiment
from minto.logging_config import LogConfig, LogLevel, LogFormat

config = LogConfig(
    level=LogLevel.DEBUG,
    format=LogFormat.SIMPLE,
    show_timestamps=False
)

exp = Experiment(
    name="custom_example",
    verbose_logging=True,
    log_config=config
)
```

### Global Configuration

```python
from minto.logger import configure_logging, get_logger
from minto.logging_config import LogLevel

configure_logging(level=LogLevel.WARNING, show_colors=False)

logger = get_logger()
logger.log_experiment_start("global_example")
```

## Error Handling

The logging functionality is designed not to interrupt the main process when errors occur. If an error occurs during log output, it is handled internally and does not affect experiment execution.

## Performance

- When `verbose_logging=False`: No overhead
- When `verbose_logging=True`: Less than 1% impact on experiment processing time
- Memory usage: Minimal additional memory usage

The logging functionality is designed efficiently and is suitable for use in production environments.