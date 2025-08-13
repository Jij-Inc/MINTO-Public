"""Environment information collection module for MINTO experiments.

This module provides functionality to automatically collect and record
environment metadata for optimization experiments, including OS information,
hardware specifications, Python environment details, and package versions.
"""

import datetime
import os
import platform
import subprocess
import sys
import typing as typ
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class EnvironmentInfo:
    """Container for environment information collected during experiment execution."""

    # OS Information
    os_name: str
    os_version: str
    platform_info: str

    # Hardware Information
    cpu_info: str
    cpu_count: int
    memory_total: int  # in bytes
    architecture: str

    # Python Environment
    python_version: str
    python_executable: str
    virtual_env: typ.Optional[str]

    # Package Information
    package_versions: dict[str, str]

    # Execution Information
    timestamp: str

    def to_dict(self) -> dict[str, typ.Any]:
        """Convert EnvironmentInfo to dictionary for serialization."""
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "platform_info": self.platform_info,
            "cpu_info": self.cpu_info,
            "cpu_count": self.cpu_count,
            "memory_total": self.memory_total,
            "architecture": self.architecture,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "virtual_env": self.virtual_env,
            "package_versions": self.package_versions,
            "timestamp": self.timestamp,
        }


def get_os_info() -> tuple[str, str, str]:
    """Get operating system information.

    Returns:
        Tuple of (os_name, os_version, platform_info)
    """
    os_name = platform.system()
    os_version = platform.release()
    platform_info = platform.platform()

    return os_name, os_version, platform_info


def get_cpu_info() -> tuple[str, int, str]:
    """Get CPU information.

    Returns:
        Tuple of (cpu_info, cpu_count, architecture)
    """
    try:
        # Try to get detailed CPU info on different platforms
        if platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                cpu_info = (
                    result.stdout.strip()
                    if result.returncode == 0
                    else platform.processor()
                )
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                cpu_info = platform.processor()
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "model name" in line:
                            cpu_info = line.split(":")[1].strip()
                            break
                    else:
                        cpu_info = platform.processor()
            except (FileNotFoundError, IOError):
                cpu_info = platform.processor()
        else:
            cpu_info = platform.processor()
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        cpu_info = platform.processor()

    # Get CPU count
    if _HAS_PSUTIL:
        cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    else:
        cpu_count = os.cpu_count() or 1

    architecture = platform.machine()

    return cpu_info, cpu_count, architecture


def get_memory_info() -> int:
    """Get total system memory in bytes.

    Returns:
        Total memory in bytes
    """
    if _HAS_PSUTIL:
        try:
            return psutil.virtual_memory().total
        except (AttributeError, OSError):
            pass

    # Fallback methods for different platforms
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal is in kB, convert to bytes
                        mem_kb = int(line.split()[1])
                        return mem_kb * 1024
    except (
        subprocess.TimeoutExpired,
        subprocess.SubprocessError,
        FileNotFoundError,
        ValueError,
    ):
        pass

    return 0


def get_python_env_info() -> tuple[str, str, typ.Optional[str]]:
    """Get Python environment information.

    Returns:
        Tuple of (python_version, python_executable, virtual_env)
    """
    python_version = sys.version
    python_executable = sys.executable

    # Detect virtual environment
    virtual_env = None
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        # Check for various virtual environment indicators
        virtual_env = os.environ.get("VIRTUAL_ENV")
        if not virtual_env:
            virtual_env = os.environ.get("CONDA_DEFAULT_ENV")
        if not virtual_env:
            # Try to detect from path
            exe_path = Path(sys.executable)
            if ".venv" in exe_path.parts or "venv" in exe_path.parts:
                virtual_env = str(exe_path.parent.parent)
            elif "site-packages" in str(exe_path):
                virtual_env = "detected"

    return python_version, python_executable, virtual_env


# Default packages to check for version information (module-level constant)
# Use an immutable tuple to prevent accidental mutation.
DEFAULT_PACKAGES: tuple[str, ...] = (
    "minto",
    "jijmodeling",
    "ommx",
    "numpy",
    "pandas",
    "scipy",
    "openjij",
    "pyscipopt",
    "matplotlib",
    "seaborn",
)


def _get_single_package_version(package: str) -> str:
    """Get version of a single package.

    Args:
        package: Package name to check

    Returns:
        Version string or error/status indicator
    """
    try:
        if package == "minto":
            # Try to get minto version
            try:
                import minto

                return getattr(minto, "__version__", "unknown")
            except ImportError:
                return "not_installed"
        else:
            # Use importlib.metadata for other packages
            try:
                import importlib.metadata

                return importlib.metadata.version(package)
            except (importlib.metadata.PackageNotFoundError, ImportError):
                try:
                    # Fallback to __version__ attribute
                    module = __import__(package)
                    return getattr(module, "__version__", "unknown")
                except ImportError:
                    return "not_installed"
    except (ImportError, AttributeError, ValueError):
        return "error"


def get_package_versions(
    key_packages: typ.Optional[typ.Iterable[str]] = None,
) -> dict[str, str]:
    """Get versions of key packages used in the experiment.

    Args:
    key_packages: Iterable of package names to check. If None, uses default list.

    Returns:
        Dictionary mapping package names to their versions
    """
    if key_packages is None:
        key_packages = DEFAULT_PACKAGES

    package_versions = {}

    for package in key_packages:
        package_versions[package] = _get_single_package_version(package)

    return package_versions


def collect_environment_info(
    include_packages: bool = True, additional_packages: typ.Optional[list[str]] = None
) -> EnvironmentInfo:
    """Collect comprehensive environment information.

    Args:
        include_packages: Whether to include package version information
        additional_packages: Additional packages to check versions for

    Returns:
        EnvironmentInfo object containing all collected information
    """
    # Collect OS information
    os_name, os_version, platform_info = get_os_info()

    # Collect hardware information
    cpu_info, cpu_count, architecture = get_cpu_info()
    memory_total = get_memory_info()

    # Collect Python environment information
    python_version, python_executable, virtual_env = get_python_env_info()

    # Collect package information
    package_versions = {}
    if include_packages:
        # Copy to a list before extending to avoid mutating the module-level constant
        default_packages_list: list[str] = list(DEFAULT_PACKAGES)
        if additional_packages:
            default_packages_list.extend(additional_packages)
        package_versions = get_package_versions(default_packages_list)

    # Generate timestamp
    timestamp = datetime.datetime.now().isoformat()

    return EnvironmentInfo(
        os_name=os_name,
        os_version=os_version,
        platform_info=platform_info,
        cpu_info=cpu_info,
        cpu_count=cpu_count,
        memory_total=memory_total,
        architecture=architecture,
        python_version=python_version,
        python_executable=python_executable,
        virtual_env=virtual_env,
        package_versions=package_versions,
        timestamp=timestamp,
    )


def format_memory_size(bytes_size: int) -> str:
    """Format memory size in human-readable format.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "8.0 GB")
    """
    size_float = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024.0:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.1f} PB"


def get_environment_summary(env_info: EnvironmentInfo) -> str:
    """Get a human-readable summary of environment information.

    Args:
        env_info: EnvironmentInfo object

    Returns:
        Formatted summary string
    """
    memory_str = format_memory_size(env_info.memory_total)

    summary = f"""Environment Summary:
OS: {env_info.os_name} {env_info.os_version}
CPU: {env_info.cpu_info} ({env_info.cpu_count} cores)
Memory: {memory_str}
Architecture: {env_info.architecture}
Python: {env_info.python_version.split()[0]}
Virtual Environment: {env_info.virtual_env or "None"}
Timestamp: {env_info.timestamp}
"""

    if env_info.package_versions:
        summary += "\nKey Package Versions:\n"
        for pkg, version in env_info.package_versions.items():
            summary += f"  {pkg}: {version}\n"

    return summary
