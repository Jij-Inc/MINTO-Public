from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import minto.exceptions.exceptions as exceptions
from minto.exceptions.exceptions import OperationalError

__all__ = ["exceptions", "OperationalError"]
