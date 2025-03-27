import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """numpy encoder for json

    Example:
        >>> import json
        >>> import numpy as np
        >>> obj = {'a': np.array([1, 2, 3])}
        >>> json.dumps(obj, cls=NumpyEncoder)
        '{"a": [1, 2, 3]}'
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
