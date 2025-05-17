import sys
import types
import math


def _install_stubs():
    # Minimal numpy stub
    np = types.ModuleType("numpy")

    class ndarray(list):
        def __mul__(self, other):
            if isinstance(other, ndarray):
                return ndarray([a * b for a, b in zip(self, other)])
            return ndarray([a * other for a in self])

        __rmul__ = __mul__

        def __add__(self, other):
            if isinstance(other, ndarray):
                return ndarray([a + b for a, b in zip(self, other)])
            return ndarray([a + other for a in self])

        __radd__ = __add__

        def __sub__(self, other):
            if isinstance(other, ndarray):
                return ndarray([a - b for a, b in zip(self, other)])
            return ndarray([a - other for a in self])

        def __rsub__(self, other):
            if isinstance(other, ndarray):
                return ndarray([b - a for a, b in zip(self, other)])
            return ndarray([other - a for a in self])

    def array(seq):
        return ndarray(list(seq))

    def clip(arr, a_min, a_max):
        return ndarray([min(max(x, a_min), a_max) for x in arr])

    def log(arr):
        if isinstance(arr, ndarray):
            return ndarray([math.log(x) for x in arr])
        return math.log(arr)

    def mean(arr):
        return sum(arr) / len(arr)

    np.ndarray = ndarray
    np.array = array
    np.clip = clip
    np.log = log
    np.mean = mean
    np.zeros = lambda shape: ndarray([0] * shape) if isinstance(shape, int) else ndarray([[0] * shape[1] for _ in range(shape[0])])

    sys.modules["numpy"] = np

    # Minimal pandas stub
    pd = types.ModuleType("pandas")
    sys.modules["pandas"] = pd

    # Minimal sklearn stub
    sk = types.ModuleType("sklearn")
    sk.__version__ = "0.0"
    submods = {
        "model_selection": ["StratifiedKFold", "StratifiedShuffleSplit"],
        "cross_validation": ["StratifiedKFold", "StratifiedShuffleSplit"],
        "ensemble": ["RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier"],
        "linear_model": ["LogisticRegression"],
        "metrics": ["matthews_corrcoef"],
        "datasets": ["make_classification"],
    }

    for name, attrs in submods.items():
        mod = types.ModuleType(f"sklearn.{name}")
        for attr in attrs:
            setattr(mod, attr, type(attr, (), {}))
        setattr(sk, name, mod)
        sys.modules[f"sklearn.{name}"] = mod

    sys.modules["sklearn"] = sk


def test_logloss_small_example():
    _install_stubs()
    from protein_blender_demo import logloss
    import numpy as np

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = logloss(proba, y_true, epsilon=1.0e-15)
    assert abs(result - 0.3251) < 0.001
