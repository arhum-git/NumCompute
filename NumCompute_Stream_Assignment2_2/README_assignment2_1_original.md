# NumCompute

NumCompute is a modular scientific-computing toolkit written with Python and NumPy. It provides reusable operations for data ingestion, preprocessing, sorting and searching, ranking, descriptive statistics, classification and regression metrics, finite-difference differentiation, pipeline composition, and benchmarking.

## Requirements

- Python 3.10 or later
- NumPy for toolkit operations
- pytest to run the test suite (development only)
- Jupyter to open the demonstration notebook (optional)

## Installation

```bash
git clone https://github.com/arhum-git/NumCompute.git
cd NumCompute
python -m pip install -e .
```

Install optional test and notebook tools separately if needed:

```bash
python -m pip install pytest notebook
```

## Layout

```text
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
├── demo/
│   ├── dataset/test_student_performance_data.csv
│   └── quickstart.ipynb
├── README.md
└── pyproject.toml
```

## Quick examples

```python
import numpy as np
from numcompute.preprocessing import StandardScaler
from numcompute.sort_search import topk, binary_search
from numcompute.optim import grad, jacobian
from numcompute.utils import softmax, logsumexp

X = np.array([[1., 2.], [3., 4.], [5., 6.]])
X_scaled = StandardScaler().fit_transform(X)
values, indices = topk(np.array([3, 8, 8, 1]), 2)
position, exists = binary_search(np.array([1, 3, 8]), 3)
g = grad(lambda x: x[0]**2 + x[1]**2, np.array([2., 3.]))
J = jacobian(lambda x: np.array([x[0] + x[1], x[0]*x[1]]), np.array([2., 3.]))
probs = softmax(np.array([1., 2., 3.]))
log_total = logsumexp(np.array([1., 2., 3.]))
```

## API overview

| Module | Core functions / classes |
|---|---|
| `io` | NumPy CSV loading and missing-value processing |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer` |
| `sort_search` | `stable_sort`, `multikey_sort`, `topk`, `quickselect`, `binary_search` |
| `rank` | `rank`, `percentile` |
| `stats` | Mean, median, std/variance, histogram, quantiles, streaming stats |
| `metrics` | Accuracy, precision, recall, F1, confusion matrix, MSE, ROC and AUC |
| `optim` | `grad`, `jacobian` |
| `pipeline` | `Pipeline`, `FeatureUnion` |
| `utils` | Stable activations, distances, batching |
| `benchmarking` | `timeit`, `compare`, `print_table`, `run_all` |

`topk` resolves equal values by lowest original index. `softmax` raises `ValueError` for an all-`-inf` reduction slice and splits probability evenly among `+inf` entries. Nonfinite or nonpositive finite-difference step sizes raise `ValueError`.

## Run tests

From the project root:

```bash
python -m pytest -q
```

## Run the demo

```bash
jupyter notebook demo/quickstart.ipynb
```

The notebook reads the bundled CSV, imputes and scales data, encodes categorical features, uses the ranking and sorting functions, calculates summary statistics and evaluation metrics, estimates both a gradient and Jacobian, and calls the reusable benchmarking harness.

## Performance comparison

The NumPy implementations below are compared against Python-loop implementations using `numcompute.benchmarking.run_all(n=2000, repeats=5, seed=42)` with `time.perf_counter`. Times are measured means; speedup is the Python-loop mean divided by NumPy mean.

| Operation | Python loop (ms) | NumPy (ms) | Speedup |
|---|---:|---:|---:|
| dot product | 0.345 | 0.004 | 96.5x |
| element-wise square | 0.154 | 0.002 | 95.1x |
| euclidean distance | 0.436 | 0.008 | 56.6x |
| row-wise mean | 0.248 | 0.017 | 14.2x |
| softmax | 1.477 | 0.015 | 96.6x |

Environment: Python 3.13.5, NumPy 2.3.5, Linux x86_64, CPU not reported, CPython. Values vary by hardware and system load. Reproduce with:

```bash
python -m numcompute.benchmarking
```

This command uses the module defaults (`n=10000`, `repeats=20`, `seed=42`); for the table's exact settings, run:

```bash
python -c "from numcompute.benchmarking import run_all; run_all(n=2000, repeats=5, seed=42)"
```
