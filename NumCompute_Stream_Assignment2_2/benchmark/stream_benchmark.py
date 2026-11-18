"""Reproducible benchmark for stream vectorisation and tree buffer sizes.

Run: python -m benchmark.stream_benchmark
No sklearn or pandas used. The experiment uses a fixed synthetic drift stream.
"""

import csv
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
from numcompute_stream import DecisionTreeClassifier, StreamTrainer, iter_array_chunks


def make_stream(n=1200, seed=42):
    """A small binary stream with a midpoint class drift for reproducibility."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    boundary = X[:, 0] + 0.4 * X[:, 1]
    boundary[n // 2:] = X[n // 2:, 0] - 0.6 * X[n // 2:, 1]
    y = (boundary > 0).astype(int)
    return X, y


def confusion_loop(y_true, y_pred):
    """Naive reference implementation, with known binary labels (0 and 1)."""
    out = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        out[int(actual), int(predicted)] += 1
    return out


def confusion_vectorised(y_true, y_pred):
    """Vectorised NumPy equivalent to the loop implementation."""
    return np.bincount(2 * y_true + y_pred, minlength=4).reshape(2, 2)


def best_of(fn, *args, repeats=7):
    """Warm up once, then best-of-k with GC disabled during timed executions."""
    fn(*args)
    enabled = gc.isenabled()
    try:
        gc.disable()
        times = []
        for _ in range(repeats):
            start = perf_counter()
            fn(*args)
            times.append(perf_counter() - start)
        return min(times)
    finally:
        if enabled:
            gc.enable()


def benchmark_vectorisation(n=100_000, seed=42):
    rng = np.random.default_rng(seed)
    truth, pred = rng.integers(0, 2, size=(2, n), dtype=np.int64)
    assert np.array_equal(confusion_loop(truth, pred), confusion_vectorised(truth, pred))
    loop = best_of(confusion_loop, truth, pred)
    vector = best_of(confusion_vectorised, truth, pred)
    return {"n_samples": n, "loop_seconds": loop, "vectorised_seconds": vector,
            "speedup": loop / vector}


def benchmark_buffers(X, y, sizes=(32, 128, 512), chunk_size=32):
    """Prequential comparison of accuracy, retraining time, and retrain count."""
    rows = []
    for size in sizes:
        tree = DecisionTreeClassifier(max_depth=5, buffer_size=size, random_state=42)
        trainer = StreamTrainer(tree)
        logs = trainer.run(iter_array_chunks(X, y, chunk_size))
        evaluated = trainer.metrics.result()["n_samples"]
        accuracy = trainer.metrics.result()["accuracy"]
        rows.append({"buffer_size": size, "evaluated_samples": evaluated,
                     "prequential_accuracy": accuracy,
                     "n_retrains": trainer.n_retrains,
                     "total_fit_seconds": sum(r["fit_seconds"] for r in logs),
                     "max_buffer_bytes": max(r["buffer_memory_bytes"] for r in logs)})
    return rows


def main(output="benchmark/results.csv"):
    X, y = make_stream()
    vec = benchmark_vectorisation()
    rows = benchmark_buffers(X, y)
    print("Loop vs NumPy confusion counting:", vec)
    print("Tree buffer-size comparison:")
    for row in rows:
        print(row)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return vec, rows


if __name__ == "__main__":
    main()
