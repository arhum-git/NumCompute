"""Command-line version of demo/stream_demo.ipynb; no Jupyter required.

Run from the repository root: python demo/run_stream.py
Outputs figures to demo/output/, and logs the same progressive experiment.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numcompute_stream import (DecisionTreeClassifier, Pipeline, StreamTrainer,
                               StandardScaler, SimpleImputer, iter_csv_chunks)
from numcompute_stream.visualise import (plot_metric_over_time, compare_models,
                                          plot_predictions_vs_ground_truth)


def run():
    root = Path(__file__).resolve().parent
    csv_path = root / "dataset" / "test_student_performance_data.csv"
    features = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
    outputs = root / "output"
    outputs.mkdir(exist_ok=True)

    def make_trainer(buffer_size):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("tree", DecisionTreeClassifier(max_depth=5, min_samples_split=4,
                                            buffer_size=buffer_size, criterion="gini")),
        ])
        return StreamTrainer(pipe, window_size=160)

    small = make_trainer(buffer_size=80)
    large = make_trainer(buffer_size=320)
    for X_chunk, y_chunk in iter_csv_chunks(csv_path, chunk_size=40,
                                            feature_columns=features, target_column="GradeClass",
                                            skip_missing_targets=True):
        small.process_chunk(X_chunk, y_chunk)
        large.process_chunk(X_chunk, y_chunk)

    print("Small buffer: evaluated", small.metrics.result()["n_samples"],
          "accuracy", small.metrics.result()["accuracy"],
          "retrains", small.n_retrains)
    print("Large buffer: evaluated", large.metrics.result()["n_samples"],
          "accuracy", large.metrics.result()["accuracy"],
          "retrains", large.n_retrains)
    print("Last log entry:", small.logs[-1])
    print("Confusion matrix:\n", small.metrics.result()["confusion_matrix"])

    fig, _ = plot_metric_over_time([x["cumulative_accuracy"] for x in small.logs],
                                   "Progressive accuracy — buffer 80", "Accuracy",
                                   save_path=outputs / "accuracy.png")
    plt.close(fig)
    fig, _ = compare_models([x["cumulative_accuracy"] for x in small.logs],
                            [x["cumulative_accuracy"] for x in large.logs],
                            ("Buffer 80", "Buffer 320"),
                            save_path=outputs / "buffer_comparison.png")
    plt.close(fig)
    if small.last_y_true_ is not None:
        fig, _ = plot_predictions_vs_ground_truth(small.last_y_true_, small.last_y_pred_,
                                                   save_path=outputs / "latest_predictions.png")
        plt.close(fig)
    return small, large


if __name__ == "__main__":
    run()
