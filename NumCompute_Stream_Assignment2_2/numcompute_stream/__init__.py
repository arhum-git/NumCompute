"""NumCompute-Stream: a NumPy-only online processing and buffered tree toolkit.

Only visualise.py requires matplotlib. The legacy numcompute/ package is kept
unchanged so Assignment 2.1 remains reproducible on the original branch.
"""

from .tree import DecisionTreeClassifier
from .preprocessing import StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
from .stats import StreamingStats
from .metrics import (StreamingClassificationMetrics, StreamingAccuracy,
                      StreamingPrecision, StreamingRecall, StreamingF1,
                      StreamingConfusionMatrix, StreamingAUC)
from .pipeline import Pipeline
from .stream import StreamTrainer
from .io import iter_csv_chunks, iter_array_chunks

__all__ = ["DecisionTreeClassifier", "StandardScaler", "MinMaxScaler",
           "SimpleImputer", "OneHotEncoder", "StreamingStats",
           "StreamingClassificationMetrics", "StreamingAccuracy",
           "StreamingPrecision", "StreamingRecall", "StreamingF1",
           "StreamingConfusionMatrix", "StreamingAUC", "Pipeline",
           "StreamTrainer", "iter_csv_chunks", "iter_array_chunks"]
