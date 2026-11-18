"""Streaming CSV and array readers using Python's csv library and NumPy only."""

import csv
import numpy as np
from ._validation import validate_positive_int


def iter_csv_chunks(path, chunk_size=128, feature_columns=None, target_column=-1,
                    delimiter=",", has_header=True, missing_values=("", "NA", "NaN", "null"),
                    skip_missing_targets=False):
    """Yield numeric ``(X_chunk, y_chunk)`` arrays from a CSV in O(chunk*d) RAM.

    Parameters
    ----------
    path : str or Path
        CSV path, opened with UTF-8-SIG to ignore an optional BOM.
    chunk_size : int
        Maximum number of data rows per yielded chunk.
    feature_columns : list[str|int] or None
        Columns selected as input features. Default: all but target and first
        column when named ``StudentID``; for generic files, all but target.
    target_column : str or int
        Label column by header name or integer position (default: last column).
    has_header : bool
        Use the first row as column names.
    missing_values : tuple[str]
        Tokens mapped to float NaN in features.
    skip_missing_targets : bool
        If True, drop CSV records whose target is missing. Otherwise raise.

    Raises ValueError for missing headers, invalid numbers, inconsistent rows,
    or a target column present among the features. The returned target is
    numeric (integers when every label in a chunk is integral).
    """
    validate_positive_int(chunk_size, "chunk_size")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("delimiter must be a single character")

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader, None) if has_header else None
        if has_header and (not header or len(header) != len(set(header))):
            raise ValueError("Header is empty or contains duplicate column names")
        first_row = next(reader, None)
        if first_row is None:
            return
        width = len(header) if has_header else len(first_row)
        if width < 2:
            raise ValueError("CSV requires at least one feature and a target column")

        def resolve(key):
            if isinstance(key, str):
                if not has_header or key not in header:
                    raise ValueError(f"Unknown CSV column: {key}")
                return header.index(key)
            if isinstance(key, bool) or not isinstance(key, int) or not -width <= key < width:
                raise ValueError(f"CSV column index out of bounds: {key}")
            return key % width

        label_index = resolve(target_column)
        if feature_columns is None:
            features = [i for i in range(width)
                        if i != label_index and not (header and header[i] == "StudentID")]
        else:
            features = [resolve(col) for col in feature_columns]
        if not features or label_index in features or len(set(features)) != len(features):
            raise ValueError("Feature columns must be distinct and exclude the target")
        missing = set(missing_values)
        rows = []

        def parse(batch):
            if skip_missing_targets:
                batch = [row for row in batch if len(row) == width and row[label_index].strip() not in missing]
            X = np.empty((len(batch), len(features)), dtype=float)
            y = np.empty(len(batch), dtype=float)
            for i, row in enumerate(batch):
                if len(row) != width:
                    raise ValueError(f"CSV row has {len(row)} columns instead of {width}")
                for j, col in enumerate(features):
                    cell = row[col].strip()
                    try:
                        X[i, j] = np.nan if cell in missing else float(cell)
                    except ValueError as exc:
                        raise ValueError(f"Non-numeric feature in column {col}: {cell!r}") from exc
                label = row[label_index].strip()
                if label in missing:
                    raise ValueError("Missing target labels are not supported")
                try:
                    y[i] = float(label)
                except ValueError as exc:
                    raise ValueError("Non-numeric target encountered") from exc
            if not np.isfinite(y).all() or np.isinf(X).any():
                raise ValueError("CSV contained nonfinite target or infinite input")
            if np.all(y == np.floor(y)):
                y = y.astype(int)
            return X, y

        # The first data record was read to establish the column count.
        rows.append(first_row)
        for row in reader:
            rows.append(row)
            if len(rows) >= chunk_size:
                chunk = parse(rows)
                if len(chunk[0]):
                    yield chunk
                rows = []
        if rows:
            chunk = parse(rows)
            if len(chunk[0]):
                yield chunk


def iter_array_chunks(X, y, chunk_size=128):
    """Yield float64 array slices as views after one input validation pass.

    Validation may initially copy X to convert non-float64 input; when X is
    already float64, slices share memory. Storage is O(n*d) at worst for
    the dtype conversion, then O(1) per yielded slice.
    """
    from ._validation import numeric_2d, labels_1d
    size = validate_positive_int(chunk_size, "chunk_size")
    X = numeric_2d(X)
    y = labels_1d(y, len(X))
    for start in range(0, len(X), size):
        yield X[start:start + size], y[start:start + size]
