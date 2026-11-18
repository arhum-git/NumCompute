"""Pipeline streaming, snapshot invariants, CSV ingestion and progressive logging."""
import csv
import numpy as np
import pytest
from numcompute_stream import (Pipeline, DecisionTreeClassifier, StandardScaler,
    OneHotEncoder, MinMaxScaler, SimpleImputer, StreamTrainer,
    iter_csv_chunks, iter_array_chunks)


def test_pipeline_buffer_threshold():
    p = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=4))])
    p.partial_fit([[0], [1]], [0, 0])
    assert p.n_buffered_samples_ == 2
    with pytest.raises(RuntimeError):
        p.predict([[0.]])
    p.partial_fit([[2], [3]], [1, 1])
    assert p.n_buffered_samples_ == 0
    np.testing.assert_array_equal(p.predict([[0], [3]]), [0, 1])


def test_pipeline_scaler_snapshot_stays_fixed_between_retrains():
    p = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=4))])
    p.partial_fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    before = p.predict([[1], [3]])
    p.partial_fit([[100]], [0])  # changes live mean/scale, not trained prediction state
    np.testing.assert_array_equal(p.predict([[1], [3]]), before)
    assert p.get_step("scale").mean_[0] > 2.0


def test_pipeline_encoder_snapshot_growing_width():
    p = Pipeline([("cat", OneHotEncoder(handle_unknown="ignore")),
                  ("tree", DecisionTreeClassifier(buffer_size=2))])
    p.partial_fit(np.array([["red"], ["blue"]]), [0, 1])
    n = p.estimator.n_features_in_
    assert p.predict([["red"]])[0] in (0, 1)
    p.partial_fit(np.array([["green"]]), [1])
    assert p.estimator.n_features_in_ == n
    assert p.predict([["red"]]).shape == (1,)
    p.partial_fit(np.array([["yellow"]]), [1])
    assert p.estimator.n_features_in_ == n + 2


def test_pipeline_fit_resets_buffer():
    p = Pipeline([("tree", DecisionTreeClassifier(buffer_size=9))])
    p.partial_fit([[0]], [0])
    p.fit([[0], [1]], [0, 1])
    assert p.n_buffered_samples_ == 0
    assert p.predict([[1]])[0] == 1


def test_pipeline_bad_duplicate_step_names():
    with pytest.raises(ValueError):
        Pipeline([("duplicate", StandardScaler()), ("duplicate", DecisionTreeClassifier())])


def test_pipeline_empty_rejected():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_bad_intermediate_step():
    with pytest.raises(TypeError):
        Pipeline([("broken", object()), ("tree", DecisionTreeClassifier())])


def test_pipeline_internal_type_error_propagates_single_call():
    class Broken:
        calls = 0
        def fit(self, X, y):
            self.calls += 1
            raise TypeError("internal failure")
        def partial_fit(self, X, y):
            self.calls += 1
            raise TypeError("internal failure")
        def predict(self, X):
            return np.zeros(len(X))
    estimator = Broken()
    p = Pipeline([("broken", estimator)])
    with pytest.raises(TypeError, match="internal failure"):
        p.partial_fit([[1]], [0])
    assert estimator.calls == 1


def test_pipeline_input_dim_mismatch():
    p = Pipeline([("tree", DecisionTreeClassifier(buffer_size=2))])
    p.partial_fit([[0]], [0])
    with pytest.raises(ValueError):
        p.partial_fit([[1, 2]], [0])


def test_pipeline_get_step():
    p = Pipeline([("tree", DecisionTreeClassifier())])
    assert isinstance(p.get_step("tree"), DecisionTreeClassifier)
    with pytest.raises(KeyError):
        p.get_step("missing")


def test_pipeline_predict_proba():
    p = Pipeline([("tree", DecisionTreeClassifier())]).fit([[0], [1]], [0, 1])
    np.testing.assert_allclose(p.predict_proba([[0]]).sum(axis=1), 1)


def test_pipeline_minmax_and_imputer():
    p = Pipeline([("impute", SimpleImputer()), ("minmax", MinMaxScaler()),
                  ("tree", DecisionTreeClassifier(buffer_size=2))])
    p.partial_fit([[1., np.nan], [2., 3.]], [0, 1])
    assert p.predict([[1., np.nan]]).shape == (1,)


def test_csv_custom_columns_and_chunk_shape(tmp_path):
    path = tmp_path / "stream.csv"
    path.write_text("id,x,y,label\n1,1,2,0\n2,,3,1\n3,4,5,1\n", encoding="utf-8")
    chunks = list(iter_csv_chunks(path, chunk_size=2, feature_columns=["x", "y"], target_column="label"))
    assert [x.shape for x, _ in chunks] == [(2, 2), (1, 2)]
    assert np.isnan(chunks[0][0][1, 0])
    np.testing.assert_array_equal(chunks[1][1], [1])


def test_csv_missing_label_rejected(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("x,label\n1,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_csv_chunks(path))


def test_csv_wrong_number_of_fields(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,y,label\n1,2,0\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_csv_chunks(path))


def test_array_chunks_view_float_data():
    X = np.arange(12, dtype=float).reshape(6, 2)
    chunks = list(iter_array_chunks(X, np.arange(6), chunk_size=4))
    assert [len(x) for x, y in chunks] == [4, 2]
    assert np.shares_memory(chunks[0][0], X)


def test_trainer_prequential_first_chunk_not_evaluated():
    t = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    first = t.process_chunk([[0], [1]], [0, 1])
    assert not first["evaluated"]
    assert np.isnan(first["chunk_accuracy"])
    second = t.process_chunk([[0], [1]], [0, 1])
    assert second["evaluated"]
    assert second["cumulative_accuracy"] == 1.0


def test_trainer_retrain_count():
    t = StreamTrainer(DecisionTreeClassifier(buffer_size=3))
    a = t.process_chunk([[0], [1]], [0, 1])
    b = t.process_chunk([[2]], [1])
    assert not a["retrained"] and b["retrained"]
    assert t.n_retrains == 1


def test_trainer_fit_and_score_standalone():
    t = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    t.fit_chunk([[0], [1]], [0, 1])
    scored = t.score_chunk([[0], [1]], [0, 1])
    assert scored["evaluated"]
    assert t.metrics.result()["n_samples"] == 2


def test_trainer_logs_memory_and_time():
    t = StreamTrainer(DecisionTreeClassifier(buffer_size=3))
    result = t.process_chunk([[0]], [0])
    assert result["buffer_memory_bytes"] > 0
    assert result["fit_seconds"] >= 0
    assert result["traced_peak_memory_bytes"] >= 0


def test_trainer_reset_logs_keeps_tree():
    t = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    t.process_chunk([[0], [1]], [0, 1])
    t.reset_logs()
    assert not t.logs
    assert t.n_retrains == 1


def test_trainer_pipeline_end_to_end():
    p = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(buffer_size=2))])
    t = StreamTrainer(p)
    t.process_chunk([[0], [1]], [0, 1])
    assert t.process_chunk([[0], [1]], [0, 1])["chunk_accuracy"] == 1.0


def test_csv_skip_missing_targets(tmp_path):
    path = tmp_path / "valid.csv"
    path.write_text("x,label\n1,0\n2,\n3,1\n", encoding="utf-8")
    pairs = list(iter_csv_chunks(path, chunk_size=2, skip_missing_targets=True))
    assert sum(len(y) for _, y in pairs) == 2
    np.testing.assert_array_equal(np.concatenate([y for _, y in pairs]), [0, 1])
