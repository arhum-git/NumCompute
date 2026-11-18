# Verification notes (supporting the later report)

This is a developer verification log, **not** the required 4-page report. It documents actual checks performed during construction and keeps debugging examples grounded.

## Reproduced integration bugs and closures

1. **Growing one-hot feature width broke buffered retraining.** After `OneHotEncoder.partial_fit()` learned new categories, the old tree expected its previous input width. A test in `tests/test_stream_pipeline_io.py` initially failed with `ValueError: Expected 2 features, got 4`. The pipeline now performs a fresh `.fit` **at the existing buffer threshold** when the encoded width changes, while maintaining the last trained model and snapshot until that point. The focused streaming tests passed afterward.
2. **The provided student CSV contains a missing target.** The initial command-line demo and notebook raised `ValueError: Missing target labels are not supported`. The CSV reader now has an explicit `skip_missing_targets=True` option (default: reject). Both demo entrypoints enable it. A test verifies that only that row is dropped and all nonmissing labels remain.
3. **A test uncovered misleading array-copy documentation.** The array stream converts non-float64 input to a float64 array, which may allocate storage. Its docstring and test now distinguish the initial conversion from slice views. The full test suite passes for float64 views.

## Verification steps

From the project root:

```bash
python -m pip install -e .
python -m pytest -q
python demo/run_stream.py
python -m benchmark.stream_benchmark
```

The runnable Jupyter notebook was additionally executed through NotebookClient for all cells. Its plotting cells and assertions completed. The benchmark uses a random seed, repeatable data, and clear evaluation denominators; timing measurements will vary by machine.

## Interpretation notes

- Initial stream chunks are **not evaluated** before the model has trained; these log accuracy=`NaN`. Reporting them as zero would introduce a fake performance penalty.
- The bundled CSV has five grades/classes. The binary AUC scores positive class 1 versus all others, so it should not be interpreted as multiclass ROC-AUC.
- The higher buffer size results in a later first evaluation; cross-buffer cumulative accuracies therefore have different denominators. Compare with this caveat rather than treating them as identical hold-out sets.
- Exact quantiles/medians and exact ROC-AUC require retaining input records or scores. Use bounded `history_size` for rolling quantiles when constant memory is more important.
