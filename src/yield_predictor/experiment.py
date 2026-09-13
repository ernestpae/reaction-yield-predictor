"""Reproducible experiments, diagnostics, and batch inference."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots on servers and CI without a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import FEATURES, TARGET, split_indices, validate_frame
from .modeling import select_model


def scores(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction)) if np.var(y) > 0 else None}


def train(data_path, output, *, strategy="random", seed=42):
    """Select on training CV, then evaluate the selected model and a baseline."""
    data_path, output = Path(data_path), Path(output)
    frame = validate_frame(pd.read_csv(data_path))
    train_ids, test_ids = split_indices(frame, strategy, seed)
    X_train, X_test = frame.iloc[train_ids][FEATURES], frame.iloc[test_ids][FEATURES]
    y_train, y_test = frame.iloc[train_ids][TARGET], frame.iloc[test_ids][TARGET]
    model, comparison = select_model(X_train, y_train, seed)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    prediction = model.predict(X_test)
    baseline_prediction = baseline.predict(X_test)
    # Post-selection diagnostic only: do not use this test-set analysis to
    # tune features and still call the same test set an independent evaluation.
    importance = permutation_importance(
        model, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=20, random_state=seed, n_jobs=1,
    )
    ranges = {f: [float(X_train[f].min()), float(X_train[f].max())] for f in FEATURES}
    result = {
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_scope": "Condition-only regression; bundled data are synthetic",
        "seed": seed, "split": strategy, "train_rows": train_ids.tolist(),
        "test_rows": test_ids.tolist(), "training_ranges": ranges,
        "selected_model": type(model.named_steps["model"]).__name__,
        "selection_metric": "Five-fold training CV MAE; lowest mean wins",
        "cv_candidates": comparison,
        "holdout": scores(y_test, prediction),
        "baseline_holdout": scores(y_test, baseline_prediction),
        "permutation_importance": {
            f: {"mae_increase_mean": float(importance.importances_mean[i]),
                "shuffle_std": float(importance.importances_std[i])}
            for i, f in enumerate(FEATURES)
        },
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame({"row_id": test_ids, "actual_yield": y_test.to_numpy(),
                  "predicted_yield": prediction, "baseline_yield": baseline_prediction,
                  "residual": y_test.to_numpy() - prediction}).to_csv(
                      output / "predictions.csv", index=False)
    joblib.dump({"model": model, "training_ranges": ranges, "features": FEATURES,
                 "dataset_sha256": result["dataset_sha256"]}, output / "model.joblib")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].scatter(y_test, prediction, color="#176b87", label="Selected model")
    lo, hi = min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="gray")
    axes[0].set(xlabel="Synthetic target yield (%)", ylabel="Predicted yield (%)", title="Holdout parity")
    axes[1].scatter(prediction, y_test - prediction, color="#176b87")
    axes[1].axhline(0, linestyle="--", color="gray")
    axes[1].set(xlabel="Predicted yield (%)", ylabel="Target − prediction (pp)", title="Holdout residuals")
    fig.suptitle(f"Synthetic data · {strategy} split · {result['selected_model']}")
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)
    return result


def predict(model_path, input_path, output_path):
    """Load only a trusted locally produced joblib artifact (pickle-based)."""
    frame = validate_frame(pd.read_csv(input_path), target=False)
    bundle = joblib.load(model_path)
    if bundle["features"] != FEATURES:
        raise ValueError("Model artifact has an incompatible feature schema")
    prediction = bundle["model"].predict(frame)
    outside = np.zeros(len(frame), dtype=bool)
    for feature, (lo, hi) in bundle["training_ranges"].items():
        outside |= ~frame[feature].between(lo, hi).to_numpy()
    result = frame.assign(predicted_yield=prediction, outside_training_range=outside,
                          outside_physical_yield_range=(prediction < 0) | (prediction > 100))
    # Flag extrapolation rather than silently clipping outputs into [0, 100].
    # Being inside each feature's range is not proof of joint-domain coverage.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
