"""Small, inspectable model search; test labels never enter model selection."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def select_model(X_train, y_train, seed=42):
    """Compare candidates using identical five-fold training partitions."""
    # Learning note: scaling belongs inside the pipeline so each fold learns
    # its own means and standard deviations, without seeing validation rows.
    pipeline = Pipeline([("scale", "passthrough"), ("model", DummyRegressor())])
    grid = [
        {"model": [DummyRegressor(strategy="mean")]},
        {"model": [LinearRegression()], "scale": [StandardScaler()]},
        {"model": [Ridge()], "scale": [StandardScaler()],
         "model__alpha": [0.1, 1.0, 10.0]},
        {"model": [RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)],
         "model__max_depth": [3, None], "model__min_samples_leaf": [1, 3]},
    ]
    search = GridSearchCV(
        pipeline, grid, scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        refit=True, n_jobs=1, error_score="raise",
    ).fit(X_train, y_train)
    # sklearn maximizes scores, so it negates losses such as MAE.
    records = []
    for i, params in enumerate(search.cv_results_["params"]):
        scores = [-search.cv_results_[f"split{j}_test_score"][i] for j in range(5)]
        records.append({
            "model": type(params["model"]).__name__,
            "parameters": {k: v for k, v in params.items() if k.startswith("model__")},
            "cv_mae_mean": float(np.mean(scores)),
            "cv_mae_std": float(np.std(scores)),
            "fold_mae": scores,
            "selected": bool(i == search.best_index_),
        })
    return search.best_estimator_, records
