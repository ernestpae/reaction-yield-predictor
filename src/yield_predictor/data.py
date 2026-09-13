"""Explicit schema checks and splits shared by training and inference."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["temperature", "time", "concentration"]
TARGET = "yield"


def validate_frame(frame: pd.DataFrame, *, target: bool = True) -> pd.DataFrame:
    """Fail early rather than silently coercing malformed experimental tables."""
    expected = FEATURES + ([TARGET] if target else [])
    if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
        raise ValueError(f"Expected exactly these columns: {expected}")
    if frame.empty:
        raise ValueError("Input table is empty")
    try:
        clean = frame.loc[:, expected].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("All inputs must be numeric") from exc
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Missing and infinite values are not supported")
    if (clean["temperature"] < -273.15).any():
        raise ValueError("Temperature must be at least -273.15 degrees Celsius")
    if (clean[["time", "concentration"]] <= 0).any().any():
        raise ValueError("Time and concentration must be positive")
    if target:
        if not clean[TARGET].between(0, 100).all():
            raise ValueError("Yield must be between 0 and 100 percent")
        # Replicates need group-aware splitting; don't accidentally leak them.
        if clean.duplicated(subset=FEATURES).any():
            raise ValueError("Duplicate conditions require a replicate-aware split")
        if len(clean) < 30 or clean[TARGET].nunique() < 2:
            raise ValueError("Training requires at least 30 rows and a varying target")
    return clean


def split_indices(frame: pd.DataFrame, strategy: str, seed: int):
    """Return positional indices; never consult yields when defining the split."""
    indices = np.arange(len(frame))
    if strategy == "random":
        return train_test_split(indices, test_size=0.2, random_state=seed)
    if strategy == "high_temperature":
        # Hold out the upper temperature range, including ties at its boundary.
        boundary = np.sort(frame["temperature"])[int(0.8 * len(frame))]
        test = indices[frame["temperature"].to_numpy() >= boundary]
        train = indices[frame["temperature"].to_numpy() < boundary]
        if len(train) < 20 or len(test) < 2:
            raise ValueError("Not enough distinct temperatures for this split")
        return train, test
    raise ValueError(f"Unknown split strategy: {strategy}")
