"""Reproduce the original toy data without overwriting the tracked CSV.

Run from the repository root: python examples/regenerate_data.py
The formula is an educational construction, not a measured chemical law.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_original():
    # RandomState intentionally preserves the original np.random.seed(0) sequence.
    rng = np.random.RandomState(0)
    frame = pd.DataFrame({
        "temperature": rng.uniform(20, 100, 100),
        "time": rng.uniform(1, 10, 100),
        "concentration": rng.uniform(0.1, 1, 100),
    })
    frame["yield"] = (0.3 * frame.temperature + 5 * frame.time
                      + 20 * frame.concentration + rng.normal(0, 5, 100))
    return frame


if __name__ == "__main__":
    output = Path("artifacts/regenerated_data.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_original().to_csv(output, index=False)
    print(output)
