import pandas as pd
import numpy as np

# Ensures the "random" numbers are the same every time you run it
np.random.seed(0)

# Creating the features
data = pd.DataFrame({
    "temperature": np.random.uniform(20, 100, 100), # Celsius
    "time": np.random.uniform(1, 10, 100),          # Hours
    "concentration": np.random.uniform(0.1, 1.0, 100), # Molarity
})

# Creating the target (Yield %) using a linear relationship + some noise
data["yield"] = (
    0.3 * data["temperature"]
    + 5 * data["time"]
    + 20 * data["concentration"]
    + np.random.normal(0, 5, 100)
)

print("--- First 5 Rows of Synthetic Research Data ---")
print(data.head())

# Optional: Save it to a CSV so you can see it like an Excel file
data.to_csv("reaction_data.csv", index=False)