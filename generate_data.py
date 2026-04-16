# DAY 1
import pandas as pd # Data management (The 'Excel' of Python)
import numpy as np  # Math & random number generation
# DAY 2
from sklearn.model_selection import train_test_split # Tool to split data into "Study" and "Test" sets
from sklearn.linear_model import LinearRegression    # The math model that finds a "Line of Best Fit"
                    # DAY 1
# Create Features: The independent variables (Inputs)
data = pd.DataFrame({
    "temperature": np.random.uniform(20, 100, 100),    # Range: 20-100 C
    "time": np.random.uniform(1, 10, 100),             # Range: 1-10 Hours
    "concentration": np.random.uniform(0.1, 1.0, 100), # Range: 0.1-1.0 M
})

# Create Target: The dependent variable (Result)
# Yield = (Math Equation) + Random Experimental Noise
data["yield"] = (
    0.3 * data["temperature"] + 
    5 * data["time"] + 
    20 * data["concentration"] + 
    np.random.normal(0, 5, 100) # Adds realistic "messy" data
)

# Preview and Export
print(data.head()) # Show top 5 rows
data.to_csv("reaction_data.csv", index=False) # Save to file


             # DAY 2
# --- 1. SEPARATING FEATURES AND TARGET ---
# X (Capitalized): The "Features" or inputs. Using double brackets [[...]] returns a table/DataFrame.
# y (Lowercase): The "Target" or outcome. A single bracket [...] returns a single column/Series.
X = data[["temperature", "time", "concentration"]]
y = data["yield"]

# --- 2. THE TRAIN-TEST SPLIT ---
# test_size=0.2: Reserves 20% of your data for the final "Exam" (testing). 
# The model "studies" with the other 80% (training).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- 3. INITIALIZING THE MODEL ---
# Creates a blank Linear Regression object. Think of this as a student ready to learn.
model = LinearRegression()

# --- 4. THE TRAINING STEP (.fit) ---
# This is where the learning happens. The model looks for the relationship between X_train and y_train.
model.fit(X_train, y_train)

# --- 5. THE PREDICTION STEP (.predict) ---
# We give the model "new" inputs (X_test) that it hasn't seen before to see what Yield % it predicts.
predictions = model.predict(X_test)