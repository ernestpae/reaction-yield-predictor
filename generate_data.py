"""
PROJECT: REACTION YIELD PREDICTOR (V1.0)
OBJECTIVE: Use Machine Learning to predict chemical yields based on reaction conditions.

--- THE LAB ASSISTANT ANALOGY ---
1. DATA (The Notebook): 
   We simulate 100 'virtual' lab experiments. Each has a specific Temperature, 
   Time, and Concentration. This is our raw research history.

2. MODEL (The Assistant's Brain): 
   We train a 'Linear Regression' model. Think of this as a junior lab assistant 
   studying our notebook to find the mathematical 'secret formula' of the reaction.

3. PREDICTIONS (The Final Exam): 
   We hide 20 experiments from the assistant and ask them to 'guess' the yields.
   By comparing their guesses to the real results, we calculate their 'Grade':
   - MAE: How many percentage points the assistant is 'off' on average.
   - R2 Score: How much of the chemistry's logic the assistant actually understood.

GOAL: Build a tool that predicts outcomes of reactions we HAVEN'T run yet.
"""

# DAY 1
import pandas as pd # Data management (The 'Excel' of Python)
import numpy as np  # Math & random number generation
# DAY 2
from sklearn.model_selection import train_test_split # Tool to split data into "Study" and "Test" sets
from sklearn.linear_model import LinearRegression    # The math model that finds a "Line of Best Fit"
from sklearn.metrics import mean_absolute_error, r2_score # Tools to score the model
import matplotlib.pyplot as plt # The standard library for creating graphs
# DAY 3
from sklearn.ensemble import RandomForestRegressor
                    # DAY 1
# Ensures the "random" numbers are the same every time you run it
np.random.seed(0)
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
#print(data.head()) # Show top 5 rows
data.to_csv("reaction_data.csv", index=False) # Save to file


             # DAY 2
# --- 1. SEPARATING FEATURES AND TARGET ---
# X (Capitalized): The "Features" or inputs. Using double brackets [[...]] returns a table/DataFrame.
# y (Lowercase): The "Target" or outcome. A single bracket [...] returns a single column/Series.
X = data[["temperature", "time", "concentration"]]
y = data["yield"]

# --- 2. THE TRAIN-TEST SPLIT ---
# test_size=0.2: Reserves 20% of the experiments for the final "Exam" (testing).
# random_state=42: Locks the "shuffle" so the split is identical every time you run it.
# This ensures your R2 score stays consistent, making your research reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#random_state=42

# --- 3. INITIALIZING THE MODEL ---
# Creates a blank Linear Regression object. Think of this as a student ready to learn.
model = LinearRegression()

# --- 4. THE TRAINING STEP (.fit) ---
# This is where the learning happens. The model looks for the relationship between X_train and y_train.
model.fit(X_train, y_train)

# --- 5. THE PREDICTION STEP (.predict) ---
# We give the model "new" inputs (X_test) that it hasn't seen before to see what Yield % it predicts.
predictions = model.predict(X_test)


# Calculate Scores: Comparing 'Predictions' to the 'Real' test answers (y_test)
mae = mean_absolute_error(y_test, predictions) # Finds the average 'miss'
r2 = r2_score(y_test, predictions)             # Finds the 'reliability' (0 to 1)

print(f"--- Model Performance ---")
print(f"Mean Absolute Error: {mae:.2f}%") # If 4.09, model is off by ~4% on average
print(f"R2 Score: {r2:.4f}")               # If 0.90, model explains 90% of the chemistry

# Create Parity Plot: A visual 'Exam' result
plt.scatter(y_test, predictions, alpha=0.5) # Plot dots for each reaction guess
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') # Red dashed line = Perfection
plt.xlabel("Actual Yield (%)")   # What happened in the 'lab'
plt.ylabel("Predicted Yield (%)") # What the 'model' guessed
plt.title("Model Accuracy: Predicted vs. Actual")
plt.show() # Opens the window to see your graph


# --- DAY 3: RANDOM FOREST REGRESSOR ---
# Instead of one 'line,' this model builds a 'forest' of 100 Decision Trees.
# Each tree votes on the predicted yield, and the average becomes the final answer.
# n_estimators=100: The number of individual 'trees' in the forest.
# random_state=42: Ensures the 'forest' grows the same way every time you run it.
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Training: The trees learn by asking "If/Then" questions about your data.
model_rf.fit(X_train, y_train)

# Prediction: The forest predicts the yield for your test experiments.
predictions_rf = model_rf.predict(X_test)

