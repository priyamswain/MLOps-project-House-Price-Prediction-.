import pandas as pd
import joblib
import os

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("data/housing.csv")

X = df.drop("price", axis=1)
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Calculate metric
mse = mean_squared_error(y_test, pred)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/house_model.pkl")

# ---------------- MLflow Tracking ----------------

mlflow.start_run()

mlflow.log_param("model", "LinearRegression")

mlflow.log_metric("mse", mse)

mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

print("Training complete and logged in MLflow")