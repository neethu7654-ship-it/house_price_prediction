# Create your models here.
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


import joblib

# Load dataset
df = pd.read_csv("C:\\Users\\NEETHU ANTONY A\\OneDrive\\Desktop\\Projects\\house_price_prediction\\ml_model\\dataset\\houses.csv")

# Data Cleaning
df
df.head()
df.isnull().sum()
df = df.dropna()
df.isnull().sum()

# Split features and target
X = df.drop("price", axis=1)
y = df["price"]

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)])

# Model
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=1000,
    random_state=42
)

# Pipeline (VERY IMPORTANT)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)    
rmse = mse ** 0.5

print("Model Performance:")
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Save model
joblib.dump(pipeline, "house_price_model.pkl")

print("✅ Model trained and saved successfully!")
