from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/house_model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(features: list):

    prediction = model.predict([features])

    return {"predicted_price": prediction.tolist()}