# app/main.py

import joblib
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import sys
import os

# Add the project root to the Python path to allow imports from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# import self-defined modules
from src.model_def import MLPRegression  #make sure src is in PYTHONPATH
from app.schemas import StockFeatures, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app.config import FEATURE_COLS

# ---load model and scaler---
model = None
scaler = None

# 2. load feature length from the config file
INPUT_DIM = len(FEATURE_COLS)

# add this otherwise run into error when running docker
# torch.set_num_threads(1)
# torch.backends.openmp.enabled = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    print("Loading model and scaler...")
    try:
        scaler_path = os.path.join(PROJECT_ROOT, "model/scaler.joblib")
        scaler = joblib.load(scaler_path)

        # load the pretrained model
        model_path = os.path.join(PROJECT_ROOT, "model/model.pth")
        model = MLPRegression(input_dim=INPUT_DIM)
        # load state_dict (ensure on correct device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        print("Model and scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model/scaler: {e}")

    yield

    # cleaning after app is closed
    print("Cleaning up...")
    model = None
    scaler = None


app = FastAPI(
    title="Stock Return Prediction API",
    description="API for predicting future stock returns using a PyTorch MLP model.",
    version="1.0.0",
    lifespan=lifespan
)


def get_prediction(features: StockFeatures) -> float:
    """
    Make prediction on single instance
    """
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")

    try:
        input_data = np.array([[getattr(features, col) for col in FEATURE_COLS]])

        input_scaled = scaler.transform(input_data)

        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor)

        return prediction.item()

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/", summary="Health Check")
def read_root():
    """
    check if the server is running
    """
    return {"status": "ok", "message": "Stock Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse, summary="Predict Single Instance")
def predict_single(features: StockFeatures):
    """
    get one instance and return predicted future return on the fifth day
    """
    prediction_value = get_prediction(features)
    return PredictionResponse(predicted_future_return_5d=prediction_value)


@app.post("/predict_batch", response_model=BatchPredictionResponse, summary="Predict Batch Instances")
def predict_batch(request: BatchPredictionRequest):
    """
    get batch instances and return batch predictions
    """
    try:
        predictions = [get_prediction(features) for features in request.instances]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,)

