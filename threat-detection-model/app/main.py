from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import logging
from keras.models import load_model
import keras

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

from keras.models import load_model
model = load_model('static/anomaly_detection_model.h5', compile=False)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

training_mean = float(np.load('static/training_mean.npy'))  
training_std = float(np.load('static/training_std.npy'))   
threshold = float(np.load('static/threshold.npy'))     

TIME_STEPS = 288

class InputData(BaseModel):
    data: list

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/normalization-params")
def get_normalization_params():
    return {
        "mean": training_mean.tolist(),
        "std": training_std.tolist()
    }

@app.post("/api/predict")
def predict(data: InputData):
    try:
        logging.info("Starting prediction")
        logging.info(f"Received data length: {len(data.data)}")

        if not isinstance(data.data, list):
            raise ValueError("Input data must be a list.")
        if not all(isinstance(x, (int, float)) for x in data.data):
            raise ValueError("All elements in the input data must be numeric.")
        if len(data.data) < TIME_STEPS:
            raise ValueError(f"Data must contain at least {TIME_STEPS} time steps.")

        logging.info("Normalizing data")
        df = pd.DataFrame(data.data, columns=['value'])
        normalized_data = (df - training_mean) / training_std

        logging.info("Creating sequences")
        sequences = create_sequences(normalized_data.values)
        if sequences.ndim != 3 or sequences.shape[2] != 1:
            sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))

        logging.info(f"Sequences shape: {sequences.shape}")

        logging.info("Running model prediction")
        predictions = model.predict(sequences)
        mae_loss = np.mean(np.abs(predictions - sequences), axis=2).flatten()

        logging.info("Calculating anomalies")
        anomalies = mae_loss > threshold
        reconstruction_errors = np.repeat(mae_loss, TIME_STEPS).tolist()

        return {
            "isAnomalous": bool(np.any(anomalies)),
            "meanReconstructionError": float(np.mean(mae_loss)),
            "reconstructionErrors": reconstruction_errors,
            "threshold": float(threshold),
        }
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}", exc_info=True)
        return 