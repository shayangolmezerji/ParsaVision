#!/usr/bin/env python3

import uvicorn
import torch
import torch.nn as nn
import logging
import click
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

def load_model(model_path: str):
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        return None
    try:
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Fatal error: Could not load the model from file. Details: {e}")
        return None

AI_MODEL = None
app = FastAPI(
    title="ParsaVision AI Server",
    description="Serves a trained PyTorch model.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    data: List[float]

@app.get("/health")
def health_check():
    if AI_MODEL is not None:
        return {"status": "ok", "model_status": "loaded"}
    return {"status": "error", "model_status": "failed_to_load"}, 500

@app.post("/predict")
def get_prediction(request: PredictionRequest):
    if AI_MODEL is None:
        return {"error": "AI model not available. See logs."}, 500
    try:
        input_array = np.array(request.data, dtype=np.float32).reshape(1, 1, 28, 28)
        input_tensor = torch.from_numpy(input_array)
    except ValueError:
        return {"error": "Invalid input data. Expected a list of 784 numbers."}, 400
    with torch.no_grad():
        output = AI_MODEL(input_tensor)
        predicted_class = torch.argmax(output).item()
    logger.info(f"Prediction made for input: {predicted_class}")
    return {"predicted_class": predicted_class}

@click.command()
@click.option("--model-path", required=True, help="Path to the trained model file (e.g., 'simple_cnn.pt').")
@click.option("--host", default="0.0.0.0", help="Host address to bind to.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
def main(model_path, host, port):
    global AI_MODEL
    AI_MODEL = load_model(model_path)
    if AI_MODEL is None:
        logger.error("Exiting due to model loading failure.")
        return
    logger.info(f"Starting the AI Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
