import os
import numpy as np
import onnxruntime as ort

from utils import load_scalers, preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

X_SCALER_PATH = os.path.join(MODELS_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(MODELS_DIR, "y_scaler.pkl")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "gru_model.onnx")

WINDOW = 50


def load_inference_assets():
    """Load the ONNX runtime session and the fitted X/Y scalers."""
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    x_scaler, y_scaler = load_scalers(X_SCALER_PATH, Y_SCALER_PATH)
    return session, x_scaler, y_scaler


def predict_flow(session, x_scaler, y_scaler, raw_values, window=WINDOW):
    """Run a single prediction through the ONNX model and return real-scale flow."""
    x_input = preprocess_input(raw_values, x_scaler, window)
    y_pred = session.run(None, {"input": x_input})[0]
    flow = y_scaler.inverse_transform(y_pred)[0, 0]
    return flow


def run_inference_loop():
    """Interactively prompt for process variables and print predicted flow."""
    session, x_scaler, y_scaler = load_inference_assets()

    while True:
        try:
            sp = float(input("Setpoint: "))
            error = float(input("Error: "))
            dp = float(input("DeltaP: "))
            valve = float(input("Valve Position: "))
            control = float(input("Control Signal: "))
            pipe = float(input("Pipe Diameter: "))
            temp = float(input("Temperature: "))

            # Correct feature order
            raw_values = [sp, error, dp, valve, control, pipe, temp]
            flow = predict_flow(session, x_scaler, y_scaler, raw_values)

            print(f"\n➡️ Predicted Flow: {flow:.4f}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    run_inference_loop()
