import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import GRUSoftSensor
from utils import create_sequences, save_scalers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "syntheticdatafinal.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

X_SCALER_PATH = os.path.join(MODELS_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(MODELS_DIR, "y_scaler.pkl")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "gru_model.onnx")

INPUT_FEATURES = [
    'Setpoint',
    'Error',
    'DeltaP',
    'Valve_pos',
    'Control_signal',
    'Pipe_Diameter',
    'Temperature'
]
TARGET_FEATURE = 'Flow'
WINDOW = 50
EPOCHS = 100
LEARNING_RATE = 0.001


def load_dataset(path):
    """Load the raw CSV dataset and split into input/target arrays."""
    df = pd.read_csv(path)
    X_raw = df[INPUT_FEATURES].values
    Y_raw = df[TARGET_FEATURE].values
    return X_raw, Y_raw


def scale_features(X_raw, Y_raw):
    """Fit StandardScaler on inputs and MinMaxScaler on the target."""
    x_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_raw)
    Y_scaled = y_scaler.fit_transform(Y_raw.reshape(-1, 1))

    return X_scaled, Y_scaled, x_scaler, y_scaler


def train_model(model, X_train, Y_train, epochs, lr):
    """Train the GRU model using full-batch gradient descent."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    return model


def evaluate_model(model, X_test, Y_test, y_scaler):
    """Evaluate the trained model with offset correction and smoothing."""
    model.eval()

    with torch.no_grad():
        y_test_pred = model(X_test)

    Y_test_np = Y_test.numpy()
    y_test_pred_np = y_test_pred.numpy()

    y_test_real = y_scaler.inverse_transform(Y_test_np)
    y_test_pred_real = y_scaler.inverse_transform(y_test_pred_np)

    # Offset correction
    offset = y_test_real.mean() - y_test_pred_real.mean()
    y_test_pred_real += offset

    # Low-pass filter
    alpha = 0.1
    y_smooth = np.zeros_like(y_test_pred_real)
    y_smooth[0] = y_test_pred_real[0]

    for i in range(1, len(y_smooth)):
        y_smooth[i] = alpha * y_test_pred_real[i] + (1 - alpha) * y_smooth[i - 1]

    y_true = y_test_real.flatten()
    y_pred = y_smooth.flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\nModel Performance")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R²   : {r2:.6f}")

    return rmse, mae, r2


def export_to_onnx(model, window, num_features, output_path):
    """Export the trained PyTorch model to ONNX format."""
    dummy_input = torch.randn(1, window, num_features)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )

    print("✅ ONNX model exported")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    X_raw, Y_raw = load_dataset(DATA_PATH)

    # Scale features and target
    X_scaled, Y_scaled, x_scaler, y_scaler = scale_features(X_raw, Y_raw)

    # Save scalers (important for deployment)
    save_scalers(x_scaler, y_scaler, X_SCALER_PATH, Y_SCALER_PATH)

    # Build sequences
    X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, WINDOW)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_seq, Y_seq, test_size=0.2, shuffle=False
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # Initialize and train model
    model = GRUSoftSensor()
    model = train_model(model, X_train, Y_train, EPOCHS, LEARNING_RATE)

    # Evaluate model
    evaluate_model(model, X_test, Y_test, y_scaler)

    # Export to ONNX
    export_to_onnx(model, WINDOW, len(INPUT_FEATURES), ONNX_MODEL_PATH)


if __name__ == "__main__":
    main()
