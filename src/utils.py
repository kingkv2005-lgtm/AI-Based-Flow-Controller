import numpy as np
import joblib


def create_sequences(X_scaled, Y_scaled, window):
    """
    Convert flat scaled arrays into sliding-window sequences for the GRU.

    Args:
        X_scaled (np.ndarray): Scaled input features, shape (N, num_features).
        Y_scaled (np.ndarray): Scaled target values, shape (N, 1).
        window (int): Number of timesteps per sequence.

    Returns:
        tuple(np.ndarray, np.ndarray): (X_seq, Y_seq) sequence arrays.
    """
    X_seq, Y_seq = [], []

    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i - window:i, :])
        Y_seq.append(Y_scaled[i])

    return np.array(X_seq), np.array(Y_seq)


def save_scalers(x_scaler, y_scaler, x_path, y_path):
    """Persist fitted X/Y scalers to disk using joblib."""
    joblib.dump(x_scaler, x_path)
    joblib.dump(y_scaler, y_path)


def load_scalers(x_path, y_path):
    """Load previously saved X/Y scalers from disk."""
    x_scaler = joblib.load(x_path)
    y_scaler = joblib.load(y_path)
    return x_scaler, y_scaler


def preprocess_input(raw_values, x_scaler, window):
    """
    Scale a single raw input sample and replicate it across the GRU's
    expected window length to form a model-ready input tensor.

    Args:
        raw_values (list or np.ndarray): Raw feature values in the order
            [Setpoint, Error, DeltaP, Valve_pos, Control_signal,
             Pipe_Diameter, Temperature].
        x_scaler: Fitted scaler used to scale the input features.
        window (int): Sequence length expected by the model.

    Returns:
        np.ndarray: Float32 array of shape (1, window, num_features).
    """
    x = np.array([raw_values])
    x_scaled = x_scaler.transform(x)

    history = np.repeat(x_scaled, window, axis=0)
    x_input = history.reshape(1, window, x_scaled.shape[1]).astype(np.float32)

    return x_input
