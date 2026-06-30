# AI-Based Flow Controller using GRU Neural Networks

An AI-driven industrial flow prediction system that estimates fluid flow without relying on physical flow sensors. The project leverages **PyTorch**, **GRU (Gated Recurrent Unit) Neural Networks**, and **ONNX Runtime** to perform real-time inference on time-series process data, making it suitable for deployment in industrial automation environments.

---

## Overview

Traditional industrial flow measurement often depends on dedicated hardware flow sensors, which can increase system cost and maintenance. This project demonstrates a **sensorless soft-sensor approach** that predicts flow using process parameters such as setpoint, pressure difference, valve position, control signal, pipe diameter, and temperature.

The trained GRU model is exported to **ONNX** and executed using **ONNX Runtime**, providing a deployment-ready inference pipeline for embedded and edge devices.

---

## Features

- Deep Learning-based industrial flow prediction
- GRU Neural Network implemented using PyTorch
- Data preprocessing and feature scaling
- Sliding window sequence generation for time-series learning
- Model evaluation using RMSE, MAE and R²
- ONNX model export
- Real-time inference using ONNX Runtime
- Deployment-ready architecture for embedded systems

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- ONNX
- ONNX Runtime
- Joblib
- Matplotlib

---

## Input Features

The model predicts flow using the following process variables:

- Setpoint
- Error
- Delta Pressure (ΔP)
- Valve Position
- Control Signal
- Pipe Diameter
- Temperature

Target Output:

- Flow

---

## Model Architecture

- Model: GRU (Gated Recurrent Unit)
- Input Features: 7
- Hidden Units: 32
- Layers: 1
- Sequence Window: 50
- Output: Predicted Flow

---

## Evaluation Metrics

The trained model is evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

These metrics are used to measure prediction accuracy and model performance.

---

## Project Structure

```
AI-Based-Flow-Controller
│
├── train.py
├── inference.py
├── syntheticdatafinal.csv
├── x_scaler.pkl
├── y_scaler.pkl
├── gru_model.onnx
├── requirements.txt
├── README.md
└── images/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI-Based-Flow-Controller.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python train.py
```

Run inference

```bash
python inference.py
```

---

## Future Improvements

- Deploy on STM32 Edge AI platforms
- Real-time sensor integration
- Web-based monitoring dashboard
- Cloud-based industrial monitoring
- Model optimization for embedded deployment

---

## License

This project is developed for academic and research purposes.

---

## 👨‍💻 Author

**Vishal K**
**Jemmy Angel K**
**Vishwa R**

- LinkedIn: https://www.linkedin.com/in/vishal-k-a23ba9255/
- GitHub: https://github.com/kingkv2005-lgtm
