# AI-Based Flow Soft Sensor Using a GRU Neural Network

A research prototype that uses a **Gated Recurrent Unit (GRU)** neural network to estimate industrial process flow from other process variables. The project explores a data-driven soft sensor and a proposed path toward deployment on an STM32 edge device and integration with a pneumatic control valve.

> **Project status:** The GRU training workflow and ONNX model export are implemented. The repository includes a PC-side ONNX inference script. STM32 deployment using STM32CubeIDE / X-CUBE-AI and physical flow-loop validation were part of the intended system design, but were **not completed or verified on hardware**. The control-valve and flow calculations were analysed theoretically because access to the college flow laboratory was unavailable.

## Contents

- [Motivation](#motivation)
- [System concept](#system-concept)
- [Implemented machine-learning workflow](#implemented-machine-learning-workflow)
- [Model architecture](#model-architecture)
- [Dataset and input variables](#dataset-and-input-variables)
- [Evaluation](#evaluation)
- [ONNX inference](#onnx-inference)
- [Proposed embedded and control-system integration](#proposed-embedded-and-control-system-integration)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Motivation

Conventional flow measurement often relies on a dedicated flow transmitter. A **soft sensor** estimates a process variable from other available measurements instead. This project investigates whether a GRU model can estimate flow from process signals such as pressure difference, valve position and controller output.

The current dataset is synthetic and intended for academic experimentation. The results should not be interpreted as proof of performance on a live industrial plant.

## System concept

The project has two related parts:

1. **Flow estimation:** A GRU model receives a time window of process variables and predicts flow.
2. **Proposed control-system integration:** A controller output is intended to drive an I/P (current-to-pressure) converter. The resulting pneumatic signal operates a control-valve actuator, changing valve-stem position. The valve flow coefficient (`Cv`) and the applicable valve-sizing relationship can then be used to estimate flow under the relevant process conditions.

Conceptual signal path:

```text
Process variables
      |
      v
Preprocessing and feature scaling
      |
      v
GRU soft-sensor model
      |
      v
Predicted flow
      |
      v
Control strategy / electrical output
      |
      v
I/P converter
(electrical current -> pneumatic pressure)
      |
      v
Control-valve actuator and stem
      |
      v
Valve opening / Cv relationship
      |
      v
Theoretical flow calculation
```

This is a **conceptual system flow**, not a claim that the complete closed-loop arrangement was physically commissioned or tested. The exact I/P input range, pneumatic output range and valve-sizing equation must be selected for the actual instruments and fluid/service conditions.

## Implemented machine-learning workflow

The Python workflow in `src/` covers:

1. Loading the process dataset.
2. Separating the seven input variables and the flow target.
3. Scaling input features with `StandardScaler` and the target with `MinMaxScaler`.
4. Building sliding-window sequences of 50 time steps.
5. Splitting sequences into training and test sets in chronological order.
6. Training a GRU regression model with PyTorch.
7. Evaluating predictions using RMSE, MAE and R².
8. Exporting the trained network to ONNX.
9. Running PC-side inference with ONNX Runtime.

### Input variables

The model uses the following features in this exact order:

| # | Feature | Description |
|---:|---|---|
| 1 | `Setpoint` | Flow setpoint |
| 2 | `Error` | Control error |
| 3 | `DeltaP` | Differential pressure |
| 4 | `Valve_pos` | Control-valve position |
| 5 | `Control_signal` | Controller output signal |
| 6 | `Pipe_Diameter` | Pipe diameter |
| 7 | `Temperature` | Process temperature |

**Target:** `Flow`

### Data preparation

- Sequence/window length: **50 time steps**
- Input scaling: `StandardScaler`
- Target scaling: `MinMaxScaler`
- Train/test split: **80% / 20%**, without shuffling
- Training epochs: **100**
- Learning rate: **0.001**

The dataset in the repository is `Dataset/synthetic_flow_dataset_filtered(20K).csv`. It is synthetic, not a dataset collected from a commissioned industrial flow rig.

## Model architecture

| Parameter | Value |
|---|---:|
| Network | GRU regression |
| Input features | 7 |
| Hidden size | 32 |
| GRU layers | 1 |
| Sequence length | 50 |
| Fully connected output | 1 flow value |
| Loss | Mean Squared Error (MSE) |
| Optimizer | Adam |

The model uses the final GRU time-step output and passes it through a fully connected layer to produce one flow estimate.

## Evaluation

The project records the following evaluation results:

| Metric | Reported value |
|---|---:|
| RMSE | 1.288615 |
| MAE | 0.973815 |
| R² | 0.644286 |

These are the project's reported results on its **synthetic test data**. The training script applies an offset correction and exponential moving-average smoothing to predictions before calculating these metrics. They should therefore be treated as results from that evaluation procedure, not as unsmoothed model-only performance or evidence of real-plant accuracy.

For a stronger evaluation, future work should report raw model metrics alongside post-processing metrics, fit preprocessing scalers on training data only, and validate against measured data from a physical flow setup.

## ONNX inference

The trained network is exported to ONNX. The repository contains:

- `model/gru_model.onnx` — exported neural-network model
- `model/x_scaler.pkl` — fitted input scaler
- `model/y_scaler.pkl` — fitted target scaler
- `src/inference.py` — PC-side ONNX Runtime inference script

The inference utility currently accepts one set of seven process values and repeats that scaled sample across the 50-step input window. It is therefore a **single-sample demonstration interface**, not a live rolling time-series buffer. A real streaming implementation should maintain the latest 50 time steps in their correct temporal order.

## Proposed embedded and control-system integration

The intended edge-deployment workflow was:

1. Export the trained PyTorch model to ONNX.
2. Import the ONNX model into STM32 X-CUBE-AI for compatibility analysis and code generation.
3. Integrate the generated embedded code into an STM32CubeIDE project targeting the **NUCLEO-F446RE**.
4. Supply correctly preprocessed input sequences and execute inference on the microcontroller.
5. Connect the predicted/control value to the electrical interface and I/P converter as appropriate for the control strategy.

**Status:** This STM32 workflow is proposed project work; the model was not flashed and verified running inference on the NUCLEO-F446RE. The repository does not currently include a verified STM32 firmware implementation or on-board inference results.

The control-system study also considered the signal path from an electrical controller output to an I/P converter, pneumatic actuator pressure, valve-stem movement and a theoretical flow calculation using the valve's `Cv`. Physical integration and flow-loop testing remain future validation work.

## Repository structure

```text
AI-Based-Flow-Controller/
├── Dataset/
│   └── synthetic_flow_dataset_filtered(20K).csv
├── document/
│   └── Final_report.pdf
├── images/
│   ├── Data_visualization_of_trained_model.png
│   ├── Final_output.png
│   ├── Hardware Setup for ESP32-Based Data Collection System.png
│   └── performace_metrics.jpeg
├── model/
│   ├── gru_model.onnx
│   ├── x_scaler.pkl
│   └── y_scaler.pkl
├── requirements/
│   └── requirements.txt
├── src/
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kingkv2005-lgtm/AI-Based-Flow-Controller.git
cd AI-Based-Flow-Controller
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements/requirements.txt
```

### 4. Run the existing scripts

The source files are under `src/`:

```bash
python src/train.py
python src/inference.py
```

> **Reproducibility note:** The checked-in scripts currently refer to older `data/` and `models/` locations, while the committed dataset and model assets are stored in `Dataset/` and `model/`. Update those paths in the scripts before running training or inference from a fresh clone. The inference script also expects the scaler and ONNX files at the paths defined in its configuration. This repository layout issue is documented here rather than implying that the commands work unchanged.

## Limitations

- The dataset is synthetic; generalisation to real process conditions has not been established.
- The reported metrics include offset correction and smoothing.
- The current inference helper repeats one input sample to construct a 50-step sequence rather than consuming a true time history.
- The STM32/X-CUBE-AI deployment path was planned but not verified on the target board.
- No physical flow-loop validation was performed because the college flow laboratory was unavailable.
- The control-valve and I/P integration was analysed theoretically; it was not demonstrated as a commissioned closed-loop system.
- The training script currently fits scalers before the train/test split. A more rigorous evaluation should fit scalers on the training partition only to avoid information leakage.

## Future work

- Correct and test the repository paths so training and inference run from a clean clone.
- Implement rolling-window inference using actual time-ordered process samples.
- Re-evaluate the model with train-only preprocessing and report raw and post-processed metrics separately.
- Complete X-CUBE-AI conversion, STM32CubeIDE integration and on-board inference testing on the NUCLEO-F446RE.
- Measure inference latency, memory use and model compatibility on the target MCU.
- Validate the soft sensor using measured data from a physical flow rig.
- Test the I/P converter, actuator and control-valve integration under supervised laboratory conditions.
- Document the valve-sizing assumptions and applicable `Cv` equation for the process fluid and operating conditions.

## Contributors

- **Vishal K**


## License

No open-source license has been added to this repository yet. Until the project contributors agree on and add a license, the repository should not be treated as having an explicit open-source reuse grant.
