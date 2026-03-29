# FedLiteCare

FedLiteCare is a one-laptop simulation of a federated learning system for diabetes prediction using Python and PyTorch.

## Folder Overview

- `Aggregator_Server/` - Central server components for aggregation, global model coordination, configuration, logs, and saved model artifacts.
- `Hospital_A/` - Local client workspace for Hospital A, including training, prediction, storage, communication, and logs.
- `Hospital_B/` - Local client workspace for Hospital B, including training, prediction, storage, communication, and logs.
- `Hospital_C/` - Local client workspace for Hospital C, including training, prediction, storage, communication, and logs.
- `Shared_Assets/` - Reusable helpers for utilities, shared model functions, and preprocessing support.
- `Datasets/` - Placeholders for the original dataset, hospital-specific splits, and test input samples.
- `Documentation/` - Notes for architecture, workflow, and future deployment planning.
- `Demo_Outputs/` - Reserved space for screenshots, demo logs, and test output artifacts.

## Current Status

Phase 2 adds a lightweight local machine learning pipeline for Hospital A only.

## Phase 2 Local ML

- Training CSVs should be placed in `Hospital_A/uploads/`.
- The default sample training file is `Hospital_A/uploads/diabetes_sample.csv`.
- Trained model checkpoints are saved to `Hospital_A/models/diabetes_classifier.pt`.
- Main path and training settings live in `Hospital_A/config/client_config.yaml`.
- Shared reusable model, preprocessing, and config utilities live under `Shared_Assets/`.

## Local Commands

- Train: `python FedLite_Project/Hospital_A/local_training/local_trainer.py`
- Predict with default Hospital A sample: `python FedLite_Project/Hospital_A/prediction/predict_diabetes.py`
- Predict with the test sample: `python FedLite_Project/Hospital_A/prediction/predict_diabetes.py --input FedLite_Project/Datasets/test_input_samples/diabetes_prediction_sample.csv`
