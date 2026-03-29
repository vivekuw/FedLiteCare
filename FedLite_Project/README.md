# FedLiteCare

FedLiteCare is a one-laptop simulation of a federated learning system for diabetes prediction using Python and PyTorch.

## Folder Overview

- `Aggregator_Server/` - Central server components for aggregation, global model coordination, configuration, logs, and saved model artifacts.
- `Hospital_A/` - Local client workspace for Hospital A, including training, prediction, storage, communication, and logs.
- `Hospital_B/` - Local client workspace for Hospital B, including training, prediction, storage, communication, and logs.
- `Hospital_C/` - Local client workspace for Hospital C, including training, prediction, storage, communication, and logs.
- `Shared_Assets/` - Reusable helpers for utilities, shared model functions, and preprocessing support.
- `Datasets/` - Master dataset samples, hospital-specific split CSV files, and test input samples.
- `Documentation/` - Notes for architecture, workflow, and future deployment planning.
- `Demo_Outputs/` - Reserved space for screenshots, demo logs, and test output artifacts.

## Current Status

The project now supports lightweight local diabetes training and prediction for Hospital A, Hospital B, and Hospital C on the same laptop, plus a local aggregator backend for federated rounds.

## Local ML Layout

- Each hospital has its own config in `Hospital_X/config/client_config.yaml`.
- Each hospital keeps its own local CSV dataset in `Hospital_X/uploads/`.
- Each hospital saves its own model checkpoint in `Hospital_X/models/`.
- Each hospital appends its own training and prediction logs in `Hospital_X/logs/`.
- Each hospital receives the current global model in `Hospital_X/communication/received_global_models/`.
- Each hospital writes round-specific local update files to `Hospital_X/communication/local_model_updates/`.
- Shared reusable model, preprocessing, and config utilities live under `Shared_Assets/`.
- The aggregator copies hospital checkpoints into `Aggregator_Server/received_models/`.
- Aggregated global checkpoints are stored in `Aggregator_Server/saved_global_model_versions/`.
- The aggregator writes human-readable round entries to `Aggregator_Server/logs/round_log.log`.
- The master sample dataset lives in `Datasets/original_dataset/diabetes_master_dataset.csv`.
- Reference split files live in `Datasets/Hospital_A_split/`, `Datasets/Hospital_B_split/`, and `Datasets/Hospital_C_split/`.

## Local Commands

- Train: `python FedLite_Project/Hospital_A/local_training/local_trainer.py`
- Train: `python FedLite_Project/Hospital_B/local_training/local_trainer.py`
- Train: `python FedLite_Project/Hospital_C/local_training/local_trainer.py`
- Predict with Hospital A default data: `python FedLite_Project/Hospital_A/prediction/predict_diabetes.py`
- Predict with Hospital B default data: `python FedLite_Project/Hospital_B/prediction/predict_diabetes.py`
- Predict with Hospital C default data: `python FedLite_Project/Hospital_C/prediction/predict_diabetes.py`
- Predict with a shared sample input: `python FedLite_Project/Hospital_A/prediction/predict_diabetes.py --input FedLite_Project/Datasets/test_input_samples/diabetes_prediction_sample.csv`
- Run one full local federated round: `python FedLite_Project/Aggregator_Server/server/server_main.py`
