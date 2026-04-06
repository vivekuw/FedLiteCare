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

The project now supports lightweight local diabetes training and prediction for Hospital A, Hospital B, and Hospital C on the same laptop, plus a local aggregator backend that uses LTX over localhost for model transfer during federated rounds.

The recommended 4-terminal demo flow is documented in `RUN_GUIDE.md`.

## Local ML Layout

- Each hospital has its own config in `Hospital_X/config/client_config.yaml`.
- Each hospital keeps its own local CSV dataset in `Hospital_X/uploads/`.
- Each hospital saves its own model checkpoint in `Hospital_X/models/`.
- Each hospital appends its own training and prediction logs in `Hospital_X/logs/`.
- Each hospital receives the current global model in `Hospital_X/communication/received_global_models/` through LTX on `127.0.0.1`.
- Each hospital writes round-specific local update files to `Hospital_X/communication/local_model_updates/` before sending them back through LTX.
- Shared reusable model, preprocessing, and config utilities live under `Shared_Assets/`.
- The aggregator receives hospital checkpoints into `Aggregator_Server/received_models/` through the aggregator communication module.
- Aggregated global checkpoints are stored in `Aggregator_Server/saved_global_model_versions/`.
- The aggregator writes human-readable round entries to `Aggregator_Server/logs/round_log.log`.
- Transfer activity is logged in `Aggregator_Server/logs/transfer.log` and each hospital's own `logs/transfer.log`.
- The master sample dataset lives in `Datasets/original_dataset/diabetes_master_dataset.csv`.
- Reference split files live in `Datasets/Hospital_A_split/`, `Datasets/Hospital_B_split/`, and `Datasets/Hospital_C_split/`.

## Local Commands

- Train: `python FedLite_Project/Hospital_A/local_training/local_trainer.py`
- Train: `python FedLite_Project/Hospital_B/local_training/local_trainer.py`
- Train: `python FedLite_Project/Hospital_C/local_training/local_trainer.py`
- Hospital A federated node: `python FedLite_Project/Hospital_A/client/hospital_a_client.py federated-round`
- Hospital B federated node: `python FedLite_Project/Hospital_B/client/hospital_b_client.py federated-round`
- Hospital C federated node: `python FedLite_Project/Hospital_C/client/hospital_c_client.py federated-round`
- Predict with Hospital A default data: `python FedLite_Project/Hospital_A/prediction/predict_diabetes.py`
- Predict with Hospital B default data: `python FedLite_Project/Hospital_B/prediction/predict_diabetes.py`
- Predict with Hospital C default data: `python FedLite_Project/Hospital_C/prediction/predict_diabetes.py`
- Predict with a shared sample input: `python FedLite_Project/Hospital_A/prediction/predict_diabetes.py --input FedLite_Project/Datasets/test_input_samples/diabetes_prediction_sample.csv`
- Run one 4-terminal federated round from the aggregator: `python FedLite_Project/Aggregator_Server/server/server_main.py --mode distributed`
- Run one-click 4-terminal demo on Windows: `powershell -ExecutionPolicy Bypass -File FedLite_Project/Run_Demo_Round.ps1`
- Run one daily one-shot round for midnight scheduling: `powershell -ExecutionPolicy Bypass -File FedLite_Project/Run_Daily_Federated_Round.ps1`
- Run the old one-terminal fallback flow: `python FedLite_Project/Aggregator_Server/server/server_main.py --mode single-process`

## Desktop GUI

- Launch Hospital A GUI: `powershell -ExecutionPolicy Bypass -File FedLite_Project/Launch_Hospital_A_GUI.ps1`
- Launch Hospital B GUI: `powershell -ExecutionPolicy Bypass -File FedLite_Project/Launch_Hospital_B_GUI.ps1`
- Launch Hospital C GUI: `powershell -ExecutionPolicy Bypass -File FedLite_Project/Launch_Hospital_C_GUI.ps1`
- Hospital A packaging-ready GUI entry point: `python FedLite_Project/Hospital_A/hospital_a_app.py`

The Tkinter hospital client includes dashboard, dataset upload, local training, single-patient prediction, aggregator sync, and log/status tabs while reusing the existing backend modules.

The Predict Patient Risk tab now expects all patient fields to be filled for manual prediction, includes a built-in example input loader, and enforces safe demo ranges before prediction.

Windows EXE preparation notes for the Hospital A client are documented in `WINDOWS_EXE_PACKAGING.md`.

Daily automation notes for midnight training are documented in `Documentation/daily_automation_notes.md`.

## Demo Exports

- Each completed federated round now exports a readable summary into `Demo_Outputs/demo_logs/`.
- A machine-readable JSON artifact for each round is written into `Demo_Outputs/test_outputs/round_xxx/`.
- The aggregator console also prints a short hospital validation summary after each round.

## Practical Hospital Utilities

- Dataset validation runs before training and saves readable reports in `Hospital_X/reports/validation/`.
- Single-patient predictions create readable reports in `Hospital_X/reports/predictions/`.
- Single-patient predictions are also appended to a separate hospital CSV registry in `Hospital_X/reports/predictions/patient_prediction_registry.csv`.
- Those prediction intake rows are not used for training automatically; they must be reviewed and given a confirmed `Outcome` first.
- Hospital client commands also support manual validation with `validate-dataset`.
- Training, sync, and prediction logs remain under each hospital's `logs/` folder for demo-friendly screenshots.
