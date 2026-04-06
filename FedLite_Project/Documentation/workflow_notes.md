# Workflow Notes

## Local Multi-Hospital Workflow

1. Each hospital has its own config file inside `FedLite_Project/Hospital_X/config/client_config.yaml`.
2. Place each hospital's training CSV inside its own `FedLite_Project/Hospital_X/uploads/` folder.
3. Run the local trainer for Hospital A, Hospital B, or Hospital C to fit that hospital's standalone model.
4. Each trained checkpoint is written into that hospital's own `models/` folder.
5. Each training and prediction run appends to that hospital's own `logs/` folder.

## Dataset Layout

- `FedLite_Project/Datasets/original_dataset/diabetes_master_dataset.csv` stores the small master sample.
- `FedLite_Project/Datasets/Hospital_A_split/hospital_a_diabetes.csv` is the reference split for Hospital A.
- `FedLite_Project/Datasets/Hospital_B_split/hospital_b_diabetes.csv` is the reference split for Hospital B.
- `FedLite_Project/Datasets/Hospital_C_split/hospital_c_diabetes.csv` is the reference split for Hospital C.

The same split files are mirrored into each hospital's `uploads/` folder so each simulated hospital can run independently.

## Expected CSV Format

Training CSV files should include these columns:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome`

Prediction CSV files can omit the `Outcome` column. If it is included, the prediction script will also report accuracy.

## Full Local Federated Workflow

1. The aggregator creates or loads the current global model.
2. The aggregator sends the current global model to each hospital through LTX on `127.0.0.1`.
3. Each hospital receives that global model inside its own `communication/received_global_models/` folder and trains locally on its own CSV dataset.
4. Each hospital saves its own standalone local model in `models/`.
5. Each hospital also saves a round-specific update checkpoint in `communication/local_model_updates/`.
6. Each hospital sends its round-specific update checkpoint back to the aggregator through LTX.
7. The aggregator stores those update files inside `Aggregator_Server/received_models/round_xxx/`.
8. FedAvg combines the three local updates into a new global checkpoint.
9. The aggregator saves the new global checkpoint as both a round-specific version and the latest global model.
10. The aggregator appends a simple human-readable round entry to `Aggregator_Server/logs/round_log.log`.
11. Transfer events are appended to `Aggregator_Server/logs/transfer.log` and the matching hospital `logs/transfer.log`.

## Four-Terminal Startup

For a one-click Windows demo, run `FedLite_Project/Run_Demo_Round.ps1`.

1. Start `Aggregator_Server/server/server_main.py --mode distributed`.
2. Wait until the aggregator reports that its listeners are ready.
3. Start `Hospital_A/client/hospital_a_client.py federated-round`.
4. Start `Hospital_B/client/hospital_b_client.py federated-round`.
5. Start `Hospital_C/client/hospital_c_client.py federated-round`.

The root-level PowerShell helpers in `FedLite_Project/Start_Aggregator.ps1`, `FedLite_Project/Start_Hospital_A.ps1`, `FedLite_Project/Start_Hospital_B.ps1`, and `FedLite_Project/Start_Hospital_C.ps1` wrap those same commands.

After each round, the aggregator also exports a demo-friendly summary into `FedLite_Project/Demo_Outputs/demo_logs/` and a JSON artifact into `FedLite_Project/Demo_Outputs/test_outputs/round_xxx/`.

## Daily Automation

For a once-per-day run such as `12:00 AM`, use `FedLite_Project/Run_Daily_Federated_Round.ps1`.

That script uses the one-process server flow so it can:

1. load the latest global model
2. simulate hospital training on the same laptop
3. aggregate the updates
4. save the new global model
5. exit automatically

Use `FedLite_Project/Register_Daily_FedLiteCare_Task.ps1` if you want to register a Windows Scheduled Task for daily execution.

## Desktop GUI

The hospital desktop client can be launched with:

- `FedLite_Project/Launch_Hospital_A_GUI.ps1`
- `FedLite_Project/Launch_Hospital_B_GUI.ps1`
- `FedLite_Project/Launch_Hospital_C_GUI.ps1`

The GUI is a thin Tkinter layer over the existing backend. It reuses local training, prediction, and sync modules rather than replacing them.

For manual patient prediction in the GUI, all patient feature fields should be entered. The prediction tab also includes a built-in example input set and a visible allowed-range guide for quicker demos and safer input entry.

## Validation And Reports

1. Before any local training run, the dataset is validated for required columns, missing values, and obvious bad rows.
2. A readable validation report is saved in `Hospital_X/reports/validation/`.
3. Each single-patient prediction generates a readable report in `Hospital_X/reports/predictions/`.
4. Each single-patient prediction is also appended to `Hospital_X/reports/predictions/patient_prediction_registry.csv` for hospital-side tracking.
5. Those prediction intake rows should stay separate from training data until a true `Outcome` is confirmed.
6. Training, prediction, and sync activity continue to append to the hospital log files in `Hospital_X/logs/`.
