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

1. Start `Aggregator_Server/server/server_main.py --mode distributed`.
2. Wait until the aggregator reports that its listeners are ready.
3. Start `Hospital_A/client/hospital_a_client.py federated-round`.
4. Start `Hospital_B/client/hospital_b_client.py federated-round`.
5. Start `Hospital_C/client/hospital_c_client.py federated-round`.

The root-level PowerShell helpers in `FedLite_Project/Start_Aggregator.ps1`, `FedLite_Project/Start_Hospital_A.ps1`, `FedLite_Project/Start_Hospital_B.ps1`, and `FedLite_Project/Start_Hospital_C.ps1` wrap those same commands.
