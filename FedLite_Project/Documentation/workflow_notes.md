# Workflow Notes

## Phase 2 Local ML Workflow

1. Place the diabetes training CSV inside `FedLite_Project/Hospital_A/uploads/`.
2. Update `FedLite_Project/Hospital_A/config/client_config.yaml` if you want different file names or training settings.
3. Run the local trainer to fit the PyTorch model and save the checkpoint into `FedLite_Project/Hospital_A/models/`.
4. Run the prediction pipeline on any CSV that uses the same feature columns.

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
