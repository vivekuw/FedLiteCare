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
