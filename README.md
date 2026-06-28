# FedLiteCare

FedLiteCare is a lightweight federated learning project for diabetes prediction in a constrained environment. It simulates one aggregator server and three hospital nodes on a single laptop. Each hospital keeps its own local CSV data, trains locally, and shares only model updates with the server using LTX-based transfer. The server combines the updates with federated averaging and saves a versioned global model for the next round.

## Key Features

- One aggregator server and three hospital nodes
- Local diabetes model training with PyTorch
- LTX-based chunked transfer for model files
- Federated averaging at the aggregator
- Tkinter desktop dashboards for server and nodes
- Local prediction using the latest global model
- Dataset validation, logs, and prediction reports
- No cloud deployment and no web application

## Technology Stack

- Python
- PyTorch
- Tkinter
- LTX file transfer
- CSV-based datasets
- PowerShell launch scripts

## Project Structure

```text
FedLite_Project/
├── Aggregator_Server/     # Server-side aggregation, round management, and dashboard
├── Hospital_A/            # Hospital A node, local training, prediction, and communication
├── Hospital_B/            # Hospital B node, local training, prediction, and communication
├── Hospital_C/            # Hospital C node, local training, prediction, and communication
├── Shared_Assets/         # Common preprocessing, model, login, and transfer utilities
├── Datasets/              # Master dataset, hospital splits, and test input samples
├── Demo_Outputs/          # Demo logs, screenshots, and exported test outputs
├── Launch_*.ps1           # GUI launch scripts for server and hospital nodes
├── Start_*.ps1            # Terminal start scripts for server and hospital nodes
├── Run_Demo_Round.ps1     # One-click demo round launcher
├── Run_Daily_Federated_Round.ps1
├── Register_Daily_FedLiteCare_Task.ps1
└── README.md
```
## Iamges

📁 **View all screenshots:** [https://github.com/USERNAME/REPOSITORY/tree/main/images](https://github.com/vivekuw/FedLiteCare/tree/master/FedLite_Project/images)

## Folder Overview

### `Aggregator_Server`
Contains the aggregator logic, aggregation engine, versioned global model storage, server logs, received hospital updates, and the server dashboard GUI.

### `Hospital_A`, `Hospital_B`, `Hospital_C`
Each hospital folder is an independent node with its own dataset, local model, communication files, logs, reports, and Tkinter client GUI.

### `Shared_Assets`
Contains reusable code shared by all nodes and the server, including preprocessing helpers, model helpers, transfer helpers, login logic, and GUI support functions.

### `Datasets`
Stores the master diabetes dataset, hospital-specific dataset splits, and test input CSV files.

### `Demo_Outputs`
Stores generated outputs for presentation and testing, such as logs, screenshots, summary files, and exported round artifacts.

## How To Run

### Option 1: Open the GUIs manually

Start the aggregator GUI:

```powershell
cd "C:\Users\Vivek wadher\OneDrive\Desktop\Documents\New project\FedLite_Project"
powershell -ExecutionPolicy Bypass -File .\Launch_Aggregator_GUI.ps1
```

Start the hospital GUIs in separate terminals:

```powershell
powershell -ExecutionPolicy Bypass -File .\Launch_Hospital_A_GUI.ps1
powershell -ExecutionPolicy Bypass -File .\Launch_Hospital_B_GUI.ps1
powershell -ExecutionPolicy Bypass -File .\Launch_Hospital_C_GUI.ps1
```

### Option 2: Run the terminal demo

Run a full demo round automatically:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run_Demo_Round.ps1
```

### Option 3: Run the nightly round

Run the single-process daily federated round:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run_Daily_Federated_Round.ps1
```

## Data Flow

1. Each hospital loads its local diabetes CSV dataset.
2. The node validates and preprocesses the data.
3. The node trains the shared PyTorch model locally.
4. The updated model is sent to the aggregator using LTX.
5. The aggregator performs federated averaging.
6. The new global model version is saved.
7. The refreshed global model is redistributed to the nodes.
8. Nodes can use the latest global model for prediction.

## Notes

- Raw patient data stays local to each hospital node.
- Only model updates are transferred between nodes and the server.
- The project is designed for a one-laptop simulation and constrained hardware.
- The current implementation uses local file-based storage for datasets, logs, reports, and model checkpoints.

## Purpose

This project demonstrates how lightweight federated learning can be used for healthcare prediction without centralizing raw patient data. It is intended for academic demonstration, testing, and future extension.

