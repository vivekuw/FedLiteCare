# FedLiteCare Run Guide

## Goal

Run one full FedLiteCare federated round on one laptop with four terminals:

- Terminal 1: Aggregator_Server
- Terminal 2: Hospital_A
- Terminal 3: Hospital_B
- Terminal 4: Hospital_C

## Before You Start

1. Install Python dependencies, including PyTorch.
2. Keep the default localhost settings in the transfer config files unless you need different ports.
3. Make sure each hospital CSV is still inside its own `uploads/` folder.

## Safe Localhost Defaults

The project is already configured for localhost testing:

- Aggregator host: `127.0.0.1`
- Hospital receive ports: `9101`, `9102`, `9103`
- Aggregator update receive ports: `9201`, `9202`, `9203`
- Transfer retries and long timeouts are enabled for demo reliability

## Startup Order

For the easiest Windows demo, use:

```powershell
.\Run_Demo_Round.ps1
```

That script opens four PowerShell windows automatically, starts the aggregator first, then starts Hospital A, Hospital B, and Hospital C, and uses a short automatic startup delay so the round can begin without manual `Enter` confirmation.

## Daily Midnight Run

For an automatic once-per-day run, use:

```powershell
.\Run_Daily_Federated_Round.ps1
```

That script runs a full one-laptop federated round in `single-process` mode and exits automatically, which is more suitable for `12:00 AM` automation than the 4-terminal demo.

## Manual Startup Order

1. Open Terminal 1 in `FedLite_Project/` and run:

```powershell
.\Start_Aggregator.ps1
```

2. Wait until the aggregator says its listeners are ready.

3. Leave the aggregator terminal open. It now pauses and waits for you to confirm that the hospitals are ready.

4. Open Terminal 2 in `FedLite_Project/` and run:

```powershell
.\Start_Hospital_A.ps1
```

5. Open Terminal 3 in `FedLite_Project/` and run:

```powershell
.\Start_Hospital_B.ps1
```

6. Open Terminal 4 in `FedLite_Project/` and run:

```powershell
.\Start_Hospital_C.ps1
```

7. Go back to Terminal 1 and press `Enter` to let the aggregator distribute the current global model.

## What Happens

1. The aggregator creates or loads the current global model.
2. The aggregator opens update listeners and waits.
3. Each hospital starts, waits for the current global model, and receives it through LTX.
4. Each hospital trains locally on its own diabetes CSV.
5. Each hospital sends its local update back through LTX.
6. The aggregator collects all three updates, runs FedAvg, and saves the new global model version.

## Alternate Python Entry Points

If you do not want to use the PowerShell scripts, these are the direct commands:

```powershell
python .\Aggregator_Server\server\server_main.py --mode distributed
python .\Hospital_A\client\hospital_a_client.py federated-round
python .\Hospital_B\client\hospital_b_client.py federated-round
python .\Hospital_C\client\hospital_c_client.py federated-round
```

## Logs

- Aggregator runtime flow: `Aggregator_Server/logs/aggregator_runtime.log`
- Aggregator aggregation summary: `Aggregator_Server/logs/aggregator.log`
- Aggregator round summary: `Aggregator_Server/logs/round_log.log`
- Aggregator transfer activity: `Aggregator_Server/logs/transfer.log`
- Hospital runtime flow: `Hospital_X/logs/federated_client.log`
- Hospital training metrics: `Hospital_X/logs/training.log`
- Hospital transfer activity: `Hospital_X/logs/transfer.log`
- Demo summary export: `Demo_Outputs/demo_logs/round_xxx_demo_summary.txt`
- Demo JSON export: `Demo_Outputs/test_outputs/round_xxx/demo_summary.json`

## Important Notes

- Raw patient CSV files stay inside each hospital folder.
- Only model-related files move between the aggregator and hospitals.
- Run the four startup commands again to simulate the next federated round.
