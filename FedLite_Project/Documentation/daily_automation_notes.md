# Daily Automation Notes

## Goal

Run one full FedLiteCare training-and-aggregation cycle automatically once per day, such as at `12:00 AM`.

## Recommended Nightly Mode

Use the root script:

`FedLite_Project/Run_Daily_Federated_Round.ps1`

This script runs:

`python .\Aggregator_Server\server\server_main.py --mode single-process`

That mode is best for nightly automation because:

- it does not need four open terminals
- it still simulates hospital training and server aggregation on one laptop
- it exits automatically after the round finishes
- it still writes the normal round logs and demo summary exports

## Midnight Scheduling

You can register a Windows Scheduled Task with:

`FedLite_Project/Register_Daily_FedLiteCare_Task.ps1`

Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\Register_Daily_FedLiteCare_Task.ps1
```

That creates a daily task at `00:00`.

Custom time example:

```powershell
powershell -ExecutionPolicy Bypass -File .\Register_Daily_FedLiteCare_Task.ps1 -StartTime 00:30
```

## Manual Hospital Action

Inside the hospital GUI, `Train Local Model` remains a local-only action.

A separate manual action is recommended for:

- training locally
- receiving the current global model
- sending the update back to the server

That full server-connected action should use the aggregator sync path, not the local-only train path.

## Important Data Rule

Nightly automation should use confirmed hospital training CSV data only. The patient prediction registry CSV is separate and should not be used for training until a real `Outcome` is confirmed.
