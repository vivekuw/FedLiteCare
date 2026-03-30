# FedLiteCare Windows EXE Notes

## Current Packaging-Ready Entry Point

Use `Hospital_A/hospital_a_app.py` as the Hospital A desktop client entry point.

This launcher now:

- resolves `client_config.yaml` from the Hospital A runtime folder
- resolves `transfer_config.yaml` from the Hospital A communication folder
- works in normal source mode and in a future bundled EXE layout
- keeps the existing backend modules unchanged

## Recommended Runtime Layout

For a future Windows EXE, keep the writable Hospital A runtime folder outside the bundled code and next to the executable:

```text
Hospital_A_Client/
├── Hospital_A_Client.exe
└── Hospital_A/
    ├── config/
    │   └── client_config.yaml
    ├── communication/
    │   ├── transfer_config.yaml
    │   ├── received_global_models/
    │   └── local_model_updates/
    ├── uploads/
    ├── models/
    ├── logs/
    └── reports/
```

This keeps datasets, models, logs, reports, and sync artifacts writable after packaging.

## Safe Packaging Direction

- Prefer `onedir` packaging over `onefile`.
- Keep `Hospital_A/config/client_config.yaml` external and editable.
- Keep `Hospital_A/communication/transfer_config.yaml` external and editable.
- Keep `Hospital_A/uploads/`, `Hospital_A/models/`, `Hospital_A/logs/`, and `Hospital_A/reports/` outside the EXE bundle.
- Continue using `127.0.0.1` in the transfer config for one-laptop simulation.

## Why One-Dir Is Safer

`onefile` EXEs usually unpack into a temporary folder at runtime. That can make logs, trained models, uploaded CSV files, validation reports, and prediction reports harder to manage or unsuitable for repeated demo runs.

`onedir` is simpler for this project because Hospital A already uses a folder-based runtime with editable config files and writable output directories.

## Things That Can Break In An EXE

- Any launcher that assumes fixed `__file__` parent depth.
- Any launcher that expects the full source tree to stay in the same place.
- Bundling configs inside the EXE without also providing writable external copies.
- Writing logs or trained models into a bundled read-only location.
- Packaging only `Hospital_A_Client.exe` without the sibling `Hospital_A/` runtime folder.
- Firewall prompts or blocked localhost ports during LTX demo sync.

## Current Project Notes

- Hospital A GUI is now prepared for EXE-style runtime path resolution.
- Hospital B, Hospital C, and Aggregator entry points still use more source-oriented launch patterns and should get the same cleanup before packaging those nodes.
- The active runtime folders are `uploads`, `models`, `logs`, `reports`, and `communication`.
- Legacy placeholder folders such as `uploaded_data` and `local_model_storage` are not part of the active desktop client flow.

## Suggested Later PyInstaller Direction

Example direction for later, not to run yet:

```powershell
pyinstaller --noconsole --onedir --name Hospital_A_Client FedLite_Project\Hospital_A\hospital_a_app.py
```

After building, place the `Hospital_A/` runtime folder next to the generated EXE so the packaged client can still find its config, models, logs, and transfer settings.
