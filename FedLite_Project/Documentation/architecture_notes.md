# Architecture Notes

## Current Local Design

- `Shared_Assets/` contains the reusable preprocessing, model, configuration, logging, training, and prediction helpers.
- `Shared_Assets/common_utilities/ltx_core.py` provides the lightweight chunked localhost transfer backbone used for model movement.
- Each hospital keeps its own config, uploads, models, and logs under its own folder.
- The same lightweight PyTorch diabetes classifier is reused for Hospital A, Hospital B, and Hospital C.
- Each hospital communication folder now acts as the isolated handoff area for received global models and outgoing local update checkpoints.
- LTX communication is isolated inside `Aggregator_Server/communication/` and `Hospital_X/communication/` so training and aggregation stay modular.
- `Aggregator_Server/` now orchestrates local federated rounds by creating or loading a global model, sending it to hospitals through LTX, averaging hospital updates, saving global model versions, and tracking round history.
- Raw patient CSV data stays inside each hospital's `uploads/` folder. Only model-related files move between nodes.
- No GUI, sockets beyond localhost simulation, cloud deployment, or EXE packaging is active yet.
