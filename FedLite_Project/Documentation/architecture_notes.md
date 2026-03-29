# Architecture Notes

## Current Local Design

- `Shared_Assets/` contains the reusable preprocessing, model, configuration, logging, training, and prediction helpers.
- Each hospital keeps its own config, uploads, models, and logs under its own folder.
- The same lightweight PyTorch diabetes classifier is reused for Hospital A, Hospital B, and Hospital C.
- `Aggregator_Server/` now orchestrates local federated rounds by copying hospital checkpoints, averaging weights, saving global model versions, and tracking round history.
- No federated aggregation, networking, GUI, or deployment packaging is active yet.
