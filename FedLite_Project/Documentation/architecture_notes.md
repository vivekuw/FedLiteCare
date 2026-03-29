# Architecture Notes

## Current Local Design

- `Shared_Assets/` contains the reusable preprocessing, model, configuration, logging, training, and prediction helpers.
- Each hospital keeps its own config, uploads, models, and logs under its own folder.
- The same lightweight PyTorch diabetes classifier is reused for Hospital A, Hospital B, and Hospital C.
- No federated aggregation, networking, GUI, or deployment packaging is active yet.
