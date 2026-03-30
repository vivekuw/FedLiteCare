# Hospital C Communication

During the local federated simulation:

- `received_global_models/` stores the global model files received from the aggregator through LTX
- `local_model_updates/` stores Hospital C's round-specific local update checkpoints before they are sent back through LTX
- `transfer_config.yaml` stores the localhost host, port, and chunk settings for Hospital C's LTX client

Only model files move through this area. Local datasets stay in `uploads/`.
