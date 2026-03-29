# Hospital B Communication

During the local federated simulation:

- `received_global_models/` stores the global model copies sent from the aggregator
- `local_model_updates/` stores Hospital B's round-specific local update checkpoints

Only model files move through this area. Local datasets stay in `uploads/`.
