from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskConfig:
    modulus: int = 113
    train_fraction: float = 0.30
    seq_len: int = 3
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    max_steps: int = 30_000
    log_every: int = 100
    checkpoint_every: int = 250
    stabilization_steps: int = 1_500
    threshold_accuracy: float = 0.95
    threshold_patience_logs: int = 3

    @property
    def vocab_size(self) -> int:
        return self.modulus + 1

    @property
    def equals_token(self) -> int:
        return self.modulus

    def to_dict(self) -> dict:
        return asdict(self)
