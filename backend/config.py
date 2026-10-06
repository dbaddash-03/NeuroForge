from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dropout: float = 0.1
    bias: bool = False

DEFAULT_CONFIG = ModelConfig()
