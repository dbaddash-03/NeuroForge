import torch
from backend.model import NeuroForgeGPT
from backend.tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    tok = CharTokenizer.byte_level()
    text = "hello neuroforge"
    assert tok.decode(tok.encode(text)) == text


def test_model_shape():
    model = NeuroForgeGPT(vocab_size=256, context_length=32, d_model=48, n_heads=4, n_layers=2)
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 256)
    assert loss.ndim == 0
