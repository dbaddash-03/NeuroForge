from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from .config import DEFAULT_CONFIG
from .model import NeuroForgeGPT
from .tokenizer import CharTokenizer

DATA = Path(__file__).resolve().parents[1] / "data" / "sample.txt"


class TextDataset(Dataset):
    def __init__(self, ids, context_length):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.ids) - self.context_length - 1)

    def __getitem__(self, i):
        x = self.ids[i:i+self.context_length]
        y = self.ids[i+1:i+self.context_length+1]
        return x, y


def train(steps=120, batch_size=8, lr=3e-4):
    text = DATA.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.byte_level()
    ids = tokenizer.encode(text)

    dataset = TextDataset(ids, DEFAULT_CONFIG.context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = NeuroForgeGPT(
        vocab_size=256,
        context_length=DEFAULT_CONFIG.context_length,
        d_model=DEFAULT_CONFIG.d_model,
        n_heads=DEFAULT_CONFIG.n_heads,
        n_layers=DEFAULT_CONFIG.n_layers,
        dropout=DEFAULT_CONFIG.dropout,
        bias=DEFAULT_CONFIG.bias,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    iterator = iter(loader)

    for step in range(steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step:04d} loss={loss.item():.4f}")

    out = Path(__file__).resolve().parents[1] / "neuroforge_tiny.pt"
    torch.save(model.state_dict(), out)
    print(f"saved: {out}")


if __name__ == "__main__":
    train()
