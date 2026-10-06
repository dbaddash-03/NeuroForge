class CharTokenizer:
    """Small byte-level tokenizer for an educational GPT."""

    def __init__(self, text: str = ""):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text: str):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "�") for i in ids)

    @classmethod
    def byte_level(cls):
        obj = cls()
        obj.stoi = {chr(i): i for i in range(256)}
        obj.itos = {i: chr(i) for i in range(256)}
        return obj
