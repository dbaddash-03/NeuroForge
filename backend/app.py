from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import DEFAULT_CONFIG
from .model import architecture_summary, NeuroForgeGPT
from .tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"

app = FastAPI(title="NeuroForge LLM Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tokenizer = CharTokenizer.byte_level()


class GenerateRequest(BaseModel):
    prompt: str = Field(default="The future of AI is")
    max_new_tokens: int = Field(default=80, ge=1, le=300)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_k: int = Field(default=40, ge=0, le=256)


def get_model():
    global _model
    if _model is None:
        try:
            _model = NeuroForgeGPT(
                vocab_size=256,
                context_length=DEFAULT_CONFIG.context_length,
                d_model=DEFAULT_CONFIG.d_model,
                n_heads=DEFAULT_CONFIG.n_heads,
                n_layers=DEFAULT_CONFIG.n_layers,
                dropout=DEFAULT_CONFIG.dropout,
                bias=DEFAULT_CONFIG.bias,
            )
            _model.eval()
        except Exception:
            _model = False
    return _model


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    model = get_model()
    return {
        "status": "online",
        "engine": "PyTorch GPT-style transformer" if model else "browser demo fallback",
        "device": "cpu",
        "project": "NeuroForge",
    }


@app.get("/api/config")
def config():
    return {
        "context_length": DEFAULT_CONFIG.context_length,
        "d_model": DEFAULT_CONFIG.d_model,
        "n_heads": DEFAULT_CONFIG.n_heads,
        "n_layers": DEFAULT_CONFIG.n_layers,
        "dropout": DEFAULT_CONFIG.dropout,
        "parameters_estimate": "≈ 2.0M educational model",
    }


@app.get("/api/architecture")
def architecture():
    return architecture_summary(DEFAULT_CONFIG)


@app.post("/api/generate")
def generate(request: GenerateRequest):
    model = get_model()
    if not model:
        return {"text": demo_completion(request.prompt), "mode": "demo"}

    try:
        import torch
        ids = _tokenizer.encode(request.prompt)
        x = torch.tensor([ids[-DEFAULT_CONFIG.context_length:]], dtype=torch.long)
        out = model.generate(
            x,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
        return {"text": _tokenizer.decode(out[0].tolist()), "mode": "random-init"}
    except Exception:
        return {"text": demo_completion(request.prompt), "mode": "demo"}


def demo_completion(prompt: str):
    suffix = (
        " NeuroForge treats language modeling as a sequence prediction problem. "
        "Tokens become vectors, attention mixes contextual information, and stacked "
        "Transformer blocks learn increasingly useful representations."
    )
    return prompt + suffix


# Mount after API routes so the API remains explicit.
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC), name="static")
