# NeuroForge — LLM From Scratch Studio

NeuroForge is an original, educational implementation inspired by the learning progression of Sebastian Raschka's
**LLMs-from-scratch** project: text data → tokenization → embeddings → attention → Transformer blocks → GPT-style
language modeling → training → generation → experimentation.

Reference:
https://github.com/rasbt/LLMs-from-scratch

This repository is **not a copy of the reference repository**. The implementation, UI, naming, layout, and code in
this project were created independently while following the same broad educational progression.

## What you get

- A premium hybrid bright/dark web UI called **NeuroForge Studio**
- Interactive dashboard for the full LLM pipeline
- GPT-style Transformer implementation in PyTorch
- Character-level tokenizer
- Attention / multi-head attention / Transformer block modules
- Text generation endpoint
- Training script for a tiny educational model
- Experiment configuration
- Architecture visualizer
- Built-in demo mode when PyTorch is unavailable
- API health/model endpoints
- Responsive UI

## Project structure

```text
NeuroForge-LLM-Studio/
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
├── static/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── tests/
│   └── test_core.py
├── data/
│   └── sample.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For CPU-only use, PyTorch can be installed using the official selector at:
https://pytorch.org/get-started/locally/

### 3. Start the app

```bash
uvicorn backend.app:app --reload
```

Open:

http://127.0.0.1:8000

## Train the tiny model

```bash
python -m backend.train
```

The training script intentionally uses a small model and sample corpus so it can be used on a normal laptop.

## API

- `GET /api/health`
- `GET /api/config`
- `GET /api/architecture`
- `POST /api/generate`

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The future of AI is","max_new_tokens":80,"temperature":0.8}'
```

## Design direction

NeuroForge intentionally combines:
- deep charcoal surfaces for technical depth
- warm white/cream workspaces for readability
- electric lime and cyan for active states
- soft glass panels
- editorial typography
- command-center style metrics

The result is a bright/dark hybrid rather than a conventional all-dark AI dashboard.

## Educational scope

This is a compact learning implementation, not a production-scale foundation model. It is designed to make the internal
LLM pipeline visible and understandable.

The reference project covers a broader chapter-by-chapter implementation including text data, attention, GPT
implementation, pretraining, classification fine-tuning, instruction fine-tuning, PyTorch foundations and LoRA.
NeuroForge provides a focused interactive studio around those core concepts.
