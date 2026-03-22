# F1-GPT: A Formula 1 Language Model Built From Scratch

A GPT-style language model trained entirely on Formula 1 content — drivers, teams, circuits, and championship history — built from the ground up using PyTorch.

This project was developed with the assistance of **Claude Code** (Anthropic's AI coding CLI), which I used throughout to accelerate development, debug issues, and make architectural decisions. The workflow reflects how I approach modern software projects: leveraging AI tools deliberately and critically, not just as an autocomplete, but as a thinking partner.

---

## Why I built this

I'm fascinated by the intersection of two things I care deeply about: **Formula 1** and **machine learning**. F1 is one of the most data-rich sports in the world — every race generates terabytes of telemetry, and the strategies that win championships are increasingly driven by simulation and predictive modeling. This project is my way of exploring that intersection at the language level: what does an LLM learn when its entire world is F1?

It's also my hands-on answer to the question *"how do LLMs actually work?"* — not by reading about attention mechanisms, but by implementing and training one myself.

---

## What I built

- A **GPT-style transformer** (~30M parameters) implemented in PyTorch from scratch
- A **custom training corpus** of ~1.9 million characters scraped from 42 Formula 1 Wikipedia articles covering drivers, teams, circuits, and seasons
- A complete **training pipeline** with cross-entropy loss, AdamW optimization, train/val split, and loss curve plotting
- Trained on **CPU** (Intel Mac) — no GPU required

The model architecture:

| Hyperparameter | Value |
|---|---|
| Parameters | ~30M |
| Layers | 6 transformer blocks |
| Attention heads | 8 |
| Embedding dimension | 256 |
| Context length | 128 tokens |
| Vocabulary | GPT-2 tokenizer (50,257 tokens) |

---

## How to run it

**1. Clone the repo and set up the environment**
```bash
git clone https://github.com/k-wouters/f1-gpt.git
cd f1-gpt
python3.11 -m venv venv
source venv/bin/activate
pip install torch tiktoken matplotlib requests wikipedia-api "numpy<2"
```

**2. Collect the training data**
```bash
python collect_data.py
```
This scrapes ~42 F1 Wikipedia articles and saves them to `data/f1_corpus.txt`.

**3. Train the model**
```bash
python train.py
```
Trains for 5 epochs on CPU (~30–60 min on a modern Intel Mac). Saves the model to `f1_gpt.pth` and a loss plot to `loss.png`.

---

## How I used Claude Code

This project was built interactively with [Claude Code](https://claude.ai/claude-code), Anthropic's CLI tool for AI-assisted development. Specifically, I used it to:

- **Design the project architecture** — deciding on model size, dataset scope, and training configuration for CPU-constrained hardware
- **Debug environment issues** — resolving Python 3.13/PyTorch incompatibility and NumPy version conflicts
- **Write and review code** — the training loop, data collection script, and model configuration
- **Make informed trade-offs** — e.g., reducing model size from 124M to 30M parameters to make CPU training feasible without sacrificing meaningful learning

I think being able to work effectively with AI coding tools is an important skill for any engineer in 2026 — this project is a demonstration of that alongside the underlying ML knowledge.

---

## Built on

- [Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka — the GPT model architecture in `gpt_model.py` is adapted from this book's codebase
- [PyTorch](https://pytorch.org/)
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's BPE tokenizer
- [Wikipedia-API](https://github.com/martin-majlis/Wikipedia-API)
