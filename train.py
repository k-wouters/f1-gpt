"""
Trains a small GPT-style language model on Formula 1 text data.
Based on "Build a Large Language Model From Scratch" by Sebastian Raschka.
"""

import os
import torch
import tiktoken
import matplotlib.pyplot as plt

from gpt_model import GPTModel, create_dataloader_v1, generate_text_simple


# ── Model configuration ────────────────────────────────────────────────────────
# Scaled down from GPT-124M to run in reasonable time on CPU (~10M parameters)
GPT_CONFIG = {
    "vocab_size": 50257,     # GPT-2 tokenizer vocabulary
    "context_length": 128,   # Tokens of context per sample (GPT-2 uses 1024)
    "emb_dim": 256,          # Embedding size (GPT-2 uses 768)
    "n_heads": 8,            # Attention heads (GPT-2 uses 12)
    "n_layers": 6,           # Transformer layers (GPT-2 uses 12)
    "drop_rate": 0.1,        # Dropout for regularization
    "qkv_bias": False,
}

TRAIN_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 5,
    "batch_size": 8,
    "weight_decay": 0.1,
    "eval_freq": 100,        # Print loss every N steps
    "eval_iter": 20,         # Batches used to estimate loss
}

# Prompt used to generate a sample after each epoch
SAMPLE_PROMPT = "Lewis Hamilton won the"


# ── Loss helpers ───────────────────────────────────────────────────────────────

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(loader, model, device, num_batches=None):
    total_loss = 0.0
    n = min(num_batches or len(loader), len(loader))
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        total_loss += calc_loss_batch(x, y, model, device).item()
    return total_loss / n


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


# ── Text generation helper ─────────────────────────────────────────────────────

def generate_sample(model, tokenizer, device, prompt):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=60, context_size=context_size
        )
    text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(f"  Sample: {text.replace(chr(10), ' ')}")
    model.train()


# ── Training loop ──────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, optimizer, device, tokenizer):
    train_losses, val_losses, tokens_seen_list = [], [], []
    tokens_seen = 0
    step = 0

    for epoch in range(TRAIN_SETTINGS["num_epochs"]):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += x.numel()
            step += 1

            if step % TRAIN_SETTINGS["eval_freq"] == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device,
                    TRAIN_SETTINGS["eval_iter"]
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen_list.append(tokens_seen)
                print(f"Epoch {epoch+1} | Step {step:05d} | "
                      f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")

        print(f"\n--- End of epoch {epoch+1} ---")
        generate_sample(model, tokenizer, device, SAMPLE_PROMPT)
        print()

    return train_losses, val_losses, tokens_seen_list


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs_seen, train_losses, label="Train loss")
    ax1.plot(epochs_seen, val_losses, linestyle="--", label="Val loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("F1-GPT Training Loss")
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig("loss.png", dpi=150)
    print("Loss plot saved to loss.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device("cpu")  # Intel Mac — CPU only
    print(f"Using device: {device}\n")

    # Load corpus
    data_path = "data/f1_corpus.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Training data not found. Run collect_data.py first."
        )
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    print(f"Loaded corpus: {len(text_data):,} characters\n")

    # Build data loaders
    split = int(0.9 * len(text_data))
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader = create_dataloader_v1(
        text_data[:split],
        batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        drop_last=True, shuffle=True, num_workers=0,
    )
    val_loader = create_dataloader_v1(
        text_data[split:],
        batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        drop_last=False, shuffle=False, num_workers=0,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    # Build model
    model = GPTModel(GPT_CONFIG)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_SETTINGS["learning_rate"],
        weight_decay=TRAIN_SETTINGS["weight_decay"],
    )

    # Train
    train_losses, val_losses, tokens_seen = train(
        model, train_loader, val_loader, optimizer, device, tokenizer
    )

    # Save model
    torch.save(model.state_dict(), "f1_gpt.pth")
    print("Model saved to f1_gpt.pth")

    # Plot
    epochs_tensor = torch.linspace(0, TRAIN_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
