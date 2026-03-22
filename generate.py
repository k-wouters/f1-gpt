"""
Generate Formula 1 text using the trained F1-GPT model.
Usage: python generate.py "Your prompt here"
"""

import sys
import torch
import tiktoken
from gpt_model import GPTModel, generate_text_simple

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def generate(prompt, max_new_tokens=100, temperature=1.0):
    device = torch.device("cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(torch.load("f1_gpt.pth", weights_only=True))
    model.eval()
    model.to(device)

    encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=GPT_CONFIG["context_length"],
        )

    return tokenizer.decode(token_ids.squeeze(0).tolist())


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The Monaco Grand Prix"

    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    print(generate(prompt))
    print()
