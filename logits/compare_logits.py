import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import os

# === Config ===
model_name = "EleutherAI/pythia-70m"
input_text = "what is a rose tell me about it?"
save_dir = "pythia_debug_dumps"
os.makedirs(save_dir, exist_ok=True)

# === Load model ===
print(f"Loading {model_name}...")
model = HookedTransformer.from_pretrained(model_name, device="cpu")

# === Tokenize and run model ===
tokens = model.to_tokens(input_text)
logits, cache = model.run_with_cache(tokens)

logits = logits[0].detach().cpu().numpy()  # shape: (seq_len, vocab_size)

# === Save logits ===
logits_path = os.path.join(save_dir, "logits_transformerlens.csv")
pd.DataFrame(logits).to_csv(logits_path, index=False, header=False)
print(f"Saved logits to {logits_path}")

# === Save per-layer MLP activations ===
for l in range(model.cfg.n_layers):
    mlp_acts = cache[f'blocks.{l}.mlp.hook_post']  # shape: (1, seq_len, d_mlp)
    mlp_acts = mlp_acts[0].detach().cpu().numpy()
    df = pd.DataFrame(mlp_acts)
    df.to_csv(os.path.join(save_dir, f"layer_{l}_mlp.csv"), index=False, header=False)

print("✅ Saved all TransformerLens logits and MLP activations.")
