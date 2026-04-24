r"""Placeholder"""

import logging

logging.getLogger("waldo")


import os

import numpy as np
import torch
from cuml.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoModel

# --- SETTINGS ---
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"  # Example VLM/Embedding model
NUM_SAMPLES = 100000
EMBED_DIM = 4096
NUM_LAYERS = 640
BATCH_SIZE = 32
STORAGE_DIR = "./ssd_embeddings"
os.makedirs(STORAGE_DIR, exist_ok=True)

# 1. INITIALIZE MODEL
print("Loading model to H200...")
# H200 has 141GB, so we keep the model on GPU for maximum speed
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).cuda().eval()
# Note: Ensure your model's forward pass is configured to return all hidden states
# e.g., output_hidden_states=True

# 2. PRE-ALLOCATE SSD FILES (Memory Mapping)
# This creates 640 "virtual" arrays on your SSD.
mmaps = {}
for i in range(NUM_LAYERS):
    filename = os.path.join(STORAGE_DIR, f"layer_{i}.npy")
    # mode='w+' initializes a zero-filled file of the exact final size
    mmaps[i] = np.memmap(filename, dtype="float32", mode="w+", shape=(NUM_SAMPLES, EMBED_DIM))


def extract_and_save(dataloader):
    """Loop over dataloader and stream each batch to SSD."""
    current_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference & SSD Streaming"):
            inputs = {k: v.cuda() for k, v in batch.items()}

            # Forward pass: request all hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # outputs.hidden_states is usually a tuple of length 640
            all_layers = outputs.hidden_states

            b_size = inputs["input_ids"].size(0)
            end_idx = current_idx + b_size

            for i, layer_tensor in enumerate(all_layers):
                # layer_tensor shape: [batch, seq_len, dim] -> we take mean or [CLS]
                # Assuming you want the [CLS] token (index 0)
                embeddings = layer_tensor[:, 0, :].cpu().numpy()

                # Write directly to the SSD-backed array
                mmaps[i][current_idx:end_idx] = embeddings

            current_idx = end_idx

            # Periodically force disk write to prevent RAM caching
            if current_idx % 1000 == 0:
                for m in mmaps.values():
                    m.flush()

    print("Finished streaming all layers to SSD.")


# 3. RUN LOGISTIC REGRESSION (Using cuML on H200)
def run_classification(y_labels):
    """Reads one layer at a time from SSD and trains on H200."""
    y_gpu = torch.tensor(y_labels).cuda()
    layer_results = {}

    for i in range(NUM_LAYERS):
        print(f"Training Layer {i}/640...")

        # Load from SSD (This is instant because it's only a pointer)
        filename = os.path.join(STORAGE_DIR, f"layer_{i}.npy")
        X_mmap = np.memmap(filename, dtype="float32", mode="r", shape=(NUM_SAMPLES, EMBED_DIM))

        # Move ONLY this layer to the H200 (1.6 GB)
        X_gpu = torch.from_numpy(X_mmap).cuda()

        # cuML Logistic Regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_gpu, y_gpu)

        # Save weights back to CPU/Dictionary
        layer_results[i] = clf.coef_.get()  # .get() moves to CPU numpy

        # CRITICAL: Clear VRAM for the next layer
        del X_gpu, clf
        torch.cuda.empty_cache()

    return layer_results


# --- EXECUTION ---
# extract_and_save(my_dataloader)
# results = run_classification(my_labels)
