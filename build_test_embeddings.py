import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

PREP_DIR = "prepared"
MODEL_NAME = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def main():
    records = load_jsonl(os.path.join(PREP_DIR, "chunk_records.jsonl"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    texts = [r["text"] for r in records]
    embeddings = []

    bs = 64
    for i in tqdm(range(0, len(texts), bs)):
        batch = texts[i:i+bs]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, dim=-1)
        embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.save(os.path.join(PREP_DIR, "chunk_text_embeddings.npy"), embeddings)
    print("Saved text embeddings:", embeddings.shape)


if __name__ == "__main__":
    main()