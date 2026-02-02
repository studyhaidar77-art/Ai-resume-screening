import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

IN_CSV = Path("data/processed/jobs_clean.csv")
OUT_EMB = Path("data/processed/jd_embeddings.npy")

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing: {IN_CSV} (run clean_jobs.py first)")

    df = pd.read_csv(IN_CSV)
    texts = df["jd_clean"].fillna("").astype(str).tolist()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, emb)

    print(f"✅ Saved embeddings: {OUT_EMB}")
    print("Shape:", emb.shape)

if __name__ == "__main__":
    main()