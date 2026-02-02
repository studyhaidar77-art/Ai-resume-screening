import pandas as pd
import re
from pathlib import Path

RAW_PATH = Path("data/raw/jobs_dataset.csv")
SAVE_PATH = Path("data/processed/jobs_clean.csv")

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text)
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Required columns
    required = ["company", "location", "positionName", "description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    df["jd_clean"] = df["description"].apply(clean_text)

    before = len(df)
    df = df[df["jd_clean"].str.len() >= 80].copy()
    after = len(df)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)

    print(f"Loaded rows: {before}")
    print(f"Kept rows after cleaning: {after}")
    print(f"✅ Saved: {SAVE_PATH}")

if __name__ == "__main__":
    main()
