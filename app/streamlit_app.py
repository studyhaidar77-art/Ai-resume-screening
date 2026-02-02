import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

SKILLS = [
    "python","sql","excel","power bi","tableau",
    "machine learning","deep learning","nlp","llm",
    "pandas","numpy","scikit-learn","tensorflow","pytorch",
    "streamlit","fastapi","flask",
    "docker","kubernetes","aws","gcp","azure",
    "spark","hadoop","airflow",
    "git","linux",
    "statistics","probability","regression","classification"
]

def extract_skills(text: str):
    t = (text or "").lower()
    found = []
    for s in SKILLS:
        if re.search(r"\b" + re.escape(s) + r"\b", t):
            found.append(s)
    return sorted(set(found))


JOBS_CSV = Path("data/processed/jobs_clean.csv")
EMB_PATH = Path("data/processed/jd_embeddings.npy")

@st.cache_resource
def load_model():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model

model = load_model()



@st.cache_data
def load_data():
    jobs = pd.read_csv(JOBS_CSV)
    emb = np.load(EMB_PATH)
    return jobs, emb

def main():
    st.set_page_config(page_title="Resume Screening AI", layout="wide")
    st.title("Resume Screening AI — Resume → Best Jobs")

    if not JOBS_CSV.exists() or not EMB_PATH.exists():
        st.error(
            "Missing processed files. Run:\n"
            "python src\\preprocessing\\clean_jobs.py\n"
            "python src\\models\\embed_jobs.py"
        )
        st.stop()

    # Load resources
    jobs, jd_emb = load_data()
    model = load_model()

    # --- Resume input (PDF or text) ---
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="resume_pdf")

    if uploaded_file is not None:
        from pypdf import PdfReader
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        resume_text = " ".join(pages)
        st.success("✅ Resume uploaded successfully!")
    else:
        resume_text = st.text_area("Or paste your resume text", height=250)

    top_k = st.slider("Top results", 5, 30, 10)
    resume_skills = extract_skills(resume_text)

    # --- Matching ---
    if st.button("Match Jobs"):
        if len(resume_text.strip()) < 80:
            st.warning("Paste more resume text (at least 80 characters).")
            st.stop()

        res_emb = model.encode([resume_text], normalize_embeddings=True)
        sims = cosine_similarity(res_emb, jd_emb)[0]

        out = jobs.copy()
        out["match_score"] = (sims * 100).round(2)
        out = out.sort_values("match_score", ascending=False).head(top_k)

        # Download button
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Match Results",
            data=csv_bytes,
            file_name="job_matches.csv",
            mime="text/csv",
        )

        # Show cards
        st.write(f"Showing top **{len(out)}** matches")

        for _, row in out.iterrows():
            title = row.get("positionName", "Unknown Title")
            company = row.get("company", "Unknown Company")
            location = row.get("location", "")
            score = row.get("match_score", 0.0)

            apply_url = row.get("url", "") or row.get("externalApplyLink", "")
            jd_text = row.get("jd_clean", "") or row.get("description", "")

            job_skills = extract_skills(jd_text)
            matched = sorted(set(resume_skills) & set(job_skills))
            missing = sorted(set(job_skills) - set(resume_skills))

            st.subheader(f"{title} — {company}")
            if location:
                st.caption(f"📍 {location}")

            st.write(f"✅ Match Score: **{float(score):.2f}%**")

            if apply_url:
                st.write(f"🔗 Apply link: {apply_url}")

            st.write("**Matched skills:**", ", ".join(matched) if matched else "—")
            st.write("**Missing skills (top 10):**", ", ".join(missing[:10]) if missing else "—")

            with st.expander("Show job description"):
                st.write((row.get("description", "") or "")[:3000])

            st.markdown("---")
if __name__ == "__main__":
    main()
