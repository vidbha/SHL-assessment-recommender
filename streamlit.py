import streamlit as st
import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer, util

# === Load model and data ===
model = SentenceTransformer("intfloat/e5-large-v2",device='cpu')
df = pd.read_csv("assessments_with_combined_text.csv")
df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0).astype(int)

TOP_K = 10
DESC_CAND_K = 30
BATCH_SIZE = 64

# === Recommendation Logic ===
def recommend_assessments(query, df, top_k=TOP_K, desc_cand_k=DESC_CAND_K):
    q_lower = query.lower()
    m = re.search(r"(\d{1,3})\s*(?:minutes|min)", q_lower)
    dur_target = int(m.group(1)) if m else None

    desc_title_texts = (df["cleaned_title"].fillna("") + " " + df["description"].fillna("")).tolist()
    desc_title_embs = model.encode(desc_title_texts, convert_to_tensor=True, batch_size=BATCH_SIZE)
    q_emb = model.encode(query, convert_to_tensor=True)
    desc_sims = util.cos_sim(q_emb, desc_title_embs)[0]
    top_desc_idx = torch.topk(desc_sims, k=min(len(df), desc_cand_k)).indices.tolist()
    sem_filtered_df = df.iloc[top_desc_idx].copy()
    sem_filtered_df["desc_title_sim"] = [desc_sims[i].item() for i in top_desc_idx]

    query_tokens = set(re.findall(r'\w+', q_lower))
    def keyword_match(kw):
        if pd.isna(kw): return False
        kw_tokens = set(re.findall(r'\w+', kw.lower()))
        return not query_tokens.isdisjoint(kw_tokens)
    keyword_filtered_df = df[df["keywords"].apply(keyword_match)]

    common_ids = set(sem_filtered_df.index).intersection(set(keyword_filtered_df.index))
    if not common_ids:
        return pd.DataFrame()

    final_df = df.loc[sorted(common_ids)].copy()
    combined_embs = model.encode(final_df["combined_text"].fillna("").tolist(), convert_to_tensor=True, batch_size=BATCH_SIZE)
    combined_sims = util.cos_sim(q_emb, combined_embs)[0]
    final_df["combined_sim"] = [combined_sims[i].item() for i in range(len(final_df))]

    dur_scores = []
    for _, row in final_df.iterrows():
        if dur_target and pd.notna(row["duration_minutes"]):
            diff = abs(int(row["duration_minutes"]) - dur_target)
            score = max(0.0, 1 - diff / max(1, dur_target))
        else:
            score = 0.5
        dur_scores.append(score)
    final_df["dur_score"] = dur_scores

    final_df["final_score"] = list(zip(final_df["dur_score"], final_df["combined_sim"]))
    final_df_sorted = final_df.sort_values("final_score", ascending=False)

    return final_df_sorted.head(top_k)

# === Streamlit UI ===
st.title("Assessment Recommender")
query = st.text_input("Enter a job description or keywords:")

if query:
    with st.spinner("Generating recommendations..."):
        results = recommend_assessments(query, df)

        if results.empty:
            st.warning("No matching assessments found.")
        else:
            st.success(f"Top {len(results)} recommendations:")
            for i, row in results.iterrows():
                st.markdown(f"""
                ### {row['title']}
                - Duration: {row['duration_minutes']} min  
                - Test Type: {row['test_type']}  
                - Adaptive: {row['adaptive_support']}  
                - Remote: {row['remote_support']}  
                - [View Assessment]({row['url']})
                """)

