from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import os

# === Load model, data, and precomputed embeddings ===
model = SentenceTransformer("intfloat/e5-large-v2")
df = pd.read_csv("assessments_with_combined_text.csv")
desc_title_embs = torch.load("desc_title_embs.pt", map_location=torch.device('cpu'))
combined_embs = torch.load("combined_embs.pt", map_location=torch.device('cpu'))

df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0).astype(int)

TOP_K = 10
DESC_CAND_K = 30

# === FastAPI setup ===
app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "SHL Assessment Recommender API is live"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
def recommend_endpoint(input: QueryInput):
    query = input.query
    q_lower = query.lower()
    m = re.search(r"(\d{1,3})\s*(?:minutes|min)", q_lower)
    dur_target = int(m.group(1)) if m else None

    # Step 1: Encode query
    q_emb = model.encode(query, convert_to_tensor=True, device='cpu')

    # Step 2: Initial similarity using desc_title_embs (faster)
    desc_sims = util.cos_sim(q_emb, desc_title_embs)[0]
    top_desc_idx = torch.topk(desc_sims, k=min(len(df), DESC_CAND_K)).indices.tolist()
    sem_filtered_df = df.iloc[top_desc_idx].copy()
    sem_filtered_df["desc_title_sim"] = [desc_sims[i].item() for i in top_desc_idx]

    # Step 3: Keyword matching
    query_tokens = set(re.findall(r'\w+', q_lower))
    def keyword_match(kw):
        if pd.isna(kw): return False
        kw_tokens = set(re.findall(r'\w+', kw.lower()))
        return not query_tokens.isdisjoint(kw_tokens)
    keyword_filtered_df = df[df["keywords"].apply(keyword_match)]

    # Step 4: Combine both filters
    common_ids = set(sem_filtered_df.index).intersection(set(keyword_filtered_df.index))
    if not common_ids:
        return {"results": []}

    final_df = df.loc[sorted(common_ids)].copy()
    final_combined_embs = combined_embs[sorted(common_ids)]
    combined_sims = util.cos_sim(q_emb, final_combined_embs)[0]
    final_df["combined_sim"] = [combined_sims[i].item() for i in range(len(final_df))]

    # Step 5: Duration scoring
    dur_scores = []
    for _, row in final_df.iterrows():
        if dur_target and pd.notna(row["duration_minutes"]):
            diff = abs(int(row["duration_minutes"]) - dur_target)
            score = max(0.0, 1 - diff / max(1, dur_target))
        else:
            score = 0.5
        dur_scores.append(score)
    final_df["dur_score"] = dur_scores

    # Step 6: Final scoring and top-k
    final_df["final_score"] = list(zip(final_df["dur_score"], final_df["combined_sim"]))
    final_df_sorted = final_df.sort_values("final_score", ascending=False).head(TOP_K)

    # Step 7: Return results
    results = []
    for _, row in final_df_sorted.iterrows():
        results.append({
            "title": row["title"],
            "url": row["url"],
            "duration": row["duration_minutes"],
            "test_type": row["test_type"],
            "adaptive_support": row["adaptive_support"],
            "remote_support": row["remote_support"]
        })

    return {"results": results}

# === Run server ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)