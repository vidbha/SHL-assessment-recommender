# SHL Assessment Recommender

This project is a GenAI-powered SHL Assessment Recommender system that helps users find the most relevant SHL assessments based on natural language job descriptions or hiring queries. It supports semantic search, keyword filtering, and duration-based ranking to return accurate and job-aligned assessment recommendations.

---

## 🔗 Demo Links

- **API Swagger Docs**: [https://shl-assessment-recommender-wfao.onrender.com/docs](https://shl-assessment-recommender-wfao.onrender.com/docs)
- **Web App** (HuggingFace Spaces): [https://huggingface.co/spaces/vidbha214/shl-assessment-recommender-shf5a7yyhkahaxdcartkya](https://huggingface.co/spaces/vidbha214/shl-assessment-recommender-shf5a7yyhkahaxdcartkya)
- **GitHub Repository**: [https://github.com/vidbha/SHL-assessment-recommender](https://github.com/vidbha/SHL-assessment-recommender)
- **FastApi Deployment**: [https://huggingface.co/spaces/vidbha214/link](https://huggingface.co/spaces/vidbha214/link)
---

## 📁 Project Structure

```
├── api.py                            # FastAPI server with endpoints
├── requirements.txt
├── shl_assessment_recommendet.ipynb # Development Notebook  
```

---

## 🚀 Features

- Natural language job query input (e.g., "Looking for sales assessments under 60 minutes")
- Two-stage semantic and keyword filtering
- Duration-based scoring for user time preferences
- Final ranking using SentenceTransformer embeddings and similarity metrics
- Evaluation with Recall@10 and MAP@10

---

## ⚙️ How It Works

- Loads the sentence transformer model `intfloat/e5-large-v2`
- Encodes query and assessment metadata (titles, descriptions, keywords)
- Filters candidates semantically and by keyword overlap
- Computes duration match score
- Ranks and returns top 10 assessments based on similarity + duration

---

## 📦 API Usage

**Endpoint**: `POST /recommend`

**Payload**:
```json
{
  "query": "I'm hiring an entry-level content writer with strong English and SEO skills."
}
```

**Response**:
```json
{
  "recommendations": [
    {
      "name": "Search Engine Optimization (New)",
      "url": "https://www.shl.com/products/product-catalog/view/search-engine-optimization-new/",
      "duration": 40,
      "remote": true,
      "adaptive": true
    },
    ...
  ]
}
```

---

## 🧱 Core Dependencies

- `sentence-transformers`
- `pandas`
- `torch`
- `FastAPI`
- `Uvicorn`
- `Gradio` (for HuggingFace UI)

---

## 📊 Run Locally

```bash
git clone https://github.com/vidbha/SHL-assessment-recommender
cd SHL-assessment-recommender
pip install -r requirements.txt
uvicorn api:app --reload
```

