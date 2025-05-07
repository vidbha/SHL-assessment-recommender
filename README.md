🔍 SHL Assessment Recommender System
This project is a GenAI-powered system that recommends SHL assessments based on natural language job descriptions or queries. It combines semantic search, keyword filtering, and duration-aware scoring using SentenceTransformer embeddings for accurate and context-aware recommendations.

🔗 Demo App: Hugging Face Space
📡 API Endpoint: Render App Docs
📁 GitHub Repo: GitHub - SHL Assessment Recommender

🚀 Features
🔍 Semantic Search using SentenceTransformer (intfloat/e5-large-v2)

🧠 Keyword Matching to filter relevant assessments

⏱️ Duration-Aware Ranking based on query constraints

🧾 Evaluation with Recall@10 and MAP@10

🌐 API Access via FastAPI

🎛️ Interactive Web App via Hugging Face Spaces

🧱 Project Structure
bash
Copy
Edit
.
├── api.py                         # FastAPI backend
├── assessments_with_combined_text.csv
├── combined_embs.pt               # Saved embeddings for combined text
├── desc_title_embs.pt            # Saved embeddings for title + description
├── requirements.txt              # Dependencies
├── shl_assessment_recommendet.ipynb  # Main notebook
├── .devcontainer/                # Dev container config
├── chromedriver_win32/          # ChromeDriver (for scraping)
├── experimented_datasets/       # Raw and cleaned datasets
└── __pycache__/
🧠 How It Works
Input Query Example:

css
Copy
Edit
"I want to hire Java developers for a 40-minute assessment who can work well with business teams."
Semantic Filtering:

Computes embeddings for job descriptions and queries.

Retrieves top matching assessments using cosine similarity.

Keyword Filtering:

Filters based on matching keywords from the query and metadata.

Final Scoring:

Computes a final score based on semantic similarity + duration closeness.

Returns top 10 recommended assessments with URLs.


📦 Installation
bash
Copy
Edit
git clone https://github.com/vidbha/SHL-assessment-recommender.git
cd SHL-assessment-recommender
pip install -r requirements.txt
▶️ Run Locally
1. Generate Embeddings
Run the Jupyter notebook shl_assessment_recommendet.ipynb to:

Load and clean data

Generate and save embeddings using intfloat/e5-large-v2

2. Start API
bash
Copy
Edit
uvicorn api:app --reload
Then open http://localhost:8000/docs to test.

🌍 Deployment
Backend: Render (FastAPI)

Frontend: Hugging Face Spaces (Gradio or Streamlit)

Data Storage: Local CSV + PyTorch-serialized embeddings

🙌 Acknowledgements
SHL for publicly available assessment catalog

Hugging Face for SentenceTransformer

Hackathon organizers for the evaluation dataset
