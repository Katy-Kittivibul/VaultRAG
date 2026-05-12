import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = os.environ["OBSIDIAN_VAULT_PATH"]
OUTPUT_PATH = Path("results/testset.json")

TESTSET = [
    {
        "question": "What deep learning frameworks do I know?",
        "ground_truth": "PyTorch and TensorFlow/Keras are the primary deep learning frameworks used, with PyTorch preferred for research and custom model work.",
    },
    {
        "question": "What is the Sheffield air quality project?",
        "ground_truth": "A project analysing air quality sensor data across Sheffield, involving time-series processing, geospatial visualisation, and anomaly detection on pollution readings.",
    },
    {
        "question": "What databases have I worked with?",
        "ground_truth": "Experience includes PostgreSQL, SQLite, and BigQuery for structured data, alongside ChromaDB and FAISS for vector storage.",
    },
    {
        "question": "What are my geospatial tools?",
        "ground_truth": "GeoPandas, Folium, and Shapely are used for geospatial analysis, with QGIS for desktop GIS work and Leaflet for interactive web maps.",
    },
    {
        "question": "What chemistry tools do I use?",
        "ground_truth": "RDKit is the main cheminformatics library used for molecular processing, fingerprinting, and property prediction tasks.",
    },
    {
        "question": "What machine learning libraries am I most familiar with?",
        "ground_truth": "Scikit-learn is the primary ML library for classical models, with XGBoost and LightGBM for gradient boosting and Optuna for hyperparameter optimisation.",
    },
    {
        "question": "What cloud platforms have I used?",
        "ground_truth": "Google Cloud Platform (BigQuery, Vertex AI) and AWS (S3, SageMaker) have been used for data storage, model training, and deployment.",
    },
    {
        "question": "What NLP tools or libraries have I worked with?",
        "ground_truth": "Hugging Face Transformers for fine-tuning and inference, spaCy for pipeline NLP, and LangChain for building LLM-powered applications.",
    },
    {
        "question": "What data visualisation tools do I use?",
        "ground_truth": "Matplotlib and Seaborn for static plots, Plotly and Altair for interactive charts, and Streamlit for building data dashboards.",
    },
    {
        "question": "What MLOps or deployment tools do I know?",
        "ground_truth": "MLflow for experiment tracking and model registry, Docker for containerisation, and GitHub Actions for CI/CD pipelines.",
    },
    {
        "question": "What have I worked on related to RAG or LLMs?",
        "ground_truth": "Built RAG pipelines using LangChain, ChromaDB, and Gemini embeddings to query personal knowledge bases; evaluated retrieval quality using RAGAS metrics.",
    },
    {
        "question": "What programming languages do I know?",
        "ground_truth": "Python is the primary language for all data science and ML work, with SQL used extensively for data querying and some R for statistical analysis.",
    },
    {
        "question": "What statistical methods do I apply in my projects?",
        "ground_truth": "Hypothesis testing, regression analysis, bootstrapping, and Bayesian inference are commonly applied, using scipy.stats and statsmodels.",
    },
    {
        "question": "What are my main skills in feature engineering?",
        "ground_truth": "Feature engineering skills include handling missing data, encoding categorical variables, time-series feature extraction, and using domain knowledge for derived features.",
    },
    {
        "question": "What projects have I completed involving time-series data?",
        "ground_truth": "Projects include air quality sensor analysis and financial data forecasting, using Prophet, statsmodels ARIMA, and LSTM-based models for prediction.",
    },
]


def _collect_md_files(vault: Path) -> list[Path]:
    return [p for p in vault.rglob("*.md") if ".claude" not in p.parts]


def main() -> None:
    vault = Path(VAULT_PATH)
    md_files = _collect_md_files(vault)
    print(f"Vault loaded: {len(md_files)} .md files found.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(TESTSET, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Testset saved: {len(TESTSET)} pairs")


if __name__ == "__main__":
    main()
