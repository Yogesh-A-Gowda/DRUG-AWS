import pandas as pd
import torch
import joblib
import os
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIG ---
# Repo IDs are env-configurable so you can point at a renamed/new repo
# without touching code.
MODEL_REPO = os.environ.get("MODEL_REPO", "yogeshagowda/mtech-model")
DATASET_REPO = os.environ.get("DATASET_REPO", "yogeshagowda/drug-reviews-final-dataset")
CSV_FILENAME = os.environ.get("CSV_FILENAME", "drug_reviews_imputed_rf.csv")
# Optional: only needed if either repo above is ever made private
HF_TOKEN = os.environ.get("HF_TOKEN")

# Optional exact commit hash to pin to. Set by refresh.sh when it wants to
# test/roll-back to a specific known version instead of always "latest".
# Empty string / unset = latest ("main").
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None
DATASET_REVISION = os.environ.get("DATASET_REVISION") or None

app = FastAPI()

# Allow your frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup():
    print(f"Fetching model snapshot from {MODEL_REPO} (revision={MODEL_REVISION or 'latest'}) ...")
    # snapshot_download pulls the whole repo (config, weights, tokenizer,
    # label_encoder.pkl) into the local HF cache. If HF_HOME is a mounted
    # volume, unchanged files are served from cache on later restarts;
    # only changed/new files are re-downloaded. Pinning to a specific
    # revision lets refresh.sh test a new version before committing to it,
    # and roll back to a known-good revision if something breaks.
    model_dir = snapshot_download(repo_id=MODEL_REPO, token=HF_TOKEN, revision=MODEL_REVISION)

    print("Booting BERT Engine...")
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    app.state.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    app.state.model.eval()
    app.state.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    print(f"Fetching dataset CSV from {DATASET_REPO} (revision={DATASET_REVISION or 'latest'}) ...")
    csv_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=CSV_FILENAME,
        repo_type="dataset",
        token=HF_TOKEN,
        revision=DATASET_REVISION,
    )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    if 'drug_review' in df.columns:
        df.rename(columns={'drug_review': 'review_text'}, inplace=True)

    app.state.raw_df = df
    app.state.drug_list = sorted(df['drug_name'].astype(str).unique().tolist())
    print("System Ready.")

def predict_batch(texts):
    """Speeds up inference by batching instead of .apply()"""
    inputs = app.state.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128  # Shorter length = faster CPU speed
    ).to(device)

    with torch.no_grad():
        outputs = app.state.model(**inputs)

    indices = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return app.state.label_encoder.inverse_transform(indices)

@app.get("/health")
def health():
    return {"status": "ok", "drugs_loaded": len(app.state.drug_list)}

@app.get("/drugs")
def get_drugs():
    return app.state.drug_list

@app.get("/analysis/{drug_name}")
def analyze(drug_name: str, skip: int = 0, limit: int = 50):
    df = app.state.raw_df
    # Filter for the specific drug
    drug_df_full = df[df['drug_name'].str.lower() == drug_name.lower()].copy()

    if drug_df_full.empty:
        raise HTTPException(status_code=404, detail="Drug not found")

    total_database_count = len(drug_df_full)

    # Slice the dataframe based on pagination parameters
    drug_df_paged = drug_df_full.iloc[skip: skip + limit].copy()

    # Batch Predict only the current visible subset (e.g., 50 reviews)
    texts = drug_df_paged['review_text'].astype(str).tolist()
    drug_df_paged['predicted_category'] = predict_batch(texts)

    # Calculate Trust Score
    weights = {
        "Positive_Experience": 1.0, "Mixed_Feedback": 0.0, "Ineffective": -0.7,
        "Dosage_Issues": -0.4, "Severe_Side_Effects": -0.9, "Dependency/Addiction": -1.0
    }
    drug_df_paged['score_val'] = drug_df_paged['predicted_category'].map(weights)
    trust_score = (drug_df_paged['score_val'].mean() + 1) / 2

    return {
        "drug_name": drug_name,
        "trust_score": round(float(trust_score), 3),
        "total_reviews": total_database_count,
        "showing": len(drug_df_paged),
        "review_summary": drug_df_paged['predicted_category'].value_counts().to_dict(),
        "all_classified_reviews": drug_df_paged.to_dict('records')
    }