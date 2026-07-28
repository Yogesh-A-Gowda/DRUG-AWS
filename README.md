# Deployment layout

```
deploy/
├── classifier/                    # DistilBERT drug-review classifier service
│   ├── app.py                     # pulls model + CSV from HF Hub at startup
│   ├── requirements.txt
│   └── Dockerfile
├── rag_chat/                      # RAG + Groq chatbot service
│   ├── main.py
│   ├── rag.py                     # pulls embeddings dataset from HF Hub at startup
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example               # copy to .env and fill in your real GROQ_API_KEY
├── nginx/
│   └── drug-api.conf              # reverse proxy: /drugs, /analysis -> :8001, /chat -> :8002
├── push_final_dataset_to_hf.py    # run on Kaggle to push the CSV to its own dataset repo
└── refresh.sh                     # weekly restart script (cron target)
```

## No local model/CSV files needed anymore

Both services now fetch everything they need **directly from Hugging Face Hub at container startup**:
- Classifier pulls the model + `label_encoder.pkl` from `yogeshagowda/mtech-model` (public), and the CSV from a dataset repo (default name: `yogeshagowdaiiitdwd/drug-reviews-final-dataset` — override with the `DATASET_REPO` env var if you name it differently).
- Chatbot pulls the embeddings dataset from `yogeshagowdaiiitdwd/drug-reviews-embeddings` (override with `EMBEDDINGS_REPO`).

**Before first run:** push the CSV to its own dataset repo by running `push_final_dataset_to_hf.py` on Kaggle (edit `CSV_PATH` at the top if your filename differs).

## Persistent HF cache (important)

Without a persistent cache, every container restart would re-download the entire model/dataset from scratch. Create a named Docker volume once:

```bash
docker volume create hf-cache
```

Both `docker run` commands below mount this volume at `/root/.cache/huggingface` — on restart, `transformers`/`datasets` check the Hub for a newer revision and only re-download files that actually changed.

## Build & run — classifier service

```bash
cd classifier
docker build -t drug-classifier .
docker run -d --name classifier \
  -p 8001:7860 \
  -v hf-cache:/root/.cache/huggingface \
  --restart unless-stopped \
  drug-classifier
```

## Build & run — RAG chatbot service

```bash
cd rag_chat
cp .env.example .env
# edit .env and put your real GROQ_API_KEY in it
docker build -t drug-chatbot .
docker run -d --name chatbot \
  -p 8002:8000 \
  -v hf-cache:/root/.cache/huggingface \
  --env-file .env \
  --restart unless-stopped \
  drug-chatbot
```

## Weekly auto-refresh (pull latest model/dataset/embeddings)

`refresh.sh` just restarts both containers — that alone triggers `transformers`/`datasets` to check the Hub for updates and pull only what changed.

```bash
chmod +x refresh.sh
crontab -e
```

Add this line (runs every Sunday at 3 AM server time):

```
0 3 * * 0 /home/ubuntu/deploy/refresh.sh
```

Check `~/refresh.log` afterward to confirm it ran.

## Wire up Nginx

```bash
sudo cp nginx/drug-api.conf /etc/nginx/sites-available/drug-api.conf
sudo ln -s /etc/nginx/sites-available/drug-api.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

After this, everything is reachable on port 80:
- `http://<ec2-ip>/drugs`
- `http://<ec2-ip>/analysis/<drug_name>`
- `http://<ec2-ip>/chat` (POST)
