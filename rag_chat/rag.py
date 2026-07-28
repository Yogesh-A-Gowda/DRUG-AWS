import os
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from groq import Groq

HF_DATASET_REPO = os.environ.get("EMBEDDINGS_REPO", "yogeshagowda/drug-reviews-embeddings")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


class RAGEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        print(f"Loading dataset {HF_DATASET_REPO}...")
        dataset = load_dataset(HF_DATASET_REPO, split="train")

        print("Converting to pandas for per-drug filtering...")
        df = dataset.to_pandas()
        # Normalize for case-insensitive drug_name matching
        df["drug_name_lower"] = df["drug_name"].astype(str).str.lower()
        # Stack embeddings into a single numpy array per row for fast cosine sim
        df["embedding"] = df["embedding"].apply(lambda e: np.array(e, dtype=np.float32))
        self.df = df

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        self.groq_client = Groq(api_key=groq_api_key)
        print("RAG engine ready.")

    def _cosine_topk(self, query_embedding, candidate_embeddings, k):
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        cand_matrix = np.stack(candidate_embeddings.values)
        cand_norms = cand_matrix / (np.linalg.norm(cand_matrix, axis=1, keepdims=True) + 1e-8)
        scores = cand_norms @ query_norm
        top_idx = np.argsort(scores)[::-1][:k]
        return top_idx, scores[top_idx]

    def retrieve(self, drug_name: str, query: str, k: int = 5):
        subset = self.df[self.df["drug_name_lower"] == drug_name.strip().lower()]

        # Fallback: if no exact match, search the whole dataset instead of
        # returning nothing
        if subset.empty:
            subset = self.df

        query_embedding = self.embed_model.encode([query])[0]
        top_idx, scores = self._cosine_topk(query_embedding, subset["embedding"], k)

        results = []
        subset_reset = subset.reset_index(drop=True)
        for i, score in zip(top_idx, scores):
            row = subset_reset.iloc[i]
            results.append({
                "score": float(score),
                "drug_name": row["drug_name"],
                "review_text": row["review_text"],
                "predicted_category": row["predicted_category"],
            })
        return results

    def build_context_message(self, drug_name: str, query: str, context_chunks: list):
        context_text = "\n\n".join(
            f"[Category: {c['predicted_category']}]\n{c['review_text']}"
            for c in context_chunks
        )
        return (
            f"Context — real patient reviews about {drug_name}:\n{context_text}\n\n"
            f"Question: {query}"
        )

    def chat(self, drug_name: str, message: str, history: list, k: int = 5):
        context_chunks = self.retrieve(drug_name, message, k=k)
        user_content = self.build_context_message(drug_name, message, context_chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant discussing patient reviews for a "
                    "specific drug. Use only the provided review context to answer. "
                    "If the context doesn't contain the answer, say you don't have "
                    "enough information. Be concise and factual."
                ),
            }
        ]
        # Preserve prior conversation turns for continuity
        for turn in history[-20:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_content})

        completion = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )

        return {
            "response": completion.choices[0].message.content,
            "sources": context_chunks,
        }