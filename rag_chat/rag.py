import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from groq import Groq

HF_DATASET_REPO = os.environ.get("EMBEDDINGS_REPO", "yogeshagowdaiiitdwd/drug-reviews-embeddings")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


class RAGEngine:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        print(f"Loading dataset {HF_DATASET_REPO}...")
        self.dataset = load_dataset(HF_DATASET_REPO, split="train")

        print("Building FAISS index...")
        self.dataset.add_faiss_index(column="embedding")

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        self.groq_client = Groq(api_key=groq_api_key)
        print("RAG engine ready.")

    def retrieve(self, query: str, k: int = 5):
        query_embedding = self.embed_model.encode([query])[0]
        scores, retrieved = self.dataset.get_nearest_examples("embedding", query_embedding, k=k)
        results = []
        for score, drug, review, category in zip(
            scores,
            retrieved["drug_name"],
            retrieved["review_text"],
            retrieved["predicted_category"],
        ):
            results.append({
                "score": float(score),
                "drug_name": drug,
                "review_text": review,
                "predicted_category": category,
            })
        return results

    def build_prompt(self, query: str, context_chunks: list):
        context_text = "\n\n".join(
            f"[Drug: {c['drug_name']} | Category: {c['predicted_category']}]\n{c['review_text']}"
            for c in context_chunks
        )
        return (
            "You are a helpful assistant answering questions about drug reviews. "
            "Use only the context below to answer. If the context doesn't contain "
            "the answer, say you don't have enough information.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    def chat(self, query: str, k: int = 5):
        context_chunks = self.retrieve(query, k=k)
        prompt = self.build_prompt(query, context_chunks)

        completion = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise, factual assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "sources": context_chunks,
        }
