from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from rag import RAGEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    drug_name: str
    message: str
    history: Optional[List[HistoryMessage]] = []


@app.on_event("startup")
async def startup():
    app.state.rag = RAGEngine()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    history = [h.dict() for h in req.history]
    return app.state.rag.chat(req.drug_name, req.message, history)