# rag_service.py — FINAL, 100% WORKING RAG + STREAMING
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import re


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant = QdrantClient("http://localhost:6333")
OLLAMA_URL = "http://localhost:11434/api/generate"
COLLECTION = "rag_service"

class Query(BaseModel):
    question: str
    top_k: int = 8

@app.post("/api/ask")
async def ask(query: Query):
    # 1. Retrieve from Qdrant
    q_emb = embedder.encode(query.question, normalize_embeddings=True).tolist()
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_emb,
        limit=query.top_k,
        with_payload=True,
    ).points

    # 2. Build context
    feature_parts = []
    functionality_parts = []
    context_parts = []
    for hit in hits:
        p = hit.payload
        if p.get("functionality"):
            functionality_parts.append(p['functionality'])
            context_parts.append(p['explanation'])
            # context_parts.append(f"functionality '{p['functionality']}': {p['explanation']}")
        if p.get("feature"):
            feature_parts.append(p['feature'])
            context_parts.append(p['explanation'])
            # context_parts.append(f"feature '{p['feature']}': {p['explanation']}")
    
    functionality = "\n".join(functionality_parts) if functionality_parts else ""
    feature = "\n".join(feature_parts) if feature_parts else ""
    context = "\n".join(context_parts) if context_parts else ""

    feature_functionality_size = len(feature) + len(functionality)

    # 3. Stream — THIS PROMPT FORCES GEMMA3 TO SPEAK
    def stream():
        yield f"event: sources\ndata: {json.dumps([hit.payload for hit in hits])}\n\ncontext:{context}\n\n{len(context)}"
        prompt = f"""Answer the user's question in simple steps using ONLY the information below.

User asked: {query.question}

Feature: {feature}

Functionality: {functionality}

Relevant explanation:
{context[:1000-feature_functionality_size]}

Answer in detail only if question is specific on a feature else answer short:
"""
        print(prompt)
# Now write a clear, step-by-step guide for the user.
# Start your answer immediately with the first step:
        payload = {
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192
            },
            "keep_alive": -1
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8').strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        cleaned_token = clean_token(token)
                        yield f"data: {cleaned_token}\n\n"
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

            yield "event: done\ndata: \n\n"

        except Exception as e:
            yield f"event: error\ndata: Ollama failed: {str(e)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

def clean_token(token: str) -> str:
    # 1. Remove LEADING spaces from tokens (root cause)
    token = token.lstrip()

    # 2. Remove space before punctuation
    token = re.sub(r"\s+([,:;.!?])", r"\1", token)

    # 3. Fix bold markdown spacing
    token = re.sub(r"\*\*\s+", "**", token)
    token = re.sub(r"\s+\*\*", "**", token)

    # 4. Fix spacing around slashes
    token = re.sub(r"\s*/\s*", "/", token)

    # 5. Collapse accidental double spaces
    token = re.sub(r" {2,}", " ", token)

    return token