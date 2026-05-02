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
OLLAMA_URL = "http://176.9.82.157:11434/api/generate"
COLLECTION = "rag_service"

class Query(BaseModel):
    question: str
    top_k: int = 8

GREETINGS = {"hi", "hello", "hey", "hiya", "howdy", "sup", "what's up", "whats up", "good morning", "good evening"}

PLATFORM_KEYWORDS = {"about this platform", "what is this", "what is this platform", "what does this do", "tell me about this", "what can you do", "what is mark-i"}

@app.post("/api/ask")
async def ask(query: Query):
    q = query.question.strip().lower().rstrip("!.?")

    # Handle greetings
    if q in GREETINGS:
        def greet_stream():
            msg = "Hi! I'm Mark-I, your personalized assistant. Ask me anything about this platform — features, functionality and more!"
            yield f"event: replace\ndata: {msg.replace(chr(10), chr(92)+'n')}\n\n"
            yield "event: done\ndata: \n\n"
        return StreamingResponse(greet_stream(), media_type="text/event-stream")

    # Handle platform overview
    if any(kw in q for kw in PLATFORM_KEYWORDS):
        def platform_stream():
            msg = (
                "## About This Platform\n"
                "This is an **IoT Device Management Platform** designed to help you monitor, manage, and maintain your IoT fleet from a single place.\n\n"
                "## Key Features\n"
                "**Dashboard**: At-a-glance overview of your entire device fleet with stats, alerts, and activity.\n"
                "**Device Inventory**: Browse, search, filter, and manage all your devices with bulk actions.\n"
                "**Device Groups**: Organize devices into logical groups for easier management.\n"
                "**Metrics**: System-wide performance monitoring with live CPU, RAM, network, and disk charts.\n"
                "**Command Executions**: Run and track commands across single devices or entire groups.\n\n"
                "## Services\n"
                "**Device API**: Manage and retrieve device data programmatically.\n"
                "**Metrics API**: Access real-time and historical performance metrics.\n"
                "**Device Group API**: Create and manage device groups via API.\n"
                "**Users API**: IAM-based user management for platform access control."
            )
            escaped = msg.replace("\n", "\\n")
            yield f"event: replace\ndata: {escaped}\n\n"
            yield "event: done\ndata: \n\n"
        return StreamingResponse(platform_stream(), media_type="text/event-stream")

    # 1. Retrieve from Qdrant
    q_emb = embedder.encode(query.question, normalize_embeddings=True).tolist()
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_emb,
        limit=query.top_k,
        with_payload=True,
        score_threshold=0.4,  # Only return relevant results
    ).points

    # If no relevant hits, return a fallback
    if not hits:
        def no_hits_stream():
            msg = "I don't have information about that. Try asking about features, devices, commands, metrics, or other platform functionality."
            yield f"event: replace\ndata: {msg}\n\n"
            yield "event: done\ndata: \n\n"
        return StreamingResponse(no_hits_stream(), media_type="text/event-stream")

    # Detect query intent
    list_keywords = {"list", "all", "show", "what are", "available", "give me"}
    is_list_query = any(kw in query.question.lower() for kw in list_keywords)

    # Detect if user is asking specifically about features or functionality
    asking_features = any(kw in query.question.lower() for kw in {"feature", "features", "pages", "ui", "interface"})
    asking_functionality = any(kw in query.question.lower() for kw in {"functionality", "functionalities", "api", "apis", "services"})

    # 2. Build context — filter by intent
    feature_parts = []
    functionality_parts = []
    context_parts = []
    for hit in hits:
        p = hit.payload
        if p.get("functionality"):
            # Only include functionality if user asked for it, or asked generically
            if asking_functionality or (not asking_features):
                functionality_parts.append(p['functionality'])
                context_parts.append(p['explanation'])
        if p.get("feature"):
            # Only include features if user asked for it, or asked generically
            if asking_features or (not asking_functionality):
                feature_parts.append(p['feature'])
                context_parts.append(p['explanation'])
    
    functionality = "\n".join(functionality_parts) if functionality_parts else ""
    feature = "\n".join(feature_parts) if feature_parts else ""
    context = "\n".join(context_parts) if context_parts else ""

    # feature_functionality_size = len(feature) + len(functionality)

    # Detect if user wants a list or detailed explanation
    list_keywords = {"list", "all", "show", "what are", "available", "give me"}
    is_list_query = any(kw in query.question.lower() for kw in list_keywords)

    if is_list_query:
        format_instruction = """STRICT Response format (follow exactly):
## Category Name
- Item Name
- Item Name

## Category Name
- Item Name

Example:
## Devices
- Device Info
- Dashboard
- Metrics

## Functionality
- Device API
- Metrics API

Rules:
- List ONLY the name, NO descriptions or explanations
- Every item MUST be a simple bullet point with just the name
- Do NOT add any description after the name
- Do NOT write items inline or in a paragraph"""
    else:
        format_instruction = """STRICT Response format:
## Category Header
1. **Item Name**: detailed description
2. **Item Name**: detailed description

Rules:
- Every item MUST be on its own line
- Do NOT write items inline or in a paragraph"""

    def stream():
        prompt = f"""You are Mark-I, a helpful assistant for an IoT device management platform.
Answer the user's question using ONLY the information provided below.
If the information is not relevant to the question, say "I don't have information about that."

IMPORTANT: The user asked about {'features only' if asking_features and not asking_functionality else 'functionality only' if asking_functionality and not asking_features else 'features and functionality'}.
Only include sections relevant to what was asked. Do NOT mix features and functionality unless both were requested.

{format_instruction}

User asked: {query.question}

{"Available Features:" + chr(10) + feature if feature else ""}

{"Available Functionality:" + chr(10) + functionality if functionality else ""}

Relevant details:
{context[:3000]}

Answer:
"""
        # print(prompt)
# Now write a clear, step-by-step guide for the user.
# Start your answer immediately with the first step:
        payload = {
            "model": "gemma2:9b",
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

            full_text = ""

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
                        full_text += token
                        # Apply formatting on every update and replace
                        corrected = insert_markdown_newlines(full_text)
                        escaped = corrected.replace("\n", "\\n")
                        yield f"event: replace\ndata: {escaped}\n\n"
                    if data.get("done", False):
                        yield "event: done\ndata: \n\n"
                        break
                except json.JSONDecodeError:
                    continue

            yield "event: done\ndata: \n\n"

        except Exception as e:
            yield f"event: error\ndata: Ollama failed: {str(e)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

def clean_token(token: str) -> str:
    # DO NOT lstrip — leading spaces are word separators from Ollama

    # Remove space before punctuation (but NOT before * or . to preserve markdown lists)
    token = re.sub(r"\s+([,:;!?])", r"\1", token)

    # Fix bold markdown spacing
    token = re.sub(r"\*\*\s+", "**", token)
    token = re.sub(r"\s+\*\*", "**", token)

    # Fix spacing around slashes
    token = re.sub(r"\s*/\s*", "/", token)

    # Collapse accidental double spaces
    token = re.sub(r" {2,}", " ", token)

    return token

def insert_markdown_newlines(text: str) -> str:
    """Insert newlines before list items if missing"""
    # Add newline before numbered list items: "1." "2." etc
    text = re.sub(r'(?<!\n)(\d+\.)\s', r'\n\1 ', text)
    # Add newline before bullet points: "* " or "- "
    text = re.sub(r'(?<!\n)([*\-])\s(?!\*)', r'\n\1 ', text)
    # Add newline before headers: "## "
    text = re.sub(r'(?<!\n)(#{1,3})\s', r'\n\1 ', text)
    # Fix "*Word" (italic misuse) → "* Word"
    text = re.sub(r'(?<!\*)\*(?!\*|\s)(\w)', r'* \1', text)
    # Fallback: add newline before "Word:" patterns that appear mid-sentence (capitalized word followed by colon)
    text = re.sub(r'(?<!\n)\.?\s+([A-Z][a-zA-Z\s]+:)', r'\n**\1**', text)
    return text