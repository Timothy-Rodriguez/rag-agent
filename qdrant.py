import csv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import hashlib


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = QdrantClient("localhost", port=6333)  # adjust if using Qdrant Cloud

collection_name = "rag_service"

# Create collection
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

points = []

with open("data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_path = row["file_path"].strip()
        if not file_path:
            continue
            
        # Combine all useful text for best retrieval
        text = f"File: {file_path}\n"
        if row["feature"]:
            text += f"Feature/Page: {row['feature']}\n"
        if row["functionality"]:
            text += f"Component: {row['functionality']}\n"
        text += row["explanation"]
        
        vector = model.encode(text).tolist()
        
        payload = {
            "file_path": file_path,
            "feature": row["feature"] or None,
            "functionality": row["functionality"] or None,
            "explanation": row["explanation"],
            "full_text": text
        }
        
        # Use hash of file_path as stable ID
        point_id = hashlib.md5(file_path.encode()).hexdigest()
        
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

# Batch upsert
client.upsert(collection_name=collection_name, points=points)
print(f"Successfully loaded {len(points)} chunks into Qdrant collection '{collection_name}'")