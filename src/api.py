from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from typing import Optional
from src.pipeline import RecommendationPipeline
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np
import uvicorn

app = FastAPI(title="Multi-Modal Semantic Recommendation System")

# Global variables for models
pipeline = None
clip_model = None

@app.on_event("startup")
def load_models():
    global pipeline, clip_model
    print("Loading models into API...")
    pipeline = RecommendationPipeline()
    clip_model = SentenceTransformer('clip-ViT-B-32')

# Mount static files
app.mount("/images", StaticFiles(directory="data/images"), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

@app.post("/recommend")
async def recommend(
    user_id: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    k: int = Form(10),
    image: Optional[UploadFile] = File(None)
):
    image_feat = None
    
    if image:
        # Process uploaded image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        image_feat = clip_model.encode(img, normalize_embeddings=True)
    elif query:
        # Text-based semantic search — CLIP encodes text into the same space as images
        image_feat = clip_model.encode(query, normalize_embeddings=True)
        
    # Get recommendations from pipeline
    results, retrieval_latency = pipeline.recommend(user_id=user_id, image_feat=image_feat, k=k)
    
    return {
        "user_id": user_id,
        "query": query,
        "results_count": len(results),
        "retrieval_latency_ms": round(retrieval_latency, 2),
        "recommendations": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
