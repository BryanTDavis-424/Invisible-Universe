from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import clip
import torch
import faiss
import pickle
from pathlib import Path

app = FastAPI(title="Video Frame Search API")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.read_index("index.faiss")
with open("frame_paths.pkl", 'rb') as f:
    frame_paths = pickle.load(f)

app.mount("/frames", StaticFiles(directory="frames"), name="frames")

@app.get("/")
async def root():
    return {"message": "Welcome to the Video Frame Search API"}

@app.get("/search")
async def search(query: str = Query(..., description="Search query")):
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([query]).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    query_vector = text_features.cpu().numpy().astype('float32')
    
    k = 4  # Number of results to return
    distances, indices = index.search(query_vector, k)
    
    results = []
    for idx in indices[0]:
        frame_path = Path(frame_paths[idx])
        image_url = f"/frames/{frame_path.name}"
        results.append({"image_url": image_url})
    
    return {"results": results}