import os
import io
import json
import base64
import urllib.request
import urllib.error
import faiss
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ================================================================= #
# STAGE 1: CLIP — visual embedding
# ================================================================= #
print("[INIT] Loading CLIP...")
clip_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_id)

# ================================================================= #
# STAGE 2: Moondream — image analysis and captioning (via Ollama)
# ================================================================= #
print("[INIT] Loading Moondream...")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "moondream")
OLLAMA_OPTIONS = {"temperature": 0, "num_predict": 50}
MOONDREAM_IDENTITY_PROMPT = (
    "Identify this object. Is it a real organic item or a manufactured prop/ornament? "
    "Look for artificial textures, seams, or unnatural gloss. Describe its material and nature."
)

# ================================================================= #
# STAGE 3: SBERT — semantic scoring
# ================================================================= #
print("[INIT] Loading SBERT...")
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# ================================================================= #
# FAISS index for image gallery
# ================================================================= #
index = faiss.IndexFlatL2(512)
image_gallery = {}

def get_clip_features(image: Image.Image):
    """Extract CLIP visual embeddings from image"""
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        tensor = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
    return tensor.cpu().numpy()

def _ollama_generate(image: Image.Image, prompt: str, options: dict | None = None) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": options or OLLAMA_OPTIONS
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", "ignore")
        raise ValueError(f"Ollama request failed: {e.code} {error_body}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Ollama not reachable at {OLLAMA_BASE_URL}. Is it running?") from e

    parsed = json.loads(body)
    if "response" not in parsed:
        raise ValueError(f"Unexpected Ollama response: {parsed}")

    return parsed["response"].strip()

def get_moondream_caption(image: Image.Image):
    """Generate identity reasoning using Moondream"""
    with torch.no_grad():
        caption = _ollama_generate(image, MOONDREAM_IDENTITY_PROMPT)
    return caption


@app.get("/")
def read_root():
    return {
        "message": "Multi-Stage Image Search Pipeline",
        "status": "running",
        "gallery_size": len(image_gallery)
    }

@app.post("/add-to-gallery")
async def add_to_gallery(file: UploadFile = File(...), image_id: str = None):
    """Add image to gallery for later retrieval"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if image_id is None:
            image_id = file.filename
        
        # Extract visual features and add to index
        emb = get_clip_features(image)
        faiss.normalize_L2(emb)
        index.add(emb)
        
        # Store caption and metadata using Moondream
        caption = get_moondream_caption(image)
        image_gallery[len(image_gallery)] = {
            "id": image_id,
            "caption": caption,
            "filename": file.filename
        }
        
        return {
            "status": "success",
            "image_id": image_id,
            "caption": caption,
            "gallery_size": len(image_gallery)
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.post("/search")
async def search_gallery(file: UploadFile = File(...), alpha: float = 0.4, top_k: int = 3):
    """Search gallery using multi-stage pipeline (visual + semantic)"""
    try:
        if len(image_gallery) == 0:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Gallery is empty"}
            )
        
        image_data = await file.read()
        query_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # STAGE 1: Visual search
        q_emb = get_clip_features(query_image)
        faiss.normalize_L2(q_emb)
        search_limit = min(max(top_k, 1), len(image_gallery))
        distances, indices = index.search(q_emb, search_limit)
        
        initial_matches = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            initial_matches.append({
                'idx': idx,
                'dist': float(dist),
                'rank': rank + 1
            })
        
        # STAGE 2: Semantic refinement
        q_caption = get_moondream_caption(query_image)
        q_text_emb = text_model.encode(q_caption, convert_to_tensor=True)
        
        final_results = []
        for match in initial_matches:
            idx = match['idx']
            gallery_item = image_gallery[idx]
            m_text_emb = text_model.encode(gallery_item['caption'], convert_to_tensor=True)
            semantic_score = util.cos_sim(q_text_emb, m_text_emb).item()
            visual_score = float(np.clip(1.0 - match['dist'] / 2.0, 0.0, 1.0))

            final_score = (alpha * visual_score) + ((1 - alpha) * semantic_score)
            analysis = (
                f"[FUSION SCORE: {final_score:.4f}] "
                f"(CLIP: {visual_score:.4f} | SBERT: {semantic_score:.4f})"
            )
            
            final_results.append({
                "id": gallery_item['id'],
                "filename": gallery_item['filename'],
                "caption": gallery_item['caption'],
                "visual_score": visual_score,
                "semantic_score": float(semantic_score),
                "final_score": float(final_score),
                "analysis": analysis,
                "old_rank": match['rank']
            })
        
        final_results = sorted(final_results, key=lambda x: x['final_score'], reverse=True)
        
        return {
            "status": "success",
            "query_caption": q_caption,
            "alpha": alpha,
            "results": final_results
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    """Generate caption for an image using Moondream"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        caption = get_moondream_caption(image)
        
        return {
            "status": "success",
            "caption": caption,
            "filename": file.filename
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.post("/moondream-analyze")
async def moondream_analyze(file: UploadFile = File(...), question: str = MOONDREAM_IDENTITY_PROMPT):
    """Analyze image with Moondream using identity reasoning"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        prompt = question.strip() if question else MOONDREAM_IDENTITY_PROMPT
        with torch.no_grad():
            answer = _ollama_generate(image, prompt)
        
        return {
            "status": "success",
            "question": prompt,
            "answer": answer,
            "filename": file.filename
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
