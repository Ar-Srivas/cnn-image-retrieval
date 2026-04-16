import os
import io
import faiss
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import (
    CLIPProcessor, CLIPModel,
    LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer, util
from moondream import Moondream

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
# STAGE 2: LLaVA-1.5-7B — image captioning
# ================================================================= #
print("[INIT] Loading LLaVA-1.5 VLM...")
vlm_id = "llava-hf/llava-1.5-7b-hf"

if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    vlm_model = LlavaForConditionalGeneration.from_pretrained(
        vlm_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
else:
    vlm_model = LlavaForConditionalGeneration.from_pretrained(
        vlm_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

vlm_processor = AutoProcessor.from_pretrained(vlm_id)
vlm_model.eval()

# ================================================================= #
# STAGE 3: Moondream — alternative VLM for image analysis
# ================================================================= #
print("[INIT] Loading Moondream...")
moondream_model = Moondream.from_pretrained("vikhyatk/moondream2", trust_remote_code=True).to(device)
moondream_tokenizer = moondream_model.tokenizer

# ================================================================= #
# STAGE 4: SBERT — semantic scoring
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

def get_vlm_caption(image: Image.Image):
    """Generate caption using LLaVA-1.5"""
    prompt = "USER: <image>\nDescribe this object and its setting briefly.\nASSISTANT:"
    inputs = vlm_processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(vlm_model.device)

    with torch.no_grad():
        output_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    caption = vlm_processor.decode(generated, skip_special_tokens=True).strip()
    return caption

def get_moondream_caption(image: Image.Image):
    """Generate caption using Moondream"""
    with torch.no_grad():
        caption = moondream_model.caption(image, length="normal")
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
        
        # Store caption and metadata
        caption = get_vlm_caption(image)
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
async def search_gallery(file: UploadFile = File(...), alpha: float = 0.4):
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
        distances, indices = index.search(q_emb, len(image_gallery))
        
        initial_matches = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            initial_matches.append({
                'idx': idx,
                'visual_score': 1 / (1 + dist),
                'rank': rank + 1
            })
        
        # STAGE 2: Semantic refinement
        q_caption = get_vlm_caption(query_image)
        q_text_emb = text_model.encode(q_caption, convert_to_tensor=True)
        
        final_results = []
        for match in initial_matches:
            idx = match['idx']
            gallery_item = image_gallery[idx]
            m_text_emb = text_model.encode(gallery_item['caption'], convert_to_tensor=True)
            semantic_score = util.cos_sim(q_text_emb, m_text_emb).item()
            
            final_score = (alpha * match['visual_score']) + ((1 - alpha) * semantic_score)
            
            final_results.append({
                "id": gallery_item['id'],
                "filename": gallery_item['filename'],
                "caption": gallery_item['caption'],
                "visual_score": float(match['visual_score']),
                "semantic_score": float(semantic_score),
                "final_score": float(final_score),
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
async def caption_image(file: UploadFile = File(...), model: str = "llava"):
    """Generate caption for an image using LLaVA or Moondream"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if model == "moondream":
            caption = get_moondream_caption(image)
        else:
            caption = get_vlm_caption(image)
        
        return {
            "status": "success",
            "model": model,
            "caption": caption,
            "filename": file.filename
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.post("/moondream-analyze")
async def moondream_analyze(file: UploadFile = File(...), question: str = "Describe this image"):
    """Analyze image with Moondream - ask specific questions"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        with torch.no_grad():
            answer = moondream_model.query(image, question, moondream_tokenizer)
        
        return {
            "status": "success",
            "question": question,
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
