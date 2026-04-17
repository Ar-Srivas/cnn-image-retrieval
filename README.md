# CNN Image Retrieval App: Complete Pipeline Breakdown

## 1) What the app does

This project implements two retrieval systems behind one web interface:

1. **Classic CNN retrieval pipeline** (fast, feature-vector matching)
2. **Advanced CLIP + VLM + SBERT fusion pipeline** (visual shortlist + semantic reranking)

Web runtime in active use: **Flask app in [web_app.py](web_app.py)**.

At a high level, the app takes a query image, converts it into embeddings, compares those embeddings against a gallery/database, and returns top-K visually/semantically similar images.

---

## 2) End-to-end architecture

### A) Classic CNN path

```
Dataset images
  -> preprocess (resize + Gaussian blur)
  -> histogram (512-d) + ResNet50 features (2048-d)
  -> concatenate to 2560-d vector
  -> store in features/image_features.pkl

Query image
  -> same preprocess + feature extraction
  -> cosine or euclidean search vs stored vectors
  -> top-K retrieval results
```

### B) Advanced CLIP + VLM + SBERT path

```
Dataset images (capped set)
  -> CLIP image embeddings (512-d)
  -> FAISS L2 index build/cache

Query image
  -> CLIP embedding
  -> FAISS shortlist (visual candidates)
  -> VLM (Moondream) caption for query
  -> VLM caption for each candidate (with cache)
  -> SBERT text embeddings for query/candidates
  -> fused score = alpha*visual + (1-alpha)*semantic
  -> reranked top-K results
```

---

## 3) Models used

## Core/classic models

- **ResNet50 (ImageNet pretrained)**
  - Used in [src/extract_features.py](src/extract_features.py) as the default CNN feature extractor.
  - Final classification layer is removed; penultimate feature output is used.

## Model comparison module

- **ResNet50**, **ResNet18**, **VGG16**, **EfficientNet-B0**, **MobileNetV2**
  - Implemented in [src/model_comparison.py](src/model_comparison.py).
  - All loaded with ImageNet pretrained weights for feature extraction benchmarking.

## Advanced retrieval models

- **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`)
  - Image encoder for gallery indexing and query visual search.
  - Implemented in [web_app.py](web_app.py).

- **Moondream VLM via Ollama**
  - Generates semantic captions/identity descriptions from images.
  - Used for query and candidate image textual reasoning.

- **SBERT** (`all-MiniLM-L6-v2`)
  - Encodes captions and computes semantic similarity.
  - Used in reranking stage after CLIP shortlist.

- **FAISS IndexFlatL2**
  - Fast nearest-neighbor search for CLIP embeddings.

---

## 4) Preprocessing done

The main preprocessing logic is in [src/preprocess.py](src/preprocess.py):

1. Load image using OpenCV (`cv2.imread`)
2. Resize to `224x224` (`cv2.resize`)
3. Apply Gaussian smoothing (`cv2.GaussianBlur`, kernel `(5,5)`)

Feature composition in classic pipeline:

- **Color histogram**: 3D histogram with `8x8x8` bins (`512` dims)
- **CNN feature**: ResNet50 embedding (`2048` dims)
- **Combined feature**: concatenated `512 + 2048 = 2560` dims

Additional tensor/image normalization in model-comparison path:

- Converts BGR to RGB (`cv2.cvtColor`)
- Applies torchvision normalization with ImageNet mean/std

---

## 5) Basic computer vision functions used

These are the primary CV operations used in the repo:

- `cv2.imread` - read image from disk
- `cv2.resize` - standardize spatial size to model input resolution
- `cv2.GaussianBlur` - denoise/smooth local noise before feature extraction
- `cv2.calcHist` - compute color distribution histogram features
- `cv2.cvtColor` - convert OpenCV BGR image to RGB for deep models
- `PIL.Image.open(...).convert("RGB")` - image loading/format normalization in web pipeline

No edge detectors, corner detectors, morphology, or optical flow are used in the current implementation.

---

## 6) Advanced CLIP -> VLM -> SBERT pipeline idea

This is the main idea behind the advanced retrieval mode:

1. **CLIP first-pass retrieval** (fast visual similarity)
   - Query image is embedded with CLIP.
   - FAISS returns top visual candidates.

2. **VLM semantic grounding**
   - Query image is described by Moondream (caption/identity reasoning).
   - Each shortlisted candidate also gets a Moondream caption.
   - Captions are cached to avoid repeated VLM calls and improve latency.

3. **SBERT semantic matching**
   - Query caption and candidate captions are embedded by SBERT.
   - Semantic similarity is computed with cosine similarity.

4. **Score fusion + rerank**
   - Visual score from CLIP distance and semantic score from SBERT are combined:

$$
\text{final\_score} = \alpha \cdot \text{visual\_score} + (1-\alpha) \cdot \text{semantic\_score}
$$

   - Results are sorted by `final_score` and top-K returned.

### Why this helps

- CLIP gives robust visual nearest neighbors quickly.
- VLM adds higher-level object/material/context reasoning.
- SBERT makes textual reasoning comparable in a dense semantic space.
- Fusion balances appearance-level match with concept-level match.

---

## 7) Run locally on a new device (uv + Ollama CLI)

### A) Install prerequisites

```powershell
# Install uv (Python package/project manager)
winget install --id Astral-sh.uv -e

# Install Ollama
winget install --id Ollama.Ollama -e
```

### B) Get project and install Python dependencies

```powershell
git clone https://github.com/Ar-Srivas/Semantic-Reverse-Search.git
cd Semantic-Reverse-Search

# Create .venv and install dependencies from pyproject.toml
uv sync
```

### C) Start Ollama and download Moondream

```powershell
# In a separate terminal, start Ollama server (skip if already running as a service)
ollama serve

# Pull the model used by web_app.py
ollama pull moondream
```

### D) Build CNN feature database (required for classic retrieval)

```powershell
uv run python run_feature_extraction.py
```

### E) Run the web app

```powershell
uv run python web_app.py
```

Open this URL in your browser:

```text
http://127.0.0.1:5000
```

### F) Optional: run quick CLI demo search

```powershell
uv run python demo_search.py
```

## 8) Practical summary

- If you need speed and simple deployment, use the **classic CNN pipeline**.
- If you need stronger semantic matching, use **CLIP + VLM + SBERT fusion** with tuned `alpha`.
- Best semantic behavior occurs when VLM captions are high quality, because SBERT reranking quality depends on caption quality.
