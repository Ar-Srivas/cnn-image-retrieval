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

## 7) One-line functionality comments (key functions)

## [src/preprocess.py](src/preprocess.py)

- `preprocess_image(path)`: Loads an image, resizes to 224x224, and applies Gaussian blur.
- `get_histogram(img)`: Extracts a flattened 3D color histogram (8 bins per channel).

## [src/extract_features.py](src/extract_features.py)

- `get_cnn_features(img)`: Converts image to RGB tensor and extracts ResNet50 embedding with validation checks.

## [src/query_pipeline.py](src/query_pipeline.py)

- `process_query_image(image_path)`: Runs full preprocessing + histogram + CNN extraction for one query image.
- `batch_process_query_images(image_paths)`: Processes multiple query images and returns filename-to-feature mapping.
- `get_query_features_from_array(img_array)`: Same feature pipeline for in-memory uploaded images.

## [src/similarity_search.py](src/similarity_search.py)

- `load_feature_db(filepath)`: Loads serialized image feature database from pickle.
- `search_cosine(query_vec, feature_db, top_k)`: Returns top-K nearest images by cosine similarity.
- `search_euclidean(query_vec, feature_db, top_k)`: Returns top-K nearest images by Euclidean distance.

## [src/evaluate.py](src/evaluate.py)

- `get_ground_truth_category(filename)`: Derives pseudo-category label from filename pattern.
- `precision_at_k(retrieved, relevant, k)`: Computes precision among the top-K retrieved images.
- `recall_at_k(retrieved, relevant, k)`: Computes recall coverage among relevant images at K.
- `average_precision(retrieved, relevant)`: Computes AP for a single ranked retrieval list.
- `mean_average_precision(all_results)`: Computes mAP across multiple queries.
- `get_relevant_images(query_filename, all_filenames)`: Builds relevant set by same derived category.
- `evaluate_retrieval_system(feature_db, search_func, ...)`: Runs full evaluation loop over sampled queries.
- `compare_search_methods(feature_db, num_queries)`: Compares cosine vs euclidean retrieval metrics.
- `print_evaluation_report(comparison)`: Prints formatted evaluation summary to console.

## [src/model_comparison.py](src/model_comparison.py)

- `FeatureExtractor.extract_features(img)`: Converts image to normalized tensor and extracts model features.
- `FeatureExtractor.get_combined_features(img)`: Concatenates histogram and model embedding.
- `extract_features_with_model(extractor, dataset_path, max_images)`: Builds feature DB with a chosen CNN architecture.
- `compare_models(dataset_path, max_images, num_eval_queries)`: Benchmarks multiple CNN backbones on retrieval metrics.
- `print_model_comparison_report(results)`: Prints model-wise accuracy/speed comparison report.

## [run_feature_extraction.py](run_feature_extraction.py)

- `combine_features(img)`: Concatenates histogram and CNN embeddings into one vector.
- `main()`: Randomly samples dataset images, extracts features, and saves `features/image_features.pkl`.

## [demo_search.py](demo_search.py)

- `demo_search(query_image_path, top_k, method)`: End-to-end CLI demo for query processing and similarity retrieval.

## [generate_report.py](generate_report.py)

- `generate_text_report(...)`: Writes a human-readable evaluation report.
- `generate_json_report(...)`: Writes a machine-readable JSON evaluation report.
- `run_full_evaluation(...)`: Executes evaluation workflow and saves all outputs.

## [web_app.py](web_app.py) (advanced web logic)

- `_clip_features_from_image(image)`: Extracts CLIP image embedding robustly from different output wrappers.
- `_ollama_generate(image, prompt, ...)`: Calls Ollama Moondream and retries/fallbacks on low-quality captions.
- `_get_caption_for_file(filename, prompt)`: Retrieves cached caption or computes and stores a new one.
- `_build_gallery_signature(dataset_files)`: Creates cache signature to detect when CLIP index must be rebuilt.
- `_try_load_cached_clip_index(...)`: Loads persisted FAISS CLIP index when metadata matches.
- `_ensure_clip_gallery()`: Lazily builds or loads CLIP gallery index with lock-based concurrency safety.
- `_search_clip_vlm(query_image, top_k, alpha, prompt)`: Runs CLIP shortlist + VLM captions + SBERT fusion rerank.
- `search()`: Handles upload validation and dispatches request to CNN or CLIP-VLM retrieval branch.
- `serve_dataset_image(filename)`: Securely serves dataset images to UI.
- `serve_query_image(filename)`: Securely serves uploaded query images to UI.

## 8) Practical summary

- If you need speed and simple deployment, use the **classic CNN pipeline**.
- If you need stronger semantic matching, use **CLIP + VLM + SBERT fusion** with tuned `alpha`.
- Best semantic behavior occurs when VLM captions are high quality, because SBERT reranking quality depends on caption quality.
