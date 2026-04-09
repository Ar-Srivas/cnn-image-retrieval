# CNN Image Retrieval System

A content-based image retrieval system using deep learning (CNN) features and similarity search.

## Features

- **Query Image Pipeline**: Process and extract features from uploaded images
- **Multiple CNN Models**: Compare ResNet50, ResNet18, VGG16, EfficientNet, MobileNetV2
- **Similarity Search**: Cosine similarity and Euclidean distance metrics
- **Performance Evaluation**: Precision@K, Recall@K, mAP metrics
- **Comprehensive Reports**: Automated evaluation report generation

---

## Setup

### 1. Install uv (Python Package Manager)

```bash
pip install uv
```

### 2. Install Dependencies

```bash
uv sync
```

---

## Usage

### 1. Extract Features from Dataset

Process images and extract features using ResNet50:

```bash
python run_feature_extraction.py
```

This creates `features/image_features.pkl` with feature vectors.

### 2. Search for Similar Images

Query with an image to find similar ones:

```bash
python demo_search.py <query_image_path> [top_k] [method]
```

**Example:**
```bash
# Find top 5 similar images using cosine similarity
python demo_search.py data/myntradataset/images/sample.jpg 5 cosine

# Find top 10 similar images using euclidean distance
python demo_search.py data/myntradataset/images/sample.jpg 10 euclidean
```

### 3. Evaluate System Performance

Run comprehensive evaluation (search methods comparison):

```bash
python generate_report.py --queries 100
```

**With model comparison** (time-consuming):
```bash
python generate_report.py --compare-models --queries 50
```

Outputs:
- `features/evaluation_report.txt` - Human-readable report
- `features/evaluation_report.json` - Machine-readable results
- `features/evaluation_results.pkl` - Raw data

### 4. Compare Different Models

Test multiple CNN architectures:

```bash
python -m src.model_comparison
```

---

## Project Structure

```
├── src/
│   ├── preprocess.py          # Image preprocessing & histogram
│   ├── extract_features.py    # CNN feature extraction (ResNet50)
│   ├── query_pipeline.py      # Query image processing
│   ├── similarity_search.py   # Search algorithms
│   ├── evaluate.py            # Evaluation metrics
│   └── model_comparison.py    # Compare CNN models
├── run_feature_extraction.py  # Extract features from dataset
├── demo_search.py             # Demo search script
├── generate_report.py         # Generate evaluation report
└── features/                  # Extracted features & reports
```

---

## Evaluation Metrics

- **Precision@K**: How many of the top K results are relevant
- **Recall@K**: How many relevant images were found in top K
- **mAP**: Mean Average Precision across all queries
- **Search Time**: Average query processing time

---

## Requirements

- Python 3.11+
- PyTorch
- torchvision
- OpenCV
- scikit-learn
- NumPy, Pandas, tqdm

See `pyproject.toml` for complete dependencies.
