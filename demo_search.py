"""
Demo Script: Query Image Search
Shows how to use the query pipeline to search for similar images.
"""

import sys
from pathlib import Path

from src.query_pipeline import process_query_image
from src.similarity_search import load_feature_db, search_cosine, search_euclidean


def demo_search(query_image_path: str, top_k: int = 5, method: str = 'cosine'):
    """
    Demo function to search for similar images.
    
    Args:
        query_image_path: Path to query image
        top_k: Number of similar images to retrieve
        method: 'cosine' or 'euclidean'
    """
    if top_k <= 0:
        print(f"Error: top_k must be positive, got {top_k}")
        return
    
    print("="*60)
    print("IMAGE SIMILARITY SEARCH DEMO")
    print("="*60)
    
    # Step 1: Load feature database
    print("\n[1/3] Loading feature database...")
    try:
        db = load_feature_db()
        print(f"✓ Loaded features for {len(db)} images")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Please run feature extraction first:")
        print("  python run_feature_extraction.py")
        return
    
    # Validate top_k against database size
    top_k = min(top_k, len(db))
    
    # Step 2: Process query image
    print(f"\n[2/3] Processing query image: {query_image_path}")
    try:
        query_features = process_query_image(query_image_path)
        print(f"✓ Extracted feature vector (dim={query_features.shape[0]})")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Step 3: Search for similar images
    print(f"\n[3/3] Searching for top {top_k} similar images...")
    
    if method == 'cosine':
        results, search_time = search_cosine(query_features, db, top_k=top_k)
        metric_name = "Cosine Similarity"
    else:
        results, search_time = search_euclidean(query_features, db, top_k=top_k)
        metric_name = "Euclidean Distance"
    
    print(f"✓ Search completed in {search_time*1000:.2f} ms")
    
    # Display results
    print("\n" + "="*60)
    print(f"RESULTS ({metric_name}):")
    print("="*60)
    
    for i, (filename, score) in enumerate(results, 1):
        print(f"{i:2d}. {filename:<40} | Score: {score:.4f}")
    
    print("="*60)
    print("\nNOTE: Images are in data/myntradataset/images/")
    print("You can view these images to verify similarity!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_search.py <query_image_path> [top_k] [method]")
        print("\nExample:")
        print("  python demo_search.py data/myntradataset/images/sample.jpg 5 cosine")
        print("\nArguments:")
        print("  query_image_path  : Path to the query image")
        print("  top_k            : Number of results (default: 5)")
        print("  method           : 'cosine' or 'euclidean' (default: cosine)")
        sys.exit(1)
    
    query_path = sys.argv[1]
    
    try:
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    except ValueError:
        print("Error: top_k must be an integer")
        sys.exit(1)
    
    method = sys.argv[3] if len(sys.argv) > 3 else 'cosine'
    
    if method not in ['cosine', 'euclidean']:
        print("Error: method must be 'cosine' or 'euclidean'")
        sys.exit(1)
    
    demo_search(query_path, top_k, method)
