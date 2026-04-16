"""
Evaluation Metrics for Image Retrieval System
Implements precision@k, recall@k, mAP, and retrieval accuracy.
"""
import numpy as np
import pickle
from typing import List, Tuple, Dict, Callable
from pathlib import Path
import time
from tqdm import tqdm

from src.similarity_search import search_cosine, search_euclidean, load_feature_db
from src.query_pipeline import process_query_image


def get_ground_truth_category(filename: str) -> str:
    """
    Extract category from filename.
    Assumes format: category_id.jpg or similar pattern.
    Adjust based on your dataset structure.
    """
    # Example: "shirts_1234.jpg" -> "shirts"
    # Modify this based on your actual filename convention
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[0]
    return filename.split('.')[0]


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        retrieved: List of retrieved image filenames (ordered by relevance)
        relevant: List of ground truth relevant image filenames
        k: Number of top results to consider
        
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0 or len(retrieved) == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    num_relevant_retrieved = len(retrieved_at_k & relevant_set)
    return num_relevant_retrieved / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        retrieved: List of retrieved image filenames
        relevant: List of ground truth relevant images
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    num_relevant_retrieved = len(retrieved_at_k & relevant_set)
    return num_relevant_retrieved / len(relevant_set)


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Average Precision (AP) for a single query.
    
    Args:
        retrieved: Ordered list of retrieved images
        relevant: Set of relevant images
        
    Returns:
        Average Precision score
    """
    relevant_set = set(relevant)
    
    if len(relevant_set) == 0:
        return 0.0
    
    precision_sum = 0.0
    num_relevant_found = 0
    
    for i, img in enumerate(retrieved, 1):
        if img in relevant_set:
            num_relevant_found += 1
            precision_sum += num_relevant_found / i
    
    if num_relevant_found == 0:
        return 0.0
    
    return precision_sum / len(relevant_set)


def mean_average_precision(all_results: List[Tuple[List[str], List[str]]]) -> float:
    """
    Calculate Mean Average Precision (mAP) across all queries.
    
    Args:
        all_results: List of (retrieved, relevant) tuples for each query
        
    Returns:
        mAP score
    """
    if len(all_results) == 0:
        return 0.0
    
    ap_scores = [average_precision(retrieved, relevant) 
                 for retrieved, relevant in all_results]
    
    if len(ap_scores) == 0:
        return 0.0
    
    return np.mean(ap_scores)


def get_relevant_images(query_filename: str, all_filenames: List[str]) -> List[str]:
    """
    Determine which images are relevant for a given query.
    Images in the same category are considered relevant.
    
    Args:
        query_filename: Query image filename
        all_filenames: All available image filenames
        
    Returns:
        List of relevant image filenames (excluding the query itself)
    """
    query_category = get_ground_truth_category(query_filename)
    
    relevant = [
        fname for fname in all_filenames 
        if get_ground_truth_category(fname) == query_category 
        and fname != query_filename
    ]
    
    return relevant


def evaluate_retrieval_system(
    feature_db: Dict[str, np.ndarray],
    search_func: Callable,
    k_values: List[int] = [1, 5, 10, 20],
    num_queries: int = 100
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the retrieval system.
    
    Args:
        feature_db: Dictionary of image features
        search_func: Search function (search_cosine or search_euclidean)
        k_values: List of k values for precision@k and recall@k
        num_queries: Number of random queries to test
        
    Returns:
        Dictionary of evaluation metrics
    """
    all_filenames = list(feature_db.keys())
    
    if len(all_filenames) == 0:
        raise ValueError("Feature database is empty")
    
    # Randomly sample queries
    np.random.seed(42)  # For reproducibility
    query_samples = np.random.choice(
        all_filenames, 
        size=min(num_queries, len(all_filenames)), 
        replace=False
    )
    
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    ap_scores = []
    search_times = []
    
    print(f"Evaluating on {len(query_samples)} queries...")
    
    for query_name in tqdm(query_samples):
        query_vec = feature_db[query_name]
        
        # Get relevant images (same category)
        relevant = get_relevant_images(query_name, all_filenames)
        
        if len(relevant) == 0:
            continue  # Skip if no relevant images
        
        # Perform search
        max_k = max(k_values)
        results, search_time = search_func(query_vec, feature_db, top_k=max_k + 1)
        search_times.append(search_time)
        
        # Remove query image itself from results
        retrieved = [name for name, _ in results if name != query_name]
        
        # Calculate metrics
        for k in k_values:
            precision_scores[k].append(precision_at_k(retrieved, relevant, k))
            recall_scores[k].append(recall_at_k(retrieved, relevant, k))
        
        ap_scores.append(average_precision(retrieved[:max_k], relevant))
    
    # Aggregate results (handle empty arrays)
    metrics = {}
    
    if len(ap_scores) > 0:
        metrics['mAP'] = np.mean(ap_scores)
    else:
        metrics['mAP'] = 0.0
    
    if len(search_times) > 0:
        metrics['avg_search_time'] = np.mean(search_times)
        metrics['median_search_time'] = np.median(search_times)
    else:
        metrics['avg_search_time'] = 0.0
        metrics['median_search_time'] = 0.0
    
    for k in k_values:
        if len(precision_scores[k]) > 0:
            metrics[f'precision@{k}'] = np.mean(precision_scores[k])
            metrics[f'recall@{k}'] = np.mean(recall_scores[k])
        else:
            metrics[f'precision@{k}'] = 0.0
            metrics[f'recall@{k}'] = 0.0
    
    return metrics


def compare_search_methods(
    feature_db: Dict[str, np.ndarray],
    num_queries: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Compare cosine similarity vs Euclidean distance.
    
    Returns:
        Dictionary with metrics for each method
    """
    print("\n=== Evaluating Cosine Similarity ===")
    cosine_metrics = evaluate_retrieval_system(
        feature_db, search_cosine, num_queries=num_queries
    )
    
    print("\n=== Evaluating Euclidean Distance ===")
    euclidean_metrics = evaluate_retrieval_system(
        feature_db, search_euclidean, num_queries=num_queries
    )
    
    return {
        'cosine': cosine_metrics,
        'euclidean': euclidean_metrics
    }


def print_evaluation_report(comparison: Dict[str, Dict[str, float]]):
    """
    Print a formatted evaluation report.
    """
    print("\n" + "="*60)
    print("RETRIEVAL SYSTEM EVALUATION REPORT")
    print("="*60)
    
    for method_name, metrics in comparison.items():
        print(f"\n{method_name.upper()} SIMILARITY:")
        print("-" * 40)
        print(f"  mAP:                  {metrics['mAP']:.4f}")
        print(f"  Avg Search Time:      {metrics['avg_search_time']*1000:.2f} ms")
        print(f"  Median Search Time:   {metrics['median_search_time']*1000:.2f} ms")
        print("\n  Precision@K:")
        for k in [1, 5, 10, 20]:
            if f'precision@{k}' in metrics:
                print(f"    P@{k:2d}:  {metrics[f'precision@{k}']:.4f}")
        print("\n  Recall@K:")
        for k in [1, 5, 10, 20]:
            if f'recall@{k}' in metrics:
                print(f"    R@{k:2d}:  {metrics[f'recall@{k}']:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Load feature database
    print("Loading feature database...")
    db = load_feature_db()
    print(f"Loaded {len(db)} images")
    
    # Run comparison
    results = compare_search_methods(db, num_queries=100)
    
    # Print report
    print_evaluation_report(results)
    
    # Save results
    output_path = Path("features/evaluation_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
