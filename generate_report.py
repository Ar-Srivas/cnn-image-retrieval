"""
Generate System Evaluation Report
Runs comprehensive evaluation and produces a detailed report.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from src.evaluate import (
    load_feature_db,
    compare_search_methods,
    print_evaluation_report
)
from src.model_comparison import (
    compare_models,
    print_model_comparison_report
)


def generate_text_report(
    search_comparison: dict,
    model_comparison: dict,
    output_file: str = "evaluation_report.txt"
):
    """
    Generate a comprehensive text report.
    """
    with open(output_file, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("CNN IMAGE RETRIEVAL SYSTEM - EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Section 1: Search Method Comparison
        f.write("SECTION 1: SEARCH METHOD COMPARISON\n")
        f.write("-"*80 + "\n\n")
        f.write("This section compares Cosine Similarity vs Euclidean Distance.\n\n")
        
        for method_name, metrics in search_comparison.items():
            f.write(f"{method_name.upper()} SIMILARITY:\n")
            f.write(f"  Mean Average Precision (mAP):  {metrics['mAP']:.4f}\n")
            f.write(f"  Average Search Time:            {metrics['avg_search_time']*1000:.2f} ms\n")
            f.write(f"  Median Search Time:             {metrics['median_search_time']*1000:.2f} ms\n\n")
            
            f.write("  Precision@K:\n")
            for k in [1, 5, 10, 20]:
                if f'precision@{k}' in metrics:
                    f.write(f"    P@{k:2d}:  {metrics[f'precision@{k}']:.4f}\n")
            
            f.write("\n  Recall@K:\n")
            for k in [1, 5, 10, 20]:
                if f'recall@{k}' in metrics:
                    f.write(f"    R@{k:2d}:  {metrics[f'recall@{k}']:.4f}\n")
            f.write("\n")
        
        # Determine better search method
        if search_comparison['cosine']['mAP'] > search_comparison['euclidean']['mAP']:
            better_method = "Cosine Similarity"
            better_map = search_comparison['cosine']['mAP']
        else:
            better_method = "Euclidean Distance"
            better_map = search_comparison['euclidean']['mAP']
        
        f.write(f"RECOMMENDATION: {better_method} performs better (mAP: {better_map:.4f})\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Section 2: Model Comparison
        if model_comparison:
            f.write("SECTION 2: FEATURE EXTRACTOR COMPARISON\n")
            f.write("-"*80 + "\n\n")
            f.write("This section compares different CNN architectures for feature extraction.\n\n")
            
            # Summary table
            f.write(f"{'Model':<20} {'mAP':<10} {'P@5':<10} {'R@5':<10} {'Feat Dim':<12} {'Time (ms)':<12}\n")
            f.write("-"*80 + "\n")
            
            for model_name, data in model_comparison.items():
                metrics = data['metrics']
                feat_dim = data['feature_dim']
                search_time = metrics['avg_search_time'] * 1000
                
                f.write(f"{model_name:<20} {metrics['mAP']:<10.4f} "
                       f"{metrics.get('precision@5', 0):<10.4f} "
                       f"{metrics.get('recall@5', 0):<10.4f} "
                       f"{feat_dim:<12} "
                       f"{search_time:<12.2f}\n")
            
            f.write("\n")
            
            # Detailed metrics
            f.write("DETAILED MODEL METRICS:\n")
            f.write("-"*80 + "\n\n")
            
            for model_name, data in model_comparison.items():
                metrics = data['metrics']
                f.write(f"{model_name}:\n")
                f.write(f"  Feature Dimension:    {data['feature_dim']}\n")
                f.write(f"  Images Processed:     {data['num_images']}\n")
                f.write(f"  mAP:                  {metrics['mAP']:.4f}\n")
                f.write(f"  Average Search Time:  {metrics['avg_search_time']*1000:.2f} ms\n")
                f.write(f"  Precision@5:          {metrics.get('precision@5', 0):.4f}\n")
                f.write(f"  Precision@10:         {metrics.get('precision@10', 0):.4f}\n")
                f.write(f"  Recall@5:             {metrics.get('recall@5', 0):.4f}\n")
                f.write(f"  Recall@10:            {metrics.get('recall@10', 0):.4f}\n\n")
            
            # Find best models
            best_accuracy = max(model_comparison.items(), 
                              key=lambda x: x[1]['metrics']['mAP'])
            fastest = min(model_comparison.items(),
                        key=lambda x: x[1]['metrics']['avg_search_time'])
            
            f.write("RECOMMENDATIONS:\n")
            f.write(f"  Best Accuracy:  {best_accuracy[0]} ")
            f.write(f"(mAP: {best_accuracy[1]['metrics']['mAP']:.4f})\n")
            f.write(f"  Fastest Search: {fastest[0]} ")
            f.write(f"({fastest[1]['metrics']['avg_search_time']*1000:.2f} ms)\n\n")
            
            f.write("="*80 + "\n\n")
        
        # Section 3: Key Findings
        f.write("SECTION 3: KEY FINDINGS & CONCLUSIONS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("1. Search Method:\n")
        f.write(f"   - {better_method} is recommended for this dataset\n")
        f.write(f"   - Achieves mAP of {better_map:.4f}\n\n")
        
        if model_comparison:
            f.write("2. Feature Extractor:\n")
            f.write(f"   - {best_accuracy[0]} provides the highest accuracy\n")
            f.write(f"   - {fastest[0]} offers the fastest retrieval speed\n")
            f.write(f"   - Trade-off between accuracy and speed should be considered\n\n")
        
        f.write("3. Performance Metrics:\n")
        f.write("   - Precision@K measures how many retrieved images are relevant\n")
        f.write("   - Recall@K measures how many relevant images were retrieved\n")
        f.write("   - mAP provides an overall quality measure of the ranking\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


def generate_json_report(
    search_comparison: dict,
    model_comparison: dict,
    output_file: str = "evaluation_report.json"
):
    """
    Generate a JSON report for programmatic access.
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'search_method_comparison': search_comparison,
        'model_comparison': model_comparison,
        'recommendations': {}
    }
    
    # Add recommendations
    if search_comparison['cosine']['mAP'] > search_comparison['euclidean']['mAP']:
        report['recommendations']['search_method'] = 'cosine'
    else:
        report['recommendations']['search_method'] = 'euclidean'
    
    if model_comparison:
        best_model = max(model_comparison.items(), 
                        key=lambda x: x[1]['metrics']['mAP'])
        report['recommendations']['best_model'] = best_model[0]
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)


def run_full_evaluation(
    use_existing_features: bool = True,
    run_model_comparison: bool = False,
    num_eval_queries: int = 100
):
    """
    Run complete evaluation pipeline.
    
    Args:
        use_existing_features: Use existing feature database
        run_model_comparison: Whether to run model comparison (time-consuming)
        num_eval_queries: Number of queries for evaluation
    """
    print("="*80)
    print("STARTING COMPREHENSIVE SYSTEM EVALUATION")
    print("="*80)
    
    # Step 1: Load existing features
    print("\n[1/3] Loading feature database...")
    if use_existing_features:
        try:
            db = load_feature_db()
            print(f"✓ Loaded {len(db)} image features")
        except Exception as e:
            print(f"✗ Error loading features: {e}")
            print("Please run feature extraction first: python run_feature_extraction.py")
            return
    
    # Step 2: Compare search methods
    print("\n[2/3] Evaluating search methods...")
    search_comparison = compare_search_methods(db, num_queries=num_eval_queries)
    print_evaluation_report(search_comparison)
    
    # Step 3: Compare models (optional)
    model_comparison = {}
    if run_model_comparison:
        print("\n[3/3] Comparing CNN models...")
        print("WARNING: This will take significant time!")
        
        try:
            model_comparison = compare_models(
                max_images=1000,
                num_eval_queries=num_eval_queries
            )
            print_model_comparison_report(model_comparison)
        except Exception as e:
            print(f"Warning: Model comparison failed: {e}")
    else:
        print("\n[3/3] Skipping model comparison (use --compare-models to enable)")
    
    # Generate reports
    print("\n" + "="*80)
    print("GENERATING REPORTS...")
    print("="*80)
    
    output_dir = Path("features")
    output_dir.mkdir(exist_ok=True)
    
    # Text report
    text_path = output_dir / "evaluation_report.txt"
    generate_text_report(search_comparison, model_comparison, str(text_path))
    print(f"✓ Text report saved to: {text_path}")
    
    # JSON report
    json_path = output_dir / "evaluation_report.json"
    generate_json_report(search_comparison, model_comparison, str(json_path))
    print(f"✓ JSON report saved to: {json_path}")
    
    # Save raw data
    results_path = output_dir / "evaluation_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'search_comparison': search_comparison,
            'model_comparison': model_comparison
        }, f)
    print(f"✓ Raw results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nView the detailed report at: {text_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Run model comparison (time-consuming)'
    )
    parser.add_argument(
        '--queries',
        type=int,
        default=100,
        help='Number of test queries (default: 100)'
    )
    
    args = parser.parse_args()
    
    run_full_evaluation(
        use_existing_features=True,
        run_model_comparison=args.compare_models,
        num_eval_queries=args.queries
    )
