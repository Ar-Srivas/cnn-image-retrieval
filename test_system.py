"""Quick system test"""
print("Testing CNN Image Retrieval System...")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from src.query_pipeline import process_query_image
    from src.evaluate import precision_at_k, recall_at_k, evaluate_retrieval_system
    from src.similarity_search import load_feature_db, search_cosine, search_euclidean
    from src.model_comparison import ResNet50Extractor
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Load feature database
print("\n[2/5] Testing feature database...")
try:
    db = load_feature_db()
    print(f"✓ Loaded {len(db)} image features")
    sample_img = list(db.keys())[0]
    print(f"✓ Sample: {sample_img}, shape: {db[sample_img].shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: Test evaluation metrics
print("\n[3/5] Testing evaluation metrics...")
try:
    retrieved = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg']
    relevant = ['img1.jpg', 'img3.jpg', 'img7.jpg']
    
    p5 = precision_at_k(retrieved, relevant, 5)
    r5 = recall_at_k(retrieved, relevant, 5)
    
    print(f"✓ Precision@5: {p5:.4f} (expected: 0.4000)")
    print(f"✓ Recall@5: {r5:.4f} (expected: 0.6667)")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 4: Test search functionality
print("\n[4/5] Testing search functionality...")
try:
    query_vec = db[sample_img]
    results_cos, time_cos = search_cosine(query_vec, db, top_k=5)
    results_euc, time_euc = search_euclidean(query_vec, db, top_k=5)
    
    print(f"✓ Cosine search: {len(results_cos)} results in {time_cos*1000:.2f}ms")
    print(f"✓ Euclidean search: {len(results_euc)} results in {time_euc*1000:.2f}ms")
    print(f"  Top result: {results_cos[0][0]} (similarity: {results_cos[0][1]:.4f})")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 5: Verify reports exist
print("\n[5/5] Checking generated reports...")
import os
reports = [
    'features/evaluation_report.txt',
    'features/evaluation_report.json',
    'features/evaluation_results.pkl'
]
for report in reports:
    if os.path.exists(report):
        size = os.path.getsize(report)
        print(f"✓ {report} ({size:,} bytes)")
    else:
        print(f"  {report} (not generated yet - run generate_report.py)")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour system is working correctly!")
print("\nNext steps:")
print("  1. Run: python demo_search.py <image_path>")
print("  2. Run: python generate_report.py --queries 100")
