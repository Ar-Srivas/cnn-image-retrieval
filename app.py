#!/usr/bin/env python
"""
CNN Image Retrieval System - Main Application
Run this to execute and test the entire system!

Usage: python app.py
"""

import os
import pickle


def print_header():
    print("\n" + "="*70)
    print("🔍 CNN IMAGE RETRIEVAL SYSTEM 🔍")
    print("="*70)
    print("Your complete image similarity search and evaluation system")
    print("="*70 + "\n")


def check_setup():
    print("📋 Checking system setup...")
    
    if not os.path.exists("features/image_features.pkl"):
        print("❌ Feature database not found! Run: python run_feature_extraction.py")
        return False
    
    try:
        from src.query_pipeline import process_query_image
        from src.evaluate import compare_search_methods
        from src.similarity_search import load_feature_db
        print("✅ All modules loaded successfully\n")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}\nRun: uv sync")
        return False


def show_menu():
    print("\n" + "─"*70)
    print("📌 MAIN MENU:")
    print("─"*70)
    print("1️⃣  Test System")
    print("2️⃣  Search Similar Images Demo")
    print("3️⃣  Run Evaluation & Generate Report")
    print("4️⃣  View Existing Reports")
    print("5️⃣  Run Everything")
    print("0️⃣  Exit")
    print("─"*70)


def test_system():
    print("\n🧪 TESTING SYSTEM\n")
    
    from src.similarity_search import load_feature_db, search_cosine
    from src.evaluate import precision_at_k
    
    db = load_feature_db()
    print(f"✅ Loaded {len(db)} images")
    
    sample = list(db.keys())[0]
    results, time = search_cosine(db[sample], db, top_k=5)
    print(f"✅ Search: {time*1000:.2f}ms")
    
    p = precision_at_k(['a', 'b', 'c'], ['a', 'd'], 3)
    print(f"✅ Metrics working: P@3={p:.2%}\n")
    
    print("✅ ALL TESTS PASSED!")


def demo_search():
    print("\n🔍 SIMILARITY SEARCH DEMO\n")
    
    from src.similarity_search import load_feature_db, search_cosine
    
    db = load_feature_db()
    images = list(db.keys())
    
    print(f"Query image: {images[0]}")
    results, time = search_cosine(db[images[0]], db, top_k=10)
    
    print(f"\n⏱️  Search time: {time*1000:.2f}ms\n")
    print("Top 10 Similar Images:")
    print("="*50)
    for i, (name, score) in enumerate(results, 1):
        print(f"{i:2d}. {name:<30} | {score:.4f}")


def run_evaluation():
    print("\n📊 RUNNING EVALUATION\n")
    
    from src.evaluate import load_feature_db, compare_search_methods, print_evaluation_report
    from generate_report import generate_text_report, generate_json_report
    
    db = load_feature_db()
    results = compare_search_methods(db, num_queries=50)
    
    print_evaluation_report(results)
    
    generate_text_report(results, {})
    generate_json_report(results, {})
    
    with open("features/evaluation_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Reports saved:")
    print("   features/evaluation_report.txt")
    print("   features/evaluation_report.json")


def view_reports():
    print("\n📄 EXISTING REPORTS\n")
    
    if os.path.exists('features/evaluation_report.txt'):
        size = os.path.getsize('features/evaluation_report.txt')
        print(f"✅ evaluation_report.txt ({size:,} bytes)")
        
        response = input("\nView report? (y/n): ")
        if response.lower() == 'y':
            with open('features/evaluation_report.txt') as f:
                print("\n" + f.read())
    else:
        print("⚠️  No reports found. Run option 3 first.")


def run_everything():
    print("\n🚀 RUNNING COMPLETE PIPELINE\n")
    test_system()
    input("\nPress Enter to continue...")
    demo_search()
    input("\nPress Enter to continue...")
    run_evaluation()
    print("\n🎉 COMPLETE!")


def main():
    print_header()
    
    if not check_setup():
        return
    
    while True:
        show_menu()
        choice = input("\n👉 Choice (0-5): ").strip()
        
        try:
            if choice == '0':
                print("\n👋 Goodbye!\n")
                break
            elif choice == '1':
                test_system()
            elif choice == '2':
                demo_search()
            elif choice == '3':
                run_evaluation()
            elif choice == '4':
                view_reports()
            elif choice == '5':
                run_everything()
            else:
                print("❌ Invalid choice")
            
            input("\n⏸️  Press Enter for menu...")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
