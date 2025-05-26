"""
Simple test runner for RAG Recipe Testing
Usage: python run_tests.py [--fraction 0.1] [--api-url http://localhost:11434/api/chat]
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from test_rag_recipes import RecipeRAGTester


def main():
    parser = argparse.ArgumentParser(description="Run RAG Recipe Tests")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of database to use for testing (default: 0.1)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:11434/api/chat",
        help="API URL for recipe generator (default: http://localhost:11434/api/chat)"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="C:/Projects/Recipe-Finder/backend/recipe_finder_index",
        help="Path to vector store (default: C:/Projects/Recipe-Finder/backend/recipe_finder_index)"
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default="test_results",
        help="Folder to save results (default: test_results)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"🧪 Starting RAG Recipe Tests")
    print(f"   Database Fraction: {args.fraction * 100:.1f}%")
    print(f"   API URL: {args.api_url}")
    print(f"   Vector Store: {args.vector_store}")
    print(f"   Results Folder: {args.results_folder}")
    print("-" * 60)

    try:
        # Check if vector store exists
        if not os.path.exists(args.vector_store):
            print(f"❌ Error: Vector store not found at {args.vector_store}")
            print("   Make sure you've run your recipe_finder.py to create the index first.")
            sys.exit(1)

        # Initialize and run tests
        tester = RecipeRAGTester(
            vector_store_path=args.vector_store,
            api_url=args.api_url,
            results_folder=args.results_folder
        )

        results, report = tester.run_full_test_suite(fraction=args.fraction)

        print("\n🎉 Testing completed successfully!")
        print(f"📊 Results saved in: {args.results_folder}/")

        # Print quick summary
        total = results['total_tests']
        passed = results['passed_tests']
        failed = results['failed_tests']

        print(f"📈 Quick Summary: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

        if failed > 0:
            print(f"⚠️  {failed} tests failed - check the detailed report for more info")
            sys.exit(1)
        else:
            print("✅ All tests passed!")

    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()