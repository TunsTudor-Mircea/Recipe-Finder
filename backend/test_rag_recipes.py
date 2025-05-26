import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain.schema import Document

# Import your existing classes
from backend.recipe_finder import RecipeFinder
from backend.recipe_rag_stage4 import RecipeGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecipeRAGTester:
    def __init__(self,
                 vector_store_path: str = "C:/Projects/Recipe-Finder/backend/recipe_finder_index",
                 api_url: str = "http://localhost:11434/api/chat",
                 results_folder: str = "test_results"):
        """
        Initialize the Recipe RAG Tester.

        Args:
            vector_store_path: Path to the vector store
            api_url: URL for the recipe generator API
            results_folder: Folder to save test results
        """
        self.vector_store_path = vector_store_path
        self.api_url = api_url
        self.results_folder = results_folder

        # Create results folder if it doesn't exist
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.recipe_finder = RecipeFinder()
        self.recipe_generator = RecipeGenerator(api_url=api_url)

        # Load vector store
        try:
            self.recipe_finder.load_vector_store(vector_store_path)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    def create_test_cases(self, fraction: float = 0.1) -> List[Dict[str, Any]]:
        """
        Create test cases using a fraction of the vector database.

        Args:
            fraction: Fraction of the database to use for testing (0.1 = 10%)

        Returns:
            List of test case dictionaries
        """
        logger.info(f"Creating test cases using {fraction * 100}% of the database")

        # Sample test queries with expected characteristics
        test_scenarios = [
            {
                "query": "Quick breakfast with eggs",
                "expected_keywords": ["egg", "breakfast", "quick", "fast"],
                "expected_cuisine": None,
                "description": "Should find quick egg-based breakfast recipes"
            },
            {
                "query": "Italian pasta with tomatoes",
                "expected_keywords": ["pasta", "tomato", "italian"],
                "expected_cuisine": "Italian",
                "description": "Should find Italian pasta recipes with tomatoes"
            },
            {
                "query": "Healthy chicken salad",
                "expected_keywords": ["chicken", "salad", "healthy"],
                "expected_cuisine": None,
                "description": "Should find healthy chicken salad recipes"
            },
            {
                "query": "Vegetarian dinner with beans",
                "expected_keywords": ["vegetarian", "bean", "dinner"],
                "expected_cuisine": None,
                "description": "Should find vegetarian bean-based dinner recipes"
            },
            {
                "query": "Chocolate dessert",
                "expected_keywords": ["chocolate", "dessert", "sweet"],
                "expected_cuisine": None,
                "description": "Should find chocolate dessert recipes"
            },
            {
                "query": "Spicy Mexican food",
                "expected_keywords": ["spicy", "mexican"],
                "expected_cuisine": "Mexican",
                "description": "Should find spicy Mexican recipes"
            },
            {
                "query": "Low carb keto meal",
                "expected_keywords": ["low carb", "keto", "protein"],
                "expected_cuisine": None,
                "description": "Should find low-carb/keto friendly recipes"
            },
            {
                "query": "Asian stir fry with vegetables",
                "expected_keywords": ["asian", "stir fry", "vegetable"],
                "expected_cuisine": "Asian",
                "description": "Should find Asian vegetable stir fry recipes"
            }
        ]

        test_cases = []

        for scenario in test_scenarios:
            try:
                # Retrieve recipes for this query
                retrieved_docs = self.recipe_finder.retrieve_recipes(
                    scenario["query"],
                    top_k=int(3 * fraction * 10)  # Adjust based on fraction
                )

                if not retrieved_docs:
                    logger.warning(f"No recipes found for query: {scenario['query']}")
                    continue

                # Generate expected output based on retrieved recipes
                expected_output = self._generate_expected_output(scenario, retrieved_docs)

                # Create retrieval context
                retrieval_context = [doc.page_content for doc in retrieved_docs]

                test_case = {
                    "query": scenario["query"],
                    "expected_output": expected_output,
                    "retrieval_context": retrieval_context,
                    "retrieved_docs": retrieved_docs,
                    "scenario_info": scenario,
                    "test_id": f"test_{len(test_cases) + 1}"
                }

                test_cases.append(test_case)
                logger.info(f"Created test case for: {scenario['query']}")

            except Exception as e:
                logger.error(f"Error creating test case for '{scenario['query']}': {e}")
                continue

        logger.info(f"Created {len(test_cases)} test cases")
        return test_cases

    def _generate_expected_output(self, scenario: Dict, retrieved_docs: List[Document]) -> str:
        """
        Generate expected output based on scenario and retrieved documents.

        Args:
            scenario: Test scenario information
            retrieved_docs: Retrieved recipe documents

        Returns:
            Expected output string
        """
        # Get recipe titles and key information
        recipe_titles = [doc.metadata.get('title', 'Unknown') for doc in retrieved_docs]

        expected = f"Based on your query '{scenario['query']}', here are some great recipe recommendations:\n\n"

        for i, doc in enumerate(retrieved_docs[:2]):  # Focus on top 2 recipes
            title = doc.metadata.get('title', 'Unknown Recipe')
            cuisine = doc.metadata.get('cuisine', 'Not specified')
            expected += f"{i + 1}. {title}"
            if cuisine != 'Not specified':
                expected += f" ({cuisine} cuisine)"
            expected += "\n"

        expected += f"\nThese recipes match your preferences for {', '.join(scenario['expected_keywords'][:3])}."

        return expected

    def run_tests(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run all test cases and collect results.

        Args:
            test_cases: List of test case dictionaries

        Returns:
            Dictionary containing test results
        """
        logger.info(f"Running {len(test_cases)} test cases")

        # Define evaluation metrics
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine if the 'actual output' correctly addresses the user's recipe query and provides relevant recipe recommendations.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.6
        )

        relevance_metric = GEval(
            name="Relevance",
            criteria="Evaluate if the recipe recommendations are relevant to the user's specific ingredients and preferences mentioned in the input query.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.RETRIEVAL_CONTEXT],
            threshold=0.7
        )

        helpfulness_metric = GEval(
            name="Helpfulness",
            criteria="Assess if the response provides helpful cooking advice, tips, or substitutions beyond just listing recipes.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.6
        )

        metrics = [correctness_metric, relevance_metric, helpfulness_metric]

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "metric_scores": {metric.name: [] for metric in metrics}
        }

        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test {i + 1}/{len(test_cases)}: {test_case['test_id']}")

            try:
                # Generate actual output using your RAG system
                actual_output = self.recipe_generator.generate_response(
                    test_case["query"],
                    test_case["retrieved_docs"]
                )

                # Create LLMTestCase
                llm_test_case = LLMTestCase(
                    input=test_case["query"],
                    actual_output=actual_output,
                    expected_output=test_case["expected_output"],
                    retrieval_context=test_case["retrieval_context"]
                )

                # Run the test
                test_result = {
                    "test_id": test_case["test_id"],
                    "query": test_case["query"],
                    "actual_output": actual_output,
                    "expected_output": test_case["expected_output"],
                    "scenario_info": test_case["scenario_info"],
                    "passed": True,
                    "metric_scores": {},
                    "error": None
                }

                try:
                    # Evaluate with each metric
                    for metric in metrics:
                        metric.measure(llm_test_case)
                        score = metric.score
                        test_result["metric_scores"][metric.name] = score
                        results["metric_scores"][metric.name].append(score)

                        # Check if metric passed threshold
                        if score < metric.threshold:
                            test_result["passed"] = False

                    # Run assert_test for overall pass/fail
                    assert_test(llm_test_case, metrics)

                    if test_result["passed"]:
                        results["passed_tests"] += 1
                    else:
                        results["failed_tests"] += 1

                except Exception as eval_error:
                    logger.error(f"Evaluation error for test {test_case['test_id']}: {eval_error}")
                    test_result["passed"] = False
                    test_result["error"] = str(eval_error)
                    results["failed_tests"] += 1

                results["test_results"].append(test_result)

            except Exception as e:
                logger.error(f"Error running test {test_case['test_id']}: {e}")
                results["failed_tests"] += 1
                results["test_results"].append({
                    "test_id": test_case["test_id"],
                    "query": test_case["query"],
                    "passed": False,
                    "error": str(e)
                })

        return results

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save test results to JSON file.

        Args:
            results: Test results dictionary
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_test_results_{timestamp}.json"

        filepath = os.path.join(self.results_folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {filepath}")
        return filepath

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable test report.

        Args:
            results: Test results dictionary

        Returns:
            Report string
        """
        report = f"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║                     RAG RECIPE TESTING REPORT                   ║
        ╚══════════════════════════════════════════════════════════════════╝

        Test Run Timestamp: {results['timestamp']}

        SUMMARY:
        ────────────────────────────────────────────────────────────────────
        Total Tests:  {results['total_tests']}
        Passed:       {results['passed_tests']} ({results['passed_tests'] / results['total_tests'] * 100:.1f}%)
        Failed:       {results['failed_tests']} ({results['failed_tests'] / results['total_tests'] * 100:.1f}%)

        METRIC AVERAGES:
        ────────────────────────────────────────────────────────────────────
        """

        for metric_name, scores in results['metric_scores'].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                report += f"{metric_name:12}: {avg_score:.3f}\n        "

        report += """

        DETAILED RESULTS:
        ────────────────────────────────────────────────────────────────────
        """

        for test in results['test_results']:
            status = "✅ PASS" if test['passed'] else "❌ FAIL"
            report += f"\n        {status} | {test['test_id']} | {test['query'][:50]}..."

            if 'metric_scores' in test:
                for metric, score in test['metric_scores'].items():
                    report += f"\n                {metric}: {score:.3f}"

            if test.get('error'):
                report += f"\n                Error: {test['error']}"

        return report

    def run_full_test_suite(self, fraction: float = 0.1) -> Tuple[Dict[str, Any], str]:
        """
        Run the complete test suite and save results.

        Args:
            fraction: Fraction of database to use for testing

        Returns:
            Tuple of (results_dict, report_string)
        """
        logger.info("Starting full test suite")

        # Create test cases
        test_cases = self.create_test_cases(fraction=fraction)

        if not test_cases:
            raise ValueError("No test cases were created. Check your vector store and data.")

        # Run tests
        results = self.run_tests(test_cases)

        # Save results
        results_file = self.save_results(results)

        # Generate report
        report = self.generate_report(results)

        # Save report
        report_file = os.path.join(self.results_folder, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Test suite completed. Results: {results_file}, Report: {report_file}")
        print(report)

        return results, report


def main():
    """
    Main function to run the RAG recipe testing.
    """
    try:
        # Initialize tester
        tester = RecipeRAGTester(
            vector_store_path="backend/recipe_finder_index",
            api_url="http://localhost:11434/api/chat",
            results_folder="test_results"
        )

        # Run test suite with 10% of the database
        results, report = tester.run_full_test_suite(fraction=0.1)

        print("\n" + "=" * 80)
        print("TEST SUITE COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return results

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()