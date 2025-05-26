# test_config.py
"""
Configuration file for RAG Recipe Testing
Modify these settings to customize your tests
"""

# Test Configuration
TEST_CONFIG = {
    # Fraction of the database to use for testing (0.1 = 10%)
    "database_fraction": 0.1,

    # API configuration
    "api_url": "http://localhost:11434/api/chat",

    # Paths
    "vector_store_path": "C:/Projects/Recipe-Finder/backend/recipe_finder_index",
    "results_folder": "test_results",

    # Test parameters
    "top_k_recipes": 3,  # Number of recipes to retrieve per query

    # Evaluation thresholds (0.0 to 1.0)
    "thresholds": {
        "correctness": 0.6,
        "relevance": 0.7,
        "helpfulness": 0.6
    }
}

# Test Scenarios - Add or modify these to test different aspects
TEST_SCENARIOS = [
    {
        "query": "Quick breakfast with eggs",
        "expected_keywords": ["egg", "breakfast", "quick", "fast"],
        "expected_cuisine": None,
        "description": "Should find quick egg-based breakfast recipes",
        "category": "breakfast"
    },
    {
        "query": "Italian pasta with tomatoes",
        "expected_keywords": ["pasta", "tomato", "italian"],
        "expected_cuisine": "Italian",
        "description": "Should find Italian pasta recipes with tomatoes",
        "category": "dinner"
    },
    {
        "query": "Healthy chicken salad",
        "expected_keywords": ["chicken", "salad", "healthy"],
        "expected_cuisine": None,
        "description": "Should find healthy chicken salad recipes",
        "category": "lunch"
    },
    {
        "query": "Vegetarian dinner with beans",
        "expected_keywords": ["vegetarian", "bean", "dinner"],
        "expected_cuisine": None,
        "description": "Should find vegetarian bean-based dinner recipes",
        "category": "dinner"
    },
    {
        "query": "Chocolate dessert",
        "expected_keywords": ["chocolate", "dessert", "sweet"],
        "expected_cuisine": None,
        "description": "Should find chocolate dessert recipes",
        "category": "dessert"
    },
    {
        "query": "Spicy Mexican food",
        "expected_keywords": ["spicy", "mexican"],
        "expected_cuisine": "Mexican",
        "description": "Should find spicy Mexican recipes",
        "category": "dinner"
    },
    {
        "query": "Low carb keto meal",
        "expected_keywords": ["low carb", "keto", "protein"],
        "expected_cuisine": None,
        "description": "Should find low-carb/keto friendly recipes",
        "category": "diet"
    },
    {
        "query": "Asian stir fry with vegetables",
        "expected_keywords": ["asian", "stir fry", "vegetable"],
        "expected_cuisine": "Asian",
        "description": "Should find Asian vegetable stir fry recipes",
        "category": "dinner"
    },
    {
        "query": "Simple soup for cold weather",
        "expected_keywords": ["soup", "warm", "comfort"],
        "expected_cuisine": None,
        "description": "Should find warming soup recipes",
        "category": "soup"
    },
    {
        "query": "Gluten free bread recipe",
        "expected_keywords": ["gluten free", "bread", "flour"],
        "expected_cuisine": None,
        "description": "Should find gluten-free bread recipes",
        "category": "baking"
    }
]

# Evaluation Criteria - Customize these for your specific needs
EVALUATION_CRITERIA = {
    "correctness": "Determine if the 'actual output' correctly addresses the user's recipe query and provides relevant recipe recommendations based on the expected output.",

    "relevance": "Evaluate if the recipe recommendations are highly relevant to the user's specific ingredients, cuisine preferences, and dietary requirements mentioned in the input query.",

    "helpfulness": "Assess if the response provides helpful cooking advice, tips, substitutions, or additional context that would be valuable to someone looking to cook the recommended recipes.",

    "completeness": "Check if the response includes sufficient detail about the recommended recipes, including key ingredients, cooking methods, or preparation notes.",

    "clarity": "Evaluate if the response is well-structured, easy to understand, and provides clear recommendations without being overly verbose or confusing."
}