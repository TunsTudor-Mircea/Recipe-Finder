import os
import json
import requests
from typing import List, Dict, Any
from langchain.schema import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecipeGenerator:
    def __init__(self, api_url: str = "http://localhost:11434/api/chat"):
        """
        Initialize the recipe generator with the Gemma3 API endpoint.

        Args:
            api_url: URL of the Gemma3 API endpoint (default: local Ollama server)
        """
        self.api_url = api_url
        logger.info(f"Initializing Recipe Generator with Gemma3 at {api_url}")

    def format_recipe_for_prompt(self, recipe_doc: Document) -> str:
        """
        Format a recipe document for inclusion in the prompt.

        Args:
            recipe_doc: Document object containing recipe information

        Returns:
            Formatted recipe text
        """
        metadata = recipe_doc.metadata
        title = metadata.get('title', 'Untitled Recipe')
        cuisine = metadata.get('cuisine', 'Not specified')
        diet_type = metadata.get('diet_type', 'Not specified')
        cooking_time = metadata.get('cooking_time', 'Not specified')

        ingredients = metadata.get('ingredients', [])
        if isinstance(ingredients, list):
            ingredients_text = "\n".join([f"- {ing}" for ing in ingredients])
        else:
            ingredients_text = f"- {ingredients}"

        # Extract instructions from page content
        instructions = recipe_doc.page_content.split("Instructions:\n")[-1]

        formatted_recipe = f"""
        === {title} ===
        Cuisine: {cuisine}
        Diet: {diet_type}
        Time: {cooking_time}

        INGREDIENTS:
        {ingredients_text}

        INSTRUCTIONS:
        {instructions}

        """

        return formatted_recipe

    def generate_response(self, query: str, retrieved_recipes: List[Document]) -> str:
        """
        Generate a personalized response based on the query and retrieved recipes.

        Args:
            query: User query
            retrieved_recipes: List of retrieved recipe documents

        Returns:
            Generated response
        """
        logger.info(f"Generating response for query: '{query}'")

        if not retrieved_recipes:
            return "I couldn't find any recipes matching your query. Could you try with different ingredients or preferences?"

        # Format recipes for the prompt
        formatted_recipes = "\n\n".join(
            [self.format_recipe_for_prompt(doc) for doc in retrieved_recipes]
        )

        # Prepare prompt content
        prompt_content = f"""
        You are a helpful cooking assistant that helps people find recipes based on their ingredients and preferences.

        User query: {query}

        Here are the relevant recipes I found:

        {formatted_recipes}

        Please provide a personalized response that:
        1. Recommends the best option(s) for the user's query
        2. Explains why these recipes match their needs
        3. Offers any helpful tips or substitutions
        4. Is conversational and friendly in tone
        """

        # Prepare the payload for the API request
        payload = {
            "model": "gemma3:4b",  # Using the model specified in your example
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
        }

        try:
            # Send the POST request to the API
            response = requests.post(self.api_url, json=payload, stream=True)

            # Check if the request was successful
            if response.status_code == 200:
                # Collect all response content
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_line = json.loads(line)
                        full_response += json_line["message"]["content"]

                logger.info("Generated response successfully")
                return full_response
            else:
                error_msg = f"API request failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"

        except requests.exceptions.RequestException as e:
            error_msg = f"Request exception: {e}"
            logger.error(error_msg)
            return f"Error: {error_msg}"


def demo_recipe_generation(query: str, retrieved_recipes: List[Document],
                           api_url: str = "http://localhost:11434/api/chat"):
    """
    Demo the recipe generation functionality.

    Args:
        query: User query
        retrieved_recipes: List of retrieved recipe documents
        api_url: URL of the Gemma3 API endpoint (default: local Ollama server)
    """
    try:
        recipe_generator = RecipeGenerator(api_url=api_url)
        response = recipe_generator.generate_response(query, retrieved_recipes)

        print("\n" + "=" * 80)
        print("USER QUERY:")
        print(query)
        print("\nGENERATED RESPONSE:")
        print(response)
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error in recipe generation demo: {e}")
        print(f"An error occurred: {e}")


# Example usage (commented out as this would require the Gemma3 API to be running)
if __name__ == "__main__":
    # This example assumes you've already retrieved recipes using RecipeFinder
    from recipe_finder import RecipeFinder

    # Initialize and load vector store
    recipe_finder = RecipeFinder()
    recipe_finder.load_vector_store("recipe_finder_index")

    # Retrieve recipes for a query
    query = "What can I cook with tomatoes and eggs that's quick?"
    retrieved_recipes = recipe_finder.retrieve_recipes(query, top_k=3)

    # Generate response
    demo_recipe_generation(query, retrieved_recipes)