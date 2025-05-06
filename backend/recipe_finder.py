import ast
import json
import logging
import os
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS as LangchainFAISS
from tqdm.auto import tqdm as tqdm_auto

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecipeFinder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the Recipe Finder with the specified embedding model and use GPU if available."""
        logger.info(f"Initializing RecipeFinder with model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        logger.info(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
        self.vector_store = None
        self.recipes_df = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load recipe data from a file.

        Args:
            file_path: Path to the data file (JSON or CSV)

        Returns:
            pandas DataFrame containing recipe data
        """
        logger.info(f"Loading data from {file_path}")

        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    # Handle the RecipeNLG format specifically
                    recipes = []
                    for key, recipe in data.items():
                        recipe['id'] = key
                        recipes.append(recipe)
                    df = pd.DataFrame(recipes)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0)
            # Handle the specific CSV format from your example
            if 'ingredients' in df.columns and isinstance(df['ingredients'].iloc[0], str):
                logger.info("Converting string representations of lists to actual lists")
                try:
                    # Handle ingredients column (safely evaluate string representations of lists)
                    df['ingredients'] = df['ingredients'].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

                    # Handle directions/instructions column if present
                    if 'directions' in df.columns:
                        df['instructions'] = df['directions'].apply(
                            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

                    # Handle NER column if present
                    if 'NER' in df.columns:
                        df['NER'] = df['NER'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                except Exception as e:
                    logger.warning(f"Error parsing string lists: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        logger.info(f"Loaded {len(df)} recipes")
        self.recipes_df = df
        return df

    def preprocess_recipes(self, df: Optional[pd.DataFrame] = None) -> List[Document]:
        """
        Preprocess recipes into document format suitable for embedding.

        Args:
            df: DataFrame containing recipes, or uses self.recipes_df if None

        Returns:
            List of Document objects ready for embedding
        """
        if df is None:
            if self.recipes_df is None:
                raise ValueError("No recipe data loaded. Call load_data first.")
            df = self.recipes_df

        logger.info("Preprocessing recipes into documents")
        documents = []

        for idx, row in df.iterrows():
            try:
                # Extract and clean recipe components
                title = row.get('title', '')

                # Handle ingredients (could be list or string representation of list)
                ingredients = row.get('ingredients', [])

                # Handle instructions (could be under 'instructions' or 'directions')
                instructions = row.get('instructions', row.get('directions', ''))
                if isinstance(instructions, str) and not instructions.strip():
                    instructions = "No instructions provided."

                # Get or set default metadata fields
                cuisine = row.get('cuisine', 'Not specified')
                diet_type = row.get('diet', 'Not specified')
                cooking_time = row.get('time', 'Not specified')
                source = row.get('source', 'Not specified')

                # Create document content
                content = f"Title: {title}\n"
                content += f"Source: {source}\n"
                content += f"Cuisine: {cuisine}\n"
                content += f"Diet Type: {diet_type}\n"
                content += f"Cooking Time: {cooking_time}\n"
                content += "Ingredients:\n"

                if isinstance(ingredients, list):
                    for ing in ingredients:
                        content += f"- {ing}\n"
                else:
                    content += f"- {ingredients}\n"

                content += "\nInstructions:\n"
                if isinstance(instructions, list):
                    for i, step in enumerate(instructions, 1):
                        content += f"{i}. {step}\n"
                else:
                    content += instructions

                # Create document with metadata
                metadata = {
                    'title': title,
                    'cuisine': cuisine,
                    'diet_type': diet_type,
                    'cooking_time': cooking_time,
                    'ingredients': ingredients if isinstance(ingredients, list) else [ingredients],
                    'source': source,
                    'recipe_id': row.get('id', str(idx))
                }

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            except Exception as e:
                logger.error(f"Error processing recipe at index {idx}: {e}")

        logger.info(f"Created {len(documents)} documents")
        return documents

    def create_vector_store(self, documents: List[Document], batch_size: int = 220,
                            use_tqdm: bool = True) -> LangchainFAISS:
        """
        Create a FAISS vector store from the documents with optimized batch processing.

        Args:
            documents: List of Document objects
            batch_size: Number of documents to process in each batch
            use_tqdm: Whether to show progress bar (requires tqdm package)

        Returns:
            FAISS vector store
        """
        logger.info(f"Creating FAISS vector store with batch size: {batch_size}")

        if not documents:
            raise ValueError("No documents provided for vector store creation")

        # Extract text and metadata from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Prepare batches
        total_docs = len(texts)
        batch_indices = list(range(0, total_docs, batch_size))

        logger.info(f"Processing {total_docs} documents in {len(batch_indices)} batches")

        # Apply tqdm for progress tracking if requested
        if use_tqdm:
            batch_indices = tqdm_auto(batch_indices, desc="Processing document batches")

        # Process first batch to initialize the vector store
        end_idx = min(batch_size, total_docs)

        with torch.no_grad():  # Disable gradient calculation for inference
            vector_store = LangchainFAISS.from_texts(
                texts=texts[0:end_idx],
                embedding=self.embeddings,
                metadatas=metadatas[0:end_idx] if metadatas else None
            )

        # Process remaining batches
        for start_idx in batch_indices:
            # Skip the first batch as it's already processed
            if start_idx == 0:
                continue

            end_idx = min(start_idx + batch_size, total_docs)
            batch_texts = texts[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx] if metadatas else None

            # Add this batch to the vector store
            with torch.no_grad():  # Disable gradient calculation for inference
                vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )

        self.vector_store = vector_store
        logger.info(f"Vector store created successfully with {total_docs} documents")

        return vector_store

    def save_vector_store(self, directory: str = "recipe_finder_index"):
        """
        Save the vector store to disk.

        Args:
            directory: Directory to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create it first.")

        logger.info(f"Saving vector store to {directory}")
        self.vector_store.save_local(directory)
        logger.info("Vector store saved successfully")

    def load_vector_store(self, directory: str = "recipe_finder_index") -> LangchainFAISS:
        """
        Load the vector store from disk.

        Args:
            directory: Directory containing the vector store

        Returns:
            FAISS vector store
        """
        logger.info(f"Loading vector store from {directory}")

        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")

        vector_store = LangchainFAISS.load_local(
            folder_path=directory,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.vector_store = vector_store
        logger.info("Vector store loaded successfully")
        return vector_store

    def retrieve_recipes(self, query: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[
        Document]:
        """
        Retrieve recipes based on query and optional filters.

        Args:
            query: User query
            top_k: Number of recipes to retrieve
            filter_dict: Dictionary of metadata filters

        Returns:
            List of Document objects
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load it first.")

        logger.info(f"Retrieving recipes for query: '{query}'")

        # Perform similarity search with optional filters
        if filter_dict:
            filter_str = " AND ".join([f"{k}='{v}'" for k, v in filter_dict.items()])
            logger.info(f"Applying filters: {filter_str}")
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k
            )

        logger.info(f"Retrieved {len(results)} recipes")
        return results


def main():
    # Example usage
    recipe_finder = RecipeFinder()

    # Step 1: Load data
    try:
        df = recipe_finder.load_data("full_dataset.csv")
        df = df[df['source'] == 'Gathered'].copy()  # Only keep gathered recipes
        print(f"Filtered to {len(df)} gathered recipes")

        # Step 2: Preprocess recipes
        documents = recipe_finder.preprocess_recipes(df)
        print(f"Created {len(documents)} documents")

        # Step 3: Create vector store
        vector_store = recipe_finder.create_vector_store(documents)

        # Save vector store for later use
        recipe_finder.save_vector_store("recipe_finder_index")

        # Example query
        print("\nExample query:")
        results = recipe_finder.retrieve_recipes("quick breakfast with eggs")

        for i, doc in enumerate(results):
            print(f"\nRecipe {i + 1}:")
            print(f"Title: {doc.metadata['title']}")
            print(f"Source: {doc.metadata.get('source', 'Not specified')}")
            print("-" * 40)

            # Print a sample of ingredients
            ingredients = doc.metadata.get('ingredients', [])
            if ingredients:
                print("Sample ingredients:")
                for ing in ingredients[:3]:
                    print(f"- {ing}")
                if len(ingredients) > 3:
                    print(f"...and {len(ingredients) - 3} more ingredients")

            print("-" * 40)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()