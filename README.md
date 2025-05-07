# Recipe-Finder

AI-powered recipe finder that uses RAG to suggest personalized meals based on your ingredients and preferences.

## 1. Project Overview

#### Goal:

Allow users to ask questions like:

- “What can I cook with tomatoes and eggs?”

- “What’s a good dessert using bananas?”

#### Core Features:

- Upload or scrape a recipe dataset.

- Embed and index recipes for semantic search.

- Use a language model to generate responses based on relevant recipe chunks.

![Example](https://github.com/TunsTudor-Mircea/Recipe-Finder/blob/main/example_usage.png?raw=true)

## 2. Tech Stack

Embeddings: **BAAI/bge-small-en-v1.5 (HuggingFace)**

Vector Store: **FAISS**

Retriever: **LangChain**

LLM: **Google Gemma 3**

Frontend: **React**

## 3. Dataset - **RecipeNLG**

223k+ structured recipes with ingredients, instructions, cuisines, diet types, etc.

"RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation"

url = https://recipenlg.cs.put.poznan.pl

## 4. Development Steps
✅ Step 1: Load and Preprocess the Data
Parse each recipe into:

title, ingredients, instructions, cuisine, diet, time, etc.

Create a "chunk" per recipe (or split if very long).

✅ Step 2: Embed and Index
Generate embeddings for each recipe chunk.

Store them in a vector DB.

Add metadata (e.g., time, cuisine) for filtering.

✅ Step 3: Retrieval Logic
User enters a query like:

“Quick vegetarian lunch ideas using lentils”

Embed the query and search for top-k similar recipes.

Optionally apply metadata filters (e.g., cuisine = "Indian").

✅ Step 4: Generation
Feed the top results into an LLM prompt like:
```
User wants a quick vegetarian lunch using lentils.
Here are 3 relevant recipes:
- Title: Lentil Salad | Ingredients: ... | Instructions: ...
- ...
Suggest the best option and explain why.
```
LLM returns a summary or recommendation.
