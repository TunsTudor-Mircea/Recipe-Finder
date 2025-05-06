from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.recipe_finder import RecipeFinder
from backend.recipe_rag_stage4 import RecipeGenerator

# Initialize FastAPI
app = FastAPI()

# Enable CORS so React (on port 3000) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector store and initialize RAG model
recipe_finder = RecipeFinder()
recipe_finder.load_vector_store("backend/recipe_finder_index")
recipe_generator = RecipeGenerator(api_url="http://localhost:11434/api/chat")


# Request body model
class QueryRequest(BaseModel):
    query: str


# Endpoint: Generate personalized recipe suggestions
@app.post("/generate-recipe")
def generate_recipe(request_data: QueryRequest):
    query = request_data.query
    try:
        retrieved_docs = recipe_finder.retrieve_recipes(query, top_k=3)
        response = recipe_generator.generate_response(query, retrieved_docs)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
