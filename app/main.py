import os
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.settings import Settings

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- 1. Load Environment and Configuration ---
load_dotenv()

# Ensure your Google API Key is set in a .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Define paths
CHROMA_PERSIST_DIR = "../chroma_db" # Relative to the app directory
print(f"ChromaDB persistence directory: {os.path.abspath(CHROMA_PERSIST_DIR)}")


# --- 2. FastAPI App Initialization ---
app = FastAPI(
    title="Persona AI",
    description="Ask questions to a persona powered by a knowledge base.",
    version="1.0.0",
)

# Pydantic model for the request body
class AskRequest(BaseModel):
    question: str

# --- 3. LlamaIndex Setup (Done globally on startup) ---
query_engine = None

@app.on_event("startup")
def startup_event():
    global query_engine
    print("Server starting up...")

    # Configure global LlamaIndex settings
    print("Configuring global settings...")
    Settings.llm = GoogleGenAI(model_name="gemini-2.5-flash")
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004"
    )
    print("LLM and embedding models configured.")

    # Load the existing vector store
    print("Loading vector store...")
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("persona_ai_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    print("Index loaded from vector store.")

    # Define the Persona Prompt
    # This is the core instruction that gives the AI its personality and constraints.
    persona_prompt_template = '''
    You are Mel Robbins. Your tone is direct, motivational, and empowering. You are known for 'The 5 Second Rule'.
    
    Answer the user's question based *only* on the context provided below. Do not use any outside knowledge or your general training.
    
    If the provided context does not contain the answer to the question, you must state: "That's a great question, but it's not something I cover in the material you've given me."
    
    ---------------------
    Context: {context_str}
    ---------------------
    Question: {query_str}
    ---------------------
    Answer:
    '''
    qa_template = PromptTemplate(persona_prompt_template)

    # Create the query engine with the custom persona prompt
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=3  # Retrieve top 3 most relevant chunks
    )
    print("Query engine created with persona prompt.")
    print("--- Startup complete. Application is ready. ---")


# --- 4. Serve Static Files and Root Endpoint ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("app/index.html")

# --- 5. API Endpoint ---
@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Receives a question, queries the knowledge base, and returns an answer
    in the persona of the knowledge source.
    """
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine is not available. The server may still be starting up.")
    
    print(f"Received question: {request.question}")
    
    try:
        response = query_engine.query(request.question)
        
        # The response object has the answer in `response.response`
        # and the source nodes in `response.source_nodes`
        answer = response.response
        
        print(f"Retrieved context: {response.source_nodes}")
        print(f"Generated answer: {answer}")
        return {"answer": answer}

    except Exception as e:
        print(f"An error occurred while querying: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the question.")

# To run this app:
# 1. Make sure you are in the `persona-ai` directory.
# 2. Run the command: `source .venv/bin/activate`
# 3. Run the command: `uvicorn app.main:app --reload`
