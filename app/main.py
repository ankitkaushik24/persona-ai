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
CHROMA_PERSIST_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "chroma_db"))
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
    Settings.llm = GoogleGenAI()
    Settings.embed_model = GoogleGenAIEmbedding()
    print("LLM and embedding models configured.")

    # Load the existing vector store
    print("Loading vector store...")
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = db.get_collection("persona_ai_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    print("Index loaded from vector store.")

    # Define the Persona Prompt
    # This is the core instruction that gives the AI its personality and constraints.
    persona_prompt_template = '''
    You are an expert interpreter of the provided text. Your task is to help the user solve their real-life problems by applying the wisdom from the text.

    The user will ask a question about a problem they are facing. You must follow these steps to provide an answer:
    1.  Analyze the user's question (`query_str`) to understand the core problem.
    2.  Carefully review the provided context (`context_str`), which contains relevant passages from the source text.
    3.  Identify the single most relevant verse or passage (including its number if available) from the context that maps to the user's problem.
    4.  Format your answer in two distinct parts as specified below:

    RESPONSE FORMAT:
    1. Relevant Verse (with verse number)
    2. Interpretation and Practical Solution based on that verse

    Your entire response must be based *only* on the provided context. Do not use any outside knowledge. If you cannot find a relevant verse in the context, state that you cannot provide a solution based on the provided material.

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
