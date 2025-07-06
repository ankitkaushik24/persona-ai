import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import chromadb

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Ensure your Google API Key is set in a .env file
# GOOGLE_API_KEY="YOUR_API_KEY"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# Define paths for data and persistent storage
DATA_DIR = "data"
CHROMA_PERSIST_DIR = "./chroma_db"

print(f"Data directory: {os.path.abspath(DATA_DIR)}")
print(f"ChromaDB persistence directory: {os.path.abspath(CHROMA_PERSIST_DIR)}")

def main():
    """
    Main function to ingest documents into a ChromaDB vector store.
    """
    print("Starting ingestion process...")

    # 1. Initialize ChromaDB client and collection
    # Creates a persistent client that saves data to disk
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Create a new collection or get an existing one
    # A collection is like a table in a traditional database
    chroma_collection = db.get_or_create_collection("persona_ai_collection")
    print("ChromaDB collection 'persona_ai_collection' ready.")

    # 2. Load documents from the data directory
    # SimpleDirectoryReader is a versatile loader for various file types (PDF, TXT, etc.)
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        if not documents:
            print(f"No documents found in '{DATA_DIR}'. Please add a book/document to that folder.")
            return
        print(f"Successfully loaded {len(documents)} document(s) from '{DATA_DIR}'.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # --- DEBUG: Test Embedding Model ---
    print("\n--- Testing Embedding Model ---")
    try:
        sample_text = "This is a test to see if the embedding model is working."
        if documents:
            sample_text = documents[0].text[:200] # Use actual document text
        
        embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
        embedding = embed_model.get_text_embedding(sample_text)
        print(f"Successfully generated an embedding of dimension: {len(embedding)}")
        print("Embedding model appears to be working correctly.")
    except Exception as e:
        print(f"!!! Error testing embedding model: {e} !!!")
        print("The embedding model is likely the cause of the issue. Please check your API key and model access.")
        return
    print("-----------------------------\n")

    # 3. Set up the embedding model and global settings
    # We use Google's Gemini embedding model and set it in the global Settings
    Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
    print("Gemini embedding model initialized and set in global Settings.")

    # 4. Set up the StorageContext
    # The StorageContext defines where the index and vectors are stored
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Storage context configured.")

    # 5. Create the VectorStoreIndex
    # This is the core step where LlamaIndex does its magic:
    # - Takes documents
    # - Splits them into text nodes
    # - Generates embeddings for each node using the model from Settings
    # - Stores the embeddings in the ChromaDB vector_store
    print("Creating index and generating embeddings... (This may take a while)")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    
    print("--------------------------------------------------")
    print("Ingestion complete!")
    print(f"Index created successfully with {len(index.docstore.docs)} nodes.")
    print(f"Embeddings are stored in: {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()