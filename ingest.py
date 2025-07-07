import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import chromadb

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Ensure your Google API Key is set in a .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set. Using a placeholder.")
    GOOGLE_API_KEY = "placeholder"

# Define paths for data and persistent storage
DATA_DIR = "data"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "persona_ai_collection"

print(f"Data directory: {os.path.abspath(DATA_DIR)}")
print(f"ChromaDB persistence directory: {os.path.abspath(CHROMA_PERSIST_DIR)}")

def main():
    """
    Main function to ingest documents into a ChromaDB vector store.
    """
    print("Starting ingestion process...")

    # 1. Initialize ChromaDB client
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # 2. Clear out the old collection if it exists
    print(f"Checking for existing collection: {COLLECTION_NAME}")
    if COLLECTION_NAME in [c.name for c in db.list_collections()]:
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        db.delete_collection(name=COLLECTION_NAME)
    
    # Create a new collection
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    print(f"ChromaDB collection '{COLLECTION_NAME}' created/reset.")

    # 3. Load documents
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        if not documents:
            print(f"No documents found in '{DATA_DIR}'. Please add a book/document to that folder.")
            return
        print(f"Successfully loaded {len(documents)} document(s) from '{DATA_DIR}'.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 4. Set up the embedding model and global settings
    Settings.embed_model = GoogleGenAIEmbedding()
    print("Gemini embedding model initialized.")

    # 5. Define and run the ingestion pipeline
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )
    
    print("Running ingestion pipeline... (This may take a while)")
    # This will process documents, split them, embed them, and store them.
    pipeline.run(documents=documents, show_progress=True)

    # 6. Verification
    num_nodes = len(chroma_collection.get()["ids"])
    print("--------------------------------------------------")
    print("Ingestion complete!")
    print(f"Successfully ingested {num_nodes} nodes into ChromaDB.")
    print(f"Embeddings are stored in: {os.path.abspath(CHROMA_PERSIST_DIR)}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
