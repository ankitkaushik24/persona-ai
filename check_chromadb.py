import chromadb
import os

# Adjust this path to match your ingestion/app config
CHROMA_PERSIST_DIR = os.path.abspath("chroma_db")
COLLECTION_NAME = "persona_ai_collection"

def main():
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = db.get_collection(COLLECTION_NAME)
    
    # Get all documents with their content
    docs = collection.get(include=["documents"])
    
    if docs and docs["documents"]:
        num_docs = len(docs["documents"])
        print(f"Found {num_docs} documents in collection '{COLLECTION_NAME}'.")
        
        # Print the first 10 documents for inspection
        for i, doc in enumerate(docs["documents"][:10]):
            print(f"--- Document {i+1} (first 500 chars) ---")
            print(doc[:500])
            print("\n")
            
    else:
        print(f"No documents found in the collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main() 