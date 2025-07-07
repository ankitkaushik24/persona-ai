import chromadb
import os

# Adjust this path to match your ingestion/app config
CHROMA_PERSIST_DIR = os.path.abspath("chroma_db")
COLLECTION_NAME = "persona_ai_collection"

def main():
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = db.get_collection(COLLECTION_NAME)
    docs = collection.get(include=["documents"])
    if docs["documents"] and isinstance(docs["documents"], list) and len(docs["documents"]) > 0 and docs["documents"][0]:
        num_docs = len(docs["documents"][0])
        print(f"Number of docs in collection '{COLLECTION_NAME}': {num_docs}")
        print("First doc (first 500 chars):\n", docs["documents"][0][0][:500])
    else:
        print(f"No documents found in the collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main() 