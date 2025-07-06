# Persona AI

This project is a "ask me anything" application with a specific persona, built on a knowledge base of documents. It uses a FastAPI backend to serve a simple frontend, and leverages LlamaIndex and ChromaDB to store and query a vector database of documents. The language model is powered by Google's Gemini.

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    Create a `.env` file in the root of the project and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY"
    ```

## Running the Application

1.  **Ingest the data:**
    Run the ingest script to process the documents in the `data` directory and create the vector store:
    ```bash
    python ingest.py
    ```

2.  **Run the web application:**
    ```bash
    uvicorn app.main:app --reload
    ```

    The application will be available at `http://127.0.0.1:8000`.
