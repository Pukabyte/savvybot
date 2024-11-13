import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
model = SentenceTransformer(MODEL_NAME)

async def load_documents():
    """Load documents from JSON file."""
    try:
        if not os.path.exists('data/documents.json'):
            logging.info("No existing documents.json found. Starting fresh.")
            return []
            
        with open('data/documents.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logging.info(f"Successfully loaded {len(documents)} documents from data/documents.json")
        return documents
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        return []

async def save_documents(documents):
    """Save documents to JSON file."""
    os.makedirs('data', exist_ok=True)
    
    try:
        with open('data/documents.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved {len(documents)} documents to data/documents.json")
    except Exception as e:
        logging.error(f"Error saving documents: {e}")
        raise

async def update_faiss_index(documents):
    """Update the FAISS index with document embeddings."""
    try:
        # Create embeddings for all documents
        texts = [doc['text'] for doc in documents]
        embeddings = model.encode(texts)
        
        # Convert to float32 numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Create and populate FAISS index
        dimension = 384  # MiniLM-L6-v2 uses 384 dimensions
        index = faiss.IndexFlatL2(dimension)
        
        # Verify dimensions match
        if embeddings.shape[1] != dimension:
            logging.error(f"Embedding dimension mismatch. Expected {dimension}, got {embeddings.shape[1]}")
            return
            
        index.add(embeddings)
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save the index
        faiss.write_index(index, 'data/knowledge_base.index')
        
        # Save metadata about the index
        index_metadata = {
            "dimension": dimension,
            "model": MODEL_NAME,
            "num_documents": len(documents),
            "document_ids": list(range(len(documents)))  # Keep track of document order
        }
        
        with open('data/index_metadata.json', 'w') as f:
            json.dump(index_metadata, f)
            
        logging.info(f"Successfully updated FAISS index and metadata with {len(documents)} documents")
        
    except Exception as e:
        logging.error(f"Error updating FAISS index: {e}")
        raise