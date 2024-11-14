import os
import json
import logging
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import asyncio
import concurrent.futures
from datetime import datetime

# Constants
KNOWLEDGE_BASE_DIR = 'data'
DOCUMENTS_FILE = os.path.join(KNOWLEDGE_BASE_DIR, 'documents.json')
INDEX_FILE = os.path.join(KNOWLEDGE_BASE_DIR, 'knowledge_base.index')
METADATA_FILE = os.path.join(KNOWLEDGE_BASE_DIR, 'knowledge_base_metadata.json')

def ensure_knowledge_base_dir():
    """Ensure knowledge base directory exists."""
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

def ensure_file_dir(filepath):
    """Ensure directory exists for a given file path."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

# Initialize model and executor
model = SentenceTransformer('all-MiniLM-L6-v2')
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Global variables
_index = None
_documents = None
_metadata = {
    'created_at': None,
    'last_updated': None,
    'document_count': 0,
    'vector_dimension': 384
}

async def load_documents():
    """Load documents from JSON file."""
    try:
        if os.path.exists(DOCUMENTS_FILE):
            with open(DOCUMENTS_FILE, 'r') as f:
                documents = json.load(f)
            logging.info(f"Successfully loaded {len(documents)} documents from {DOCUMENTS_FILE}")
            return documents
        else:
            logging.info(f"No documents file found at {DOCUMENTS_FILE}")
            return []
    except Exception as e:
        logging.error(f"Error loading documents: {e}")
        return []

async def save_documents(documents, append=False):
    """Save documents to JSON file."""
    try:
        ensure_knowledge_base_dir()
        
        # If appending, load existing documents first
        if append and os.path.exists(DOCUMENTS_FILE):
            with open(DOCUMENTS_FILE, 'r') as f:
                existing_docs = json.load(f)
                
            # Create a set of existing sources
            existing_sources = {doc.get('source') for doc in existing_docs}
            
            # Only add documents with new sources
            new_docs = [
                doc for doc in documents 
                if doc.get('source') not in existing_sources
            ]
            
            if new_docs:
                documents = existing_docs + new_docs
                logging.info(f"Added {len(new_docs)} new documents to existing {len(existing_docs)}")
            else:
                documents = existing_docs
                logging.info("No new documents to add")
        
        # Save all documents
        ensure_file_dir(DOCUMENTS_FILE)
        with open(DOCUMENTS_FILE, 'w') as f:
            json.dump(documents, f, indent=2)
        logging.info(f"Successfully saved {len(documents)} documents to {DOCUMENTS_FILE}")
        
        return documents
        
    except Exception as e:
        logging.error(f"Error saving documents: {e}")
        raise

async def save_metadata():
    """Save knowledge base metadata."""
    try:
        ensure_knowledge_base_dir()
        with open(METADATA_FILE, 'w') as f:
            json.dump(_metadata, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving metadata: {e}")
        raise

async def load_metadata():
    """Load or create knowledge base metadata."""
    global _metadata
    try:
        ensure_knowledge_base_dir()
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                _metadata.update(json.load(f))
        else:
            _metadata.update({
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'document_count': 0,
                'vector_dimension': 384
            })
            await save_metadata()
        return _metadata
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        return _metadata

async def update_faiss_index(documents):
    """Update knowledge base index with documents."""
    global _metadata
    try:
        if not documents:
            logging.warning("No documents to index")
            return
            
        # Ensure metadata is loaded
        await load_metadata()
        
        texts = [doc['text'] for doc in documents]
        
        # Use smaller batch size and clear CUDA cache between batches
        batch_size = 16
        embeddings_list = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i + batch_size, len(texts))]
            try:
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    executor, 
                    lambda: model.encode(batch, show_progress_bar=False)
                )
                embeddings_list.append(batch_embeddings)
                
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.error(f"Error encoding batch {i//batch_size + 1}: {e}")
                continue
                
        if not embeddings_list:
            logging.error("Failed to encode any documents")
            return
            
        embeddings = np.vstack(embeddings_list)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        ensure_knowledge_base_dir()
        faiss.write_index(index, INDEX_FILE)
        
        _metadata['document_count'] = len(documents)
        _metadata['vector_dimension'] = dimension
        _metadata['last_updated'] = datetime.now().isoformat()
        await save_metadata()
        
        logging.info(f"Successfully updated FAISS index and metadata with {len(documents)} documents")
        
    except Exception as e:
        logging.error(f"Error updating knowledge base index: {e}")
        raise