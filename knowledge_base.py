import os
import json
import logging
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import faiss
import numpy as np
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
DOCUMENTS_URLS = os.getenv('DOCUMENTS_URLS', '').split(',')
INDEX_PATH = os.getenv('INDEX_PATH', 'data/knowledge_base.index')
DOCS_PATH = os.getenv('DOCS_PATH', 'data/documents.json')
OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL', 'http://localhost:11434')

# Create a semaphore to limit concurrent requests
REQUEST_LIMIT = 5

async def process_document(text, url, max_length=8000):
    """Process and split document into manageable chunks."""
    documents = []
    
    # Split content into sections based on headers
    sections = re.split(r'(?=#{1,6}\s)', text)
    
    current_chunk = ""
    current_length = 0
    
    for section in sections:
        # If adding this section would exceed max length, save current chunk
        if current_length + len(section) > max_length and current_chunk:
            documents.append({
                "text": current_chunk.strip(),
                "source": url
            })
            current_chunk = ""
            current_length = 0
            
        current_chunk += section
        current_length += len(section)
    
    # Add the last chunk if it exists
    if current_chunk:
        documents.append({
            "text": current_chunk.strip(),
            "source": url
        })
    
    return documents

async def fetch_text_from_url(url, visited_urls, session, semaphore):
    """Recursively fetch text content from URL and its internal links."""
    base_url = url.split('#')[0]
    
    if base_url in visited_urls:
        return []
    
    visited_urls.add(base_url)
    documents = []
    
    try:
        async with semaphore:
            async with session.get(url) as response:
                html = await response.text()
                await asyncio.sleep(0.2)
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove navigation elements
        for nav in soup.select('.VPNav, .VPFooter, .VPSidebar'):
            nav.decompose()
            
        # Process content sections
        content_sections = []
        
        # Handle code groups specifically
        code_groups = soup.find_all('div', class_='vp-code-group')
        for group in code_groups:
            # Get all tabs and code blocks in the group
            tabs = group.find_all('div', class_='tabs')
            blocks = group.find_all('div', class_='blocks')
            
            if tabs and blocks:
                # Extract tab names and code content
                for tab, block in zip(tabs[0].find_all('input'), blocks[0].find_all('div', class_='language-')):
                    tab_name = tab.get('aria-label', '').strip()
                    code = block.find('code')
                    if code:
                        lang = ''
                        if code.get('class'):
                            lang_class = next((c for c in code.get('class') if c.startswith('language-')), '')
                            lang = lang_class.replace('language-', '')
                        
                        code_text = code.get_text().strip()
                        if code_text:
                            content_sections.append(f"File: {tab_name}")
                            content_sections.append(f"```{lang}\n{code_text}\n```")
        
        # Handle regular text content
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text().strip()
            if text:
                content_sections.append(text)
        
        # Handle individual code blocks
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code and not pre.find_parent(class_='vp-code-group'):  # Skip if part of a code group
                lang = ''
                if code.get('class'):
                    lang_class = next((c for c in code.get('class') if c.startswith('language-')), '')
                    lang = lang_class.replace('language-', '')
                
                code_text = code.get_text().strip()
                if code_text:
                    content_sections.append(f"```{lang}\n{code_text}\n```")
        
        # Combine content sections and process into documents
        text_content = '\n\n'.join(content_sections)
        if text_content.strip():
            # Process and split the document into manageable chunks
            chunk_documents = await process_document(text_content, url)
            documents.extend(chunk_documents)
            logging.info(f"Successfully processed {len(chunk_documents)} chunks from {url}")
        
        # Find and follow internal links
        tasks = []
        seen_links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            base_absolute_url = absolute_url.split('#')[0]
            
            if (is_internal_link(url, absolute_url) and 
                base_absolute_url not in visited_urls and 
                base_absolute_url not in seen_links):
                
                seen_links.add(base_absolute_url)
                logging.info(f"Following internal link: {absolute_url}")
                tasks.append(fetch_text_from_url(absolute_url, visited_urls, session, semaphore))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    documents.extend(result)
    
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
    
    return documents

async def get_embedding(text, session, semaphore):
    """Get embedding from Ollama using nomic-embed-text."""
    try:
        async with semaphore:
            async with session.post(
                f"{OLLAMA_SERVER_URL}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                await asyncio.sleep(0.2)
                return np.array(data['embedding'], dtype=np.float32)
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None

def is_internal_link(base_url, link):
    """Check if a link is internal to the base domain."""
    base_domain = urlparse(base_url).netloc
    link_domain = urlparse(link).netloc
    return not link_domain or base_domain == link_domain

async def update_knowledge_base():
    """Update the knowledge base by scraping URLs and generating embeddings."""
    logging.info("Starting knowledge base update")
    documents = []
    
    # Create semaphore for this update session
    semaphore = asyncio.Semaphore(REQUEST_LIMIT)
    
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        visited_urls = set()
        
        # Fetch all documents
        for base_url in DOCUMENTS_URLS:
            if not base_url:
                continue
            docs = await fetch_text_from_url(base_url, visited_urls, session, semaphore)
            documents.extend(docs)
        
        if not documents:
            logging.warning("No documents were fetched")
            return
        
        # Generate embeddings for the documents
        logging.info(f"Generating embeddings for {len(documents)} documents")
        embeddings = []
        for doc in documents:
            embedding = await get_embedding(doc["text"], session, semaphore)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logging.error(f"Failed to get embedding for document from {doc['source']}")
        
        if not embeddings:
            logging.error("No embeddings were generated")
            return
        
        # Convert to numpy array and initialize FAISS index
        embeddings_array = np.array(embeddings)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Save the index and document data
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        with open(DOCS_PATH, "w") as f:
            json.dump(documents, f, indent=2)
        logging.info(f"Knowledge base updated successfully. Saved {len(documents)} documents.")

async def get_similar_documents(query, top_k=5):
    """Search for similar documents using Ollama embeddings."""
    try:
        # Load the FAISS index and documents
        if not os.path.exists(INDEX_PATH):
            logging.error(f"Index file not found at {INDEX_PATH}")
            return []
            
        if not os.path.exists(DOCS_PATH):
            logging.error(f"Documents file not found at {DOCS_PATH}")
            return []
        
        logging.info("Loading index and documents...")
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, 'r') as f:
            documents = json.load(f)
        
        logging.info(f"Loaded {len(documents)} documents")
        
        # Get query embedding
        logging.info("Getting query embedding...")
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(1)
            query_embedding = await get_embedding(query, session, semaphore)
        
        if query_embedding is None:
            logging.error("Failed to get query embedding")
            return []
        
        # Search the index
        logging.info("Searching index...")
        query_embedding_array = np.array([query_embedding])
        D, I = index.search(query_embedding_array, top_k)
        
        # Return the matched documents
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(documents):  # Ensure valid index
                doc = documents[idx]
                score = float(1 / (1 + dist))
                results.append({
                    'text': doc['text'],
                    'source': doc['source'],
                    'score': score
                })
                logging.info(f"Found match: {doc['source']} with score {score}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error searching knowledge base: {str(e)}", exc_info=True)  # Add full traceback
        return []

async def save_documents(documents):
    """Save documents to JSON file."""
    os.makedirs('data', exist_ok=True)  # Create data directory if it doesn't exist
    
    try:
        with open('data/documents.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved {len(documents)} documents to data/documents.json")
    except Exception as e:
        logging.error(f"Error saving documents: {e}")
        raise

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

if __name__ == "__main__":
    asyncio.run(update_knowledge_base())
