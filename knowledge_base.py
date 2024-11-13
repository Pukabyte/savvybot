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
from utils import load_documents, save_documents, update_faiss_index
from sentence_transformers import SentenceTransformer
import requests

# Load environment variables
load_dotenv()
DOCUMENTS_URLS = os.getenv('DOCUMENTS_URLS', '').split(',')
INDEX_PATH = os.getenv('INDEX_PATH', 'data/knowledge_base.index')
DOCS_PATH = os.getenv('DOCS_PATH', 'data/documents.json')
OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL', 'http://localhost:11434')

# Create a semaphore to limit concurrent requests
REQUEST_LIMIT = 5

# Initialize the sentence transformer model
MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
model = SentenceTransformer(MODEL_NAME)

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

async def update_knowledge_base(resolved_threads=None):
    """Update the knowledge base with both documentation and resolved threads."""
    try:
        # Fetch documentation
        documents = await fetch_documentation()
        
        # Process resolved threads if provided
        if resolved_threads:
            for thread in resolved_threads:
                thread_content = await process_resolved_thread(thread)
                if thread_content:
                    documents.append(thread_content)
        
        # Save all documents
        await save_documents(documents)
        
        # Update the FAISS index
        await update_faiss_index(documents)
        
        logging.info(f"Knowledge base updated successfully. Saved {len(documents)} documents.")
        
    except Exception as e:
        logging.error(f"Error updating knowledge base: {e}")

async def get_similar_documents(query, top_k=3):
    """Get similar documents from the knowledge base."""
    try:
        logging.info("Loading index and documents...")
        
        # Load the index and documents
        if not os.path.exists('data/knowledge_base.index'):
            logging.error("No index file found")
            return []
            
        index = faiss.read_index('data/knowledge_base.index')
        documents = await load_documents()
        logging.info(f"Loaded {len(documents)} documents")
        
        # Get query embedding
        logging.info("Getting query embedding...")
        query_embedding = model.encode([query])[0]
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Search
        logging.info("Searching index...")
        D, I = index.search(query_embedding_array, min(top_k, len(documents)))
        
        # Get similar documents
        similar_docs = []
        for idx in I[0]:
            if idx < len(documents):
                similar_docs.append(documents[idx])
                
        return similar_docs
        
    except Exception as e:
        logging.error(f"Error searching knowledge base: {str(e)}")
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

async def fetch_documentation():
    """Fetch documentation from configured URLs."""
    try:
        urls = os.getenv('DOCUMENTS_URLS', '').split(',')
        documents = []
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
            visited_urls = set()
            
            tasks = [fetch_text_from_url(url.strip(), visited_urls, session, semaphore) 
                    for url in urls if url.strip()]
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        documents.extend(result)
                    elif isinstance(result, Exception):
                        logging.error(f"Error fetching documentation: {result}")
            
        logging.info(f"Successfully fetched {len(documents)} documents from documentation")
        return documents
        
    except Exception as e:
        logging.error(f"Error fetching documentation: {e}")
        return []

async def update_faiss_index(documents):
    """Update the FAISS index with document embeddings."""
    try:
        # Create embeddings for all documents
        texts = [doc['text'] for doc in documents]
        embeddings = model.encode(texts)
        
        # Convert to float32 numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Create and populate FAISS index
        dimension = embeddings.shape[1]  # Get embedding dimension
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save the index
        faiss.write_index(index, 'data/knowledge_base.index')
        logging.info(f"Successfully updated FAISS index with {len(documents)} documents")
        
    except Exception as e:
        logging.error(f"Error updating FAISS index: {e}")
        raise

async def process_resolved_thread(thread):
    """Process a resolved thread and extract a clear answer using Ollama."""
    try:
        # Get the thread title and all messages
        thread_title = thread.name
        messages = []
        
        # Fetch messages from the thread
        async for message in thread.history(limit=None, oldest_first=True):
            if not message.author.bot:  # Skip bot messages
                messages.append(message.content)
        
        if not messages:
            return None
            
        # Create prompt for Ollama
        thread_content = ' '.join(messages)
        prompt = f"""Given this Discord support thread, extract the core question and solution.
        Thread title: {thread_title}
        Thread content: {thread_content}
        
        Please provide a clear and concise answer that explains the solution to the problem.
        Format as: Question: [clear problem statement]
        Answer: [concise solution]"""
        
        # Get Ollama's response
        response = await get_ollama_response(prompt)
        
        if not response:
            logging.error(f"Failed to get Ollama response for thread: {thread_title}")
            return None
            
        # Format the thread content
        return {
            "text": response,
            "source": f"Discord Thread: {thread_title}"
        }
            
    except Exception as e:
        logging.error(f"Error processing resolved thread {thread.name}: {e}")
        return None

async def get_ollama_response(prompt):
    """Get a response from Ollama."""
    try:
        response = requests.post(
            f"{os.getenv('OLLAMA_SERVER_URL')}/api/generate",
            json={
                "model": os.getenv('OLLAMA_MODEL', 'llama2'),
                "prompt": prompt,
                "system": "You are a helpful assistant that extracts clear questions and answers from support threads. Focus on the core problem and its solution.",
                "stream": False
            },
            timeout=30
        )
        
        response.raise_for_status()
        return response.json().get('response', '')
        
    except Exception as e:
        logging.error(f"Error getting Ollama response: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(update_knowledge_base())
