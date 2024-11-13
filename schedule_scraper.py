import schedule
import time
import os
import logging
from dotenv import load_dotenv
import discord
from datetime import datetime, timezone
from utils import load_documents, save_documents, update_faiss_index
import requests

load_dotenv()
FORUM_CHANNEL_ID = int(os.getenv('FORUM_CHANNEL_ID'))

async def get_resolved_threads(client):
    """Get all threads with the resolved tag from the forum channel."""
    try:
        # Check for required environment variables
        channel_id = os.getenv('FORUM_CHANNEL_ID')
        resolved_tag_id = os.getenv('RESOLVED_TAG_ID')
        
        if not channel_id or not resolved_tag_id:
            logging.error("Missing required environment variables: FORUM_CHANNEL_ID or RESOLVED_TAG_ID")
            return []
            
        channel_id = int(channel_id)
        resolved_tag_id = int(resolved_tag_id)
        
        channel = client.get_channel(channel_id)
        
        if not channel or not isinstance(channel, discord.ForumChannel):
            logging.error(f"Channel {channel_id} is not a forum channel")
            return []
            
        resolved_threads = []
        
        try:
            # Get all threads from the forum
            active_threads = channel.threads
            for thread in active_threads:
                # Check if the resolved tag is applied
                if any(tag.id == resolved_tag_id for tag in thread.applied_tags):
                    logging.info(f"Processing tagged thread: {thread.name}")
                    resolved_threads.append(thread)
                    
            # Get archived threads
            async for thread in channel.archived_threads():
                if any(tag.id == resolved_tag_id for tag in thread.applied_tags):
                    logging.info(f"Processing archived tagged thread: {thread.name}")
                    resolved_threads.append(thread)
                    
        except Exception as e:
            logging.error(f"Error getting threads: {e}")
                
        logging.info(f"Found {len(resolved_threads)} threads with resolved tag")
        return resolved_threads
        
    except Exception as e:
        logging.error(f"Error getting resolved threads: {e}")
        return []

async def process_resolved_thread(thread):
    """Process a resolved thread and return its content."""
    try:
        # Get the thread title and all messages
        thread_title = thread.name
        messages = []
        async for message in thread.history(limit=None, oldest_first=True):
            if not message.author.bot:  # Skip bot messages
                messages.append(message.content)
        
        if not messages:
            return None
            
        # Format the thread content
        return {
            "text": f"Question: {thread_title}\n\nAnswer: {' '.join(messages)}",
            "source": f"Discord Thread: {thread_title}"
        }
            
    except Exception as e:
        logging.error(f"Error processing resolved thread {thread.name}: {e}")
        return None

async def scrape_resolved_threads(client):
    """Scrape resolved threads and update knowledge base."""
    try:
        logging.info("Starting to scrape resolved threads")
        resolved_threads = await get_resolved_threads(client)
        
        # Load existing documents or start with empty list
        try:
            documents = await load_documents()
        except:
            documents = []
            
        updated = False
        
        # Process each thread
        for thread in resolved_threads:
            logging.info(f"Processing thread content for: {thread.name}")
            
            # Skip if thread already exists in documents
            if any(doc["source"] == f"Discord Thread: {thread.name}" for doc in documents):
                logging.info(f"Thread already exists in documents: {thread.name}")
                continue
                
            # Fetch all messages in the thread
            messages = []
            async for message in thread.history(limit=None, oldest_first=True):
                if not message.author.bot:
                    messages.append(message.content)
            
            if not messages:
                logging.warning(f"No messages found in thread: {thread.name}")
                continue
                
            # Create prompt for Ollama
            thread_content = ' '.join(messages)
            prompt = f"""Given this Discord support thread, extract the core question and solution.

Thread title: {thread.name}
Thread content: {thread_content}

Please provide a clear and concise answer that explains the solution to the problem.
Format as:
Question: [clear problem statement]
Answer: [concise solution]"""
            
            # Get Ollama's response
            try:
                logging.info(f"Sending request to Ollama for thread: {thread.name}")
                ollama_url = os.getenv('OLLAMA_SERVER_URL', 'http://ollama:11434')
                
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": os.getenv('OLLAMA_MODEL', 'llama2'),
                        "prompt": prompt,
                        "system": "You are a helpful assistant that extracts clear questions and answers from support threads. Focus on the core problem and its solution.",
                        "stream": False
                    },
                    timeout=30
                )
                
                logging.info(f"Ollama response status: {response.status_code}")
                response.raise_for_status()
                
                processed_content = response.json().get('response', '')
                logging.info(f"Processed content from Ollama: {processed_content[:100]}...")
                
                if processed_content:
                    thread_doc = {
                        "text": processed_content,
                        "source": f"Discord Thread: {thread.name}"
                    }
                    documents.append(thread_doc)
                    updated = True
                    logging.info(f"Added processed thread: {thread.name}")
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"HTTP Error processing thread with Ollama: {thread.name} - {e}")
                continue
            except Exception as e:
                logging.error(f"Error processing thread with Ollama: {thread.name} - {e}")
                continue
                
        # Save and index if we have new documents
        if updated:
            await save_documents(documents)
            await update_faiss_index(documents)
            logging.info(f"Updated knowledge base with new thread documents")
            
        logging.info(f"Completed scraping {len(resolved_threads)} resolved threads")
        
    except Exception as e:
        logging.error(f"Error scraping resolved threads: {e}")

async def run_scheduled_tasks(client):
    """Run all scheduled tasks."""
    try:
        # Scrape documentation
        if os.getenv('DOCUMENTATION_URLS'):
            logging.info("Starting documentation scraping")
            await scrape_documentation()
            
        # Scrape resolved threads
        logging.info("Starting thread scraping")
        await scrape_resolved_threads(client)
        
    except Exception as e:
        logging.error(f"Error in scheduled tasks: {e}")

async def scrape_documentation():
    """Scrape documentation URLs and update knowledge base."""
    try:
        # Get documentation URLs from environment
        doc_urls = os.getenv('DOCUMENTATION_URLS', '').split(',')
        doc_urls = [url.strip() for url in doc_urls if url.strip()]
        
        if not doc_urls:
            logging.info("No documentation URLs configured")
            return
            
        documents = []
        for url in doc_urls:
            try:
                logging.info(f"Processing documentation from {url}")
                chunks = await process_documentation_url(url)
                if chunks:
                    documents.extend(chunks)
                    logging.info(f"Successfully processed {len(chunks)} chunks from {url}")
            except Exception as e:
                logging.error(f"Error processing URL {url}: {e}")
                continue
                
        if documents:
            logging.info(f"Successfully fetched {len(documents)} documents from documentation")
            await save_documents(documents)
            await update_faiss_index(documents)
            logging.info("Knowledge base updated successfully. Saved {len(documents)} documents.")
            
    except Exception as e:
        logging.error(f"Error updating knowledge base: {e}")

def run_scheduled_tasks(client):
    logging.info("Starting scheduled tasks")
    schedule.every().hour.do(run_scheduled_tasks, client)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduled_tasks()
