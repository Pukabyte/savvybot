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
    """Get all resolved threads from the forum channel."""
    try:
        channel_id = int(os.getenv('FORUM_CHANNEL_ID'))
        channel = client.get_channel(channel_id)
        
        if not channel or not isinstance(channel, discord.ForumChannel):
            logging.error(f"Channel {channel_id} is not a forum channel")
            return []
            
        resolved_threads = []
        
        try:
            # Get all active threads
            active_threads = channel.threads
            for thread in active_threads:
                if thread.name.upper().startswith('RESOLVED:'):
                    logging.info(f"Processing active thread: {thread.name}")
                    resolved_threads.append(thread)
                    
            # Get archived threads - use async for directly
            async for thread in channel.archived_threads():
                if thread.name.upper().startswith('RESOLVED:'):
                    logging.info(f"Processing archived thread: {thread.name}")
                    resolved_threads.append(thread)
                    
        except Exception as e:
            logging.error(f"Error getting threads: {e}")
                
        logging.info(f"Found {len(resolved_threads)} resolved threads")
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
        
        # Load existing documents
        documents = await load_documents()
        updated = False
        
        # Process each thread
        for thread in resolved_threads:
            logging.info(f"Processing thread content for: {thread.name}")
            
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
                logging.info(f"Using Ollama URL: {ollama_url}")
                
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
                    thread_content = {
                        "text": processed_content,
                        "source": f"Discord Thread: {thread.name}"
                    }
                    
                    # Add if not already present
                    if not any(doc["source"] == thread_content["source"] for doc in documents):
                        documents.append(thread_content)
                        updated = True
                        logging.info(f"Added processed thread: {thread.name}")
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"HTTP Error processing thread with Ollama: {thread.name} - {e}")
                logging.error(f"Response content: {getattr(e.response, 'content', 'No response content')}")
                continue
            except Exception as e:
                logging.error(f"Error processing thread with Ollama: {thread.name} - {e}")
                continue
                
        # Always update the index if we have documents, even if nothing new was added
        if documents:
            await save_documents(documents)
            await update_faiss_index(documents)
            logging.info("Updated FAISS index with existing documents")
            
        logging.info(f"Completed scraping {len(resolved_threads)} resolved threads")
        
    except Exception as e:
        logging.error(f"Error scraping resolved threads: {e}")

def run_scheduled_tasks(client):
    logging.info("Starting scheduled tasks")
    schedule.every().hour.do(scrape_resolved_threads, client)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduled_tasks()
