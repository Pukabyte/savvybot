import schedule
import time
import os
import logging
from dotenv import load_dotenv
import discord
from datetime import datetime, timezone
from utils import load_documents, save_documents, update_faiss_index
import requests
import asyncio

load_dotenv()
FORUM_CHANNEL_ID = int(os.getenv('FORUM_CHANNEL_ID'))

async def get_resolved_threads(client):
    """Get all threads tagged as resolved."""
    try:
        resolved_threads = []
        support_channel_id = int(os.getenv('FORUM_CHANNEL_ID'))
        
        for guild in client.guilds:
            logging.info(f"Checking guild: {guild.name}")
            
            # Get the forum channel by ID
            forum_channel = guild.get_channel(support_channel_id)
            if not forum_channel:
                logging.warning(f"Could not find forum channel with ID {support_channel_id} in guild {guild.name}")
                continue
                
            if not isinstance(forum_channel, discord.ForumChannel):
                logging.warning(f"Channel {forum_channel.name} is not a forum channel (Type: {type(forum_channel)})")
                continue
                
            logging.info(f"Found forum channel: {forum_channel.name}")
            
            # Check archived threads
            archived_count = 0
            async for thread in forum_channel.archived_threads(limit=None):
                archived_count += 1
                logging.info(f"Checking archived thread: {thread.name}")
                if "RESOLVED" in thread.name.upper():
                    logging.info(f"Processing archived tagged thread: {thread.name}")
                    try:
                        # Get the initial message
                        messages = [msg async for msg in thread.history(limit=1, oldest_first=True)]
                        initial_message = messages[0] if messages else None
                        
                        if initial_message:
                            resolved_threads.append({
                                'title': thread.name,
                                'content': initial_message.content,
                                'timestamp': thread.created_at.isoformat()
                            })
                            logging.info(f"Successfully processed archived thread: {thread.name}")
                    except discord.Forbidden:
                        logging.warning(f"No permission to read thread: {thread.name}")
                    except Exception as e:
                        logging.error(f"Error processing archived thread {thread.name}: {e}")
                        continue
            
            logging.info(f"Checked {archived_count} archived threads")
            
            # Check active threads
            active_count = 0
            for thread in forum_channel.threads:
                active_count += 1
                logging.info(f"Checking active thread: {thread.name}")
                if "RESOLVED" in thread.name.upper():
                    logging.info(f"Processing active tagged thread: {thread.name}")
                    try:
                        # Get the initial message
                        messages = [msg async for msg in thread.history(limit=1, oldest_first=True)]
                        initial_message = messages[0] if messages else None
                        
                        if initial_message:
                            resolved_threads.append({
                                'title': thread.name,
                                'content': initial_message.content,
                                'timestamp': thread.created_at.isoformat()
                            })
                            logging.info(f"Successfully processed active thread: {thread.name}")
                    except discord.Forbidden:
                        logging.warning(f"No permission to read thread: {thread.name}")
                    except Exception as e:
                        logging.error(f"Error processing active thread {thread.name}: {e}")
                        continue
                        
            logging.info(f"Checked {active_count} active threads")
                        
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
    """Scrape resolved threads and update documents."""
    try:
        # Load existing documents
        documents = await load_documents() or []
        initial_count = len(documents)
        
        # Get resolved threads
        resolved_threads = await get_resolved_threads(client)
        if not resolved_threads:
            logging.info("No resolved threads found")
            return 0
            
        # Process new threads
        new_thread_count = 0
        for thread in resolved_threads:
            thread_doc = {
                'source': f"Thread: {thread['title']}",
                'text': f"Question: {thread['title']}\n\nAnswer: {thread['content']}",
                'type': 'thread',
                'timestamp': thread.get('timestamp', datetime.now().isoformat())
            }
            
            # Check if thread already exists
            if not any(doc.get('source') == thread_doc['source'] for doc in documents):
                documents.append(thread_doc)
                new_thread_count += 1
                logging.info(f"Added processed thread: {thread['title']}")
                
        # Save if we have new threads
        if new_thread_count > 0:
            await save_documents(documents, append=True)
            logging.info(f"Updated knowledge base with new thread documents")
            
        logging.info(f"Completed scraping {len(resolved_threads)} resolved threads")
        return new_thread_count
        
    except Exception as e:
        logging.error(f"Error scraping resolved threads: {e}")
        return 0

async def process_thread_with_ollama(thread_content, thread_name):
    """Process thread content with Ollama in a separate function for better error handling."""
    try:
        prompt = f"""Given this Discord support thread, extract the core question and solution.

Thread title: {thread_name}
Thread content: {thread_content}

Please provide a clear and concise answer that explains the solution to the problem.
Format as:
Question: [clear problem statement]
Answer: [concise solution]"""
        
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
        
        response.raise_for_status()
        return response.json().get('response', '')
        
    except Exception as e:
        logging.error(f"Error processing thread with Ollama: {e}")
        return None

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
