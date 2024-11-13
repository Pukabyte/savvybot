import discord
import os
import logging
import faiss
import json
import numpy as np
import requests
from dotenv import load_dotenv
from knowledge_base import update_knowledge_base, get_similar_documents
from schedule_scraper import scrape_resolved_threads
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

# Load environment variables from .env file
load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL')
FORUM_CHANNEL_ID = int(os.getenv('FORUM_CHANNEL_ID'))
INDEX_PATH = os.getenv('INDEX_PATH', 'data/knowledge_base.index')
DOCS_PATH = os.getenv('DOCS_PATH', 'data/documents.json')
MAX_RESPONSE_LENGTH = 1900
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')  # Default to llama2 if not set
MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
model = SentenceTransformer(MODEL_NAME)

# At the top of the file, update intents configuration
intents = discord.Intents.all()  # Enable all intents for testing
client = discord.Client(intents=intents)

async def get_ollama_response(prompt, context):
    """Get a response from Ollama using the context."""
    try:
        # Format the context to highlight different file contents
        formatted_context = ""
        current_file = ""
        
        for line in context.split('\n'):
            # Check for code block headers in VitePress format
            if line.startswith('```') and '[' in line and ']' in line:
                file_name = line.split('[')[1].split(']')[0]
                if file_name != current_file:
                    current_file = file_name
                    formatted_context += f"\nFile: {current_file}\n"
            formatted_context += line + '\n'
        
        system_prompt = """You are a helpful assistant for Savvyguides, Saltbox and related applications.
        Use the provided context to answer questions accurately and concisely. 
        If you're not sure about something, say so rather than making assumptions."""
        
        full_prompt = f"Context:\n{formatted_context}\n\nQuestion: {prompt}\n\nAnswer:"
        
        response = requests.post(
            f"{OLLAMA_SERVER_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "system": system_prompt,
                "stream": False
            },
            timeout=30
        )
        
        logging.info(f"Ollama API response status: {response.status_code}")
        response.raise_for_status()
        
        response_json = response.json()
        if 'response' in response_json:
            return response_json['response']
        else:
            logging.error(f"Unexpected response format: {response_json}")
            return "I encountered an error processing your question."
            
    except requests.exceptions.Timeout:
        logging.error("Timeout while waiting for Ollama response")
        return "The request timed out. Please try again."
    except Exception as e:
        logging.error(f"Error getting Ollama response: {e}")
        if hasattr(e, 'response'):
            logging.error(f"Response content: {e.response.content}")
        return "I encountered an error while processing your question."

async def search_knowledge_base(query, k=3):
    """Search the knowledge base for relevant information."""
    try:
        # Load the index and documents
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, 'r') as f:
            documents = json.load(f)
        
        # Encode the query
        query_vector = model.encode([query])
        
        # Search the index
        D, I = index.search(query_vector, k)
        
        # Get the most relevant documents
        context = []
        for idx in I[0]:
            if idx < len(documents):
                context.append(documents[idx]["text"])
        
        if not context:
            return "I couldn't find any relevant information in my knowledge base."
        
        # Get response from Ollama using the context
        ollama_response = await get_ollama_response(query, "\n\n".join(context))
        if ollama_response:
            return ollama_response
        return "I encountered an error while processing your question with the AI model."
        
    except Exception as e:
        logging.error(f"Error searching knowledge base: {e}")
        return "I encountered an error while searching for information."

async def initialize_bot():
    """Initialize the bot and update knowledge base."""
    logging.info("Starting bot initialization")
    try:
        await update_knowledge_base()
        await scrape_resolved_threads(client)
    except Exception as e:
        logging.error(f"Error during initialization: {e}")

@client.event
async def on_ready():
    """Called when the bot is ready."""
    logging.info(f'Logged in as {client.user}')
    client.loop.create_task(initialize_bot())

@client.event
async def on_message(message):
    """Handle incoming messages."""
    # Skip own messages
    if message.author == client.user:
        return
    
    # Check if bot is mentioned or message starts with configured prefix
    if not (client.user in message.mentions or message.content.startswith('!')):
        return
        
    # Remove the mention and clean the message content
    content = message.content
    if client.user in message.mentions:
        content = content.replace(f'<@!{client.user.id}>', '').replace(f'<@{client.user.id}>', '').strip()
    elif message.content.startswith('!'):
        content = content[1:].strip()
        
    if not content:
        await message.channel.send("How can I help you?")
        return
        
    logging.info(f"Processing message from {message.author.name}: {content}")
    
    # Use typing context manager to show typing indicator
    async with message.channel.typing():
        try:
            # Search knowledge base
            similar_docs = await get_similar_documents(content)
            
            if not similar_docs:
                await message.channel.send("I couldn't find any relevant information in my knowledge base.")
                return
            
            # Format response with context
            context = "\n\n".join([f"From {doc['source']}:\n{doc['text']}" for doc in similar_docs])
            
            # Get Ollama response
            response = await get_ollama_response(content, context)
            
            # Send response
            if len(response) > 2000:
                # Split long messages while maintaining typing indicator
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i+2000])
            else:
                await message.channel.send(response)
                
        except Exception as e:
            logging.error(f"Error processing message: {e}", exc_info=True)
            await message.channel.send("I encountered an error while processing your question.")

client.run(DISCORD_TOKEN)
