import discord
import os
import logging
import faiss
import json
import numpy as np
import requests
from dotenv import load_dotenv
from knowledge_base import update_knowledge_base, get_similar_documents
from schedule_scraper import get_resolved_threads, scrape_resolved_threads, scrape_documentation, load_documents, save_documents, update_faiss_index
from sentence_transformers import SentenceTransformer
from documentation_scraper import scrape_documentation
from discord.ui import Button, View
from chat_history import ChatHistory
from collections import defaultdict
from discord import app_commands
from discord.ext import commands

# Suppress unnecessary warnings and info logs
logging.getLogger('discord').setLevel(logging.ERROR)
logging.getLogger('discord.gateway').setLevel(logging.ERROR)
logging.getLogger('discord.client').setLevel(logging.ERROR)

# Only show our application logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
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
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

chat_history = ChatHistory()
AUTHORIZED_CORRECTORS = set(int(id.strip()) for id in os.getenv('AUTHORIZED_CORRECTORS', '').split(',') if id.strip())
user_history = defaultdict(list)
response_rankings = defaultdict(lambda: {'thumbs_down': 0, 'thumbs_up': 0})

TRUSTED_USER_IDS = [
    # Add trusted user IDs here
    123456789,  # Example ID
]

SYSTEM_PROMPT = """You are a helpful assistant for Savvyguides, Saltbox and related applications.
Use the provided context to answer questions accurately and concisely. 
If you're not sure about something, say so rather than making assumptions."""

def is_trusted_user(interaction: discord.Interaction) -> bool:
    return interaction.user.id in TRUSTED_USER_IDS

async def fetch_url_content(url):
    """Fetch content from a URL and return it as a document."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return [{'text': response.text, 'source': url}]
    except Exception as e:
        logging.error(f"Error fetching URL content: {e}")
        raise

@bot.tree.command(name="add_knowledge", description="Add documents or URLs to knowledge base")
async def add_knowledge(interaction: discord.Interaction, content: str):
    if not is_trusted_user(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
        
    try:
        if content.startswith(('http://', 'https://')):
            # Handle URL
            documents = await fetch_url_content(content)
        else:
            # Handle direct text
            documents = [{'text': content, 'source': f'Added by {interaction.user.name}'}]
            
        # Update knowledge base
        current_docs = await load_documents()
        current_docs.extend(documents)
        await save_documents(current_docs)
        await update_faiss_index(current_docs)
        
        await interaction.response.send_message("Knowledge base updated successfully!", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"Error adding to knowledge base: {str(e)}", ephemeral=True)

@bot.tree.command(name="forget", description="Forget chat history for current user")
async def forget(interaction: discord.Interaction):
    if not is_trusted_user(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
        
    try:
        user_id = interaction.user.id
        if user_id in user_history:
            del user_history[user_id]
            await interaction.response.send_message("Chat history cleared successfully!", ephemeral=True)
        else:
            await interaction.response.send_message("No chat history found.", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"Error clearing history: {str(e)}", ephemeral=True)

@bot.tree.command(name="new_chat", description="Start a new chat while preserving old context")
async def new_chat(interaction: discord.Interaction):
    if not is_trusted_user(interaction):
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
        
    try:
        user_id = interaction.user.id
        if user_id in user_history:
            # Archive current history if needed
            archived_history = user_history[user_id]
            # Start fresh conversation while keeping context
            user_history[user_id] = []
            await interaction.response.send_message("Started new chat session!", ephemeral=True)
        else:
            await interaction.response.send_message("No active chat found to reset.", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"Error starting new chat: {str(e)}", ephemeral=True)

# Add this after bot setup
@bot.event
async def on_ready():
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Error syncing commands: {e}")

    logging.info(f'Bot is ready and logged in as {bot.user.name}')
    logging.info(f'Bot ID: {bot.user.id}')
    logging.info('Connected to guilds: ' + ', '.join([guild.name for guild in bot.guilds]))
    
    # Set custom status
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.watching, 
        name="for your messages"
    ))

    # Initialize bot and update knowledge base
    await initialize_bot()

async def get_ollama_response(prompt, context):
    """Get a response from Ollama using the context."""
    try:
        # Format the context string
        formatted_context = ""
        if isinstance(context, list):
            formatted_context = "\n\n".join(str(doc) for doc in context)
        else:
            formatted_context = str(context)
        
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
        # Load existing documents
        documents = await load_documents() or []
        initial_count = len(documents)
        logging.info(f"Loaded {initial_count} existing documents")
        
        updated = False
        
        # Scrape documentation if URLs configured
        doc_urls = os.getenv('DOCUMENTATION_URLS')
        if doc_urls:
            logging.info(f"Starting documentation scraping from URLs: {doc_urls}")
            doc_urls_list = [url.strip() for url in doc_urls.split(',') if url.strip()]
            if doc_urls_list:
                doc_documents = await scrape_documentation(doc_urls_list)
                if doc_documents:
                    # Append new documents
                    documents = await save_documents(doc_documents, append=True)
                    updated = True
                    logging.info(f"Added documentation documents")
        
        # Scrape resolved threads
        logging.info("Starting thread scraping")
        thread_count = await scrape_resolved_threads(bot)  # Use client instead of self
        if thread_count > 0:
            updated = True
            documents = await load_documents()  # Reload after thread scraping
            
        # Update index if we have new documents
        if updated:
            await update_faiss_index(documents)
            logging.info(f"Updated index with {len(documents)} documents")
        else:
            logging.info("No new documents to index")
            
    except Exception as e:
        logging.error(f"Error during initialization: {e}")

async def handle_message(message):
    user_id = message.author.id
    
    # Get history for this specific user
    history = user_history[user_id]
    
    # Add new message to history
    history.append({
        'role': 'user',
        'content': message.content
    })
    
    # Generate response
    response = await generate_response(message.content, history, response_rankings)
    
    # Add response to history
    history.append({
        'role': 'assistant',
        'content': response
    })
    
    # Keep only last N messages in history
    user_history[user_id] = history[-10:]  # Keeps last 10 messages
    
    # Send response and add reaction buttons
    bot_message = await message.channel.send(response)
    await bot_message.add_reaction("👍")
    await bot_message.add_reaction("👎")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Handle corrections from authorized users
    if message.reference and message.author.id in AUTHORIZED_CORRECTORS:
        referenced_message = await message.channel.fetch_message(message.reference.message_id)
        if referenced_message.author == bot.user:
            chat_history.add_correction(str(referenced_message.id), message.content)
            await message.add_reaction('✅')
            # Update knowledge base with correction
            await update_knowledge_base_with_correction(referenced_message.content, message.content)
            return

    # Regular message handling
    if bot.user in message.mentions or message.content.startswith('!'):
        content = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        if message.content.startswith('!'):
            content = content[1:].strip()
        
        if not content:
            await message.channel.send("How can I help you?")
            return
            
        await handle_message(message)

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:  # Ignore bot's own reactions
        return
            
    if reaction.message.author == bot.user:  # Only handle reactions to bot messages
        if str(reaction.emoji) == "👍":
            response_rankings[reaction.message.content]['thumbs_up'] += 1
        elif str(reaction.emoji) == "👎":
            response_rankings[reaction.message.content]['thumbs_down'] += 1
            # Remove from user's history
            user_history[user.id] = [msg for msg in user_history[user.id] 
                                   if msg['content'] != reaction.message.content]
            await update_knowledge_base_with_correction(reaction.message.content, "")

async def process_resolved_thread(thread):
    """Process a resolved thread and add it to the knowledge base."""
    try:
        # Get the thread title and all messages
        thread_title = thread.name
        messages = []
        async for message in thread.history(limit=None, oldest_first=True):
            if not message.author.bot:  # Skip bot messages
                messages.append(message.content)
        
        if not messages:
            return
            
        # Format the thread content
        thread_content = {
            "text": f"Question: {thread_title}\n\nAnswer: {' '.join(messages)}",
            "source": f"Discord Thread: {thread_title}"
        }
        
        # Load existing documents
        documents = await load_documents()
        
        # Add the thread content if it's not already present
        if not any(doc["source"] == thread_content["source"] for doc in documents):
            documents.append(thread_content)
            await save_documents(documents)
            logging.info(f"Added resolved thread to knowledge base: {thread_title}")
            
    except Exception as e:
        logging.error(f"Error processing resolved thread {thread.name}: {e}")

async def update_knowledge_base_with_correction(original_response, correction):
    """Update knowledge base with corrected information."""
    try:
        documents = await load_documents()
        documents.append({
            "text": f"Incorrect: {original_response}\nCorrection: {correction}",
            "source": "User Corrections"
        })
        await save_documents(documents)
        await update_faiss_index(documents)
    except Exception as e:
        logging.error(f"Error updating knowledge base with correction: {e}")

async def get_embedding(text):
    """Convert text to embedding using the sentence transformer model."""
    try:
        return model.encode([text])[0]
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

async def load_faiss_index():
    """Load the FAISS index from disk."""
    try:
        if os.path.exists(INDEX_PATH):
            return faiss.read_index(INDEX_PATH)
        return None
    except Exception as e:
        logging.error(f"Error loading FAISS index: {e}")
        return None

async def get_similar_responses(query, top_k=5):
    try:
        documents = await load_documents()
        if not documents:
            return []
            
        # Convert query to embedding
        query_embedding = await get_embedding(query)
        
        # Search for similar responses
        faiss_index = await load_faiss_index()
        if not faiss_index:
            return []
            
        D, I = faiss_index.search(np.array([query_embedding]), top_k)
        
        # Get the similar responses
        similar_responses = [
            {'text': documents[i]['text'], 'source': documents[i]['source']}
            for i in I[0] if i < len(documents)
        ]
        
        return similar_responses
        
    except Exception as e:
        logging.error(f"Error getting similar responses: {e}")
        return []

async def generate_response(message, history, rankings=None):
    try:
        # Get similar responses from knowledge base
        similar_responses = await get_similar_responses(message)
        
        # Sort responses by ranking score if rankings exist
        if rankings and similar_responses:
            for response in similar_responses:
                rank_data = rankings.get(response['text'], {'thumbs_up': 0, 'thumbs_down': 0})
                total = rank_data['thumbs_up'] + rank_data['thumbs_down'] + 1
                response['score'] = (rank_data['thumbs_up'] - rank_data['thumbs_down']) / total
            
            similar_responses.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Add context from similar responses
        context = ""
        if similar_responses:
            context = "Previous relevant responses:\n" + "\n".join(
                f"[Score: {r.get('score', 0):.2f}] {r['text']}" 
                for r in similar_responses[:3]
            )

        # Format the full prompt
        full_prompt = f"{context}\n\nQuestion: {message}"
        
        # Get response from Ollama
        response = requests.post(
            f"{OLLAMA_SERVER_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "system": SYSTEM_PROMPT,
                "stream": False
            },
            timeout=30
        )
        
        response.raise_for_status()
        response_text = response.json()['response']
        
        # Split response if it's too long
        if len(response_text) > 1900:  # Leave room for formatting
            return response_text[:1900] + "\n... (response truncated)"
            
        return response_text
        
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I encountered an error while processing your request."

@bot.event
async def on_disconnect():
    logging.warning("Bot disconnected from Discord - attempting to reconnect...")

@bot.event
async def on_connect():
    logging.info("Bot reconnected to Discord successfully")

@bot.event
async def on_resumed():
    logging.info("Bot session resumed")

# Update the bot.run() call at the bottom of the file
bot.run(DISCORD_TOKEN, reconnect=True)
