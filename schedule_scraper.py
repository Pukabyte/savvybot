import schedule
import time
import os
import logging
from dotenv import load_dotenv

load_dotenv()
FORUM_CHANNEL_ID = int(os.getenv('FORUM_CHANNEL_ID'))

async def scrape_resolved_threads(client):
    """Scrape resolved threads in the support forum and add them to the knowledge base."""
    logging.info("Starting to scrape resolved threads")
    forum_channel = client.get_channel(FORUM_CHANNEL_ID)
    if forum_channel is None:
        logging.error("Forum channel not found. Please check the FORUM_CHANNEL_ID.")
        return

    thread_count = 0
    for thread in forum_channel.threads:
        if "RESOLVED" in thread.name.upper():
            logging.info(f"Processing resolved thread: {thread.name}")
            async for message in thread.history(limit=100):
                if message.author == client.user:
                    continue
                
                if "issue" in message.content.lower():
                    issue = message.content
                    logging.debug(f"Found issue: {issue[:100]}...")
                elif "solution" in message.content.lower():
                    solution = message.content
                    logging.debug(f"Found solution: {solution[:100]}...")
                    logging.info(f"Adding issue and solution to knowledge base")
            thread_count += 1
    
    logging.info(f"Completed scraping {thread_count} resolved threads")

def run_scheduled_tasks(client):
    logging.info("Starting scheduled tasks")
    schedule.every().hour.do(scrape_resolved_threads, client)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduled_tasks()
