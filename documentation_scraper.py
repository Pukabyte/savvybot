import logging
import aiohttp
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse

async def get_all_links(soup, base_url):
    """Get all links from the page that belong to the same domain."""
    domain = urlparse(base_url).netloc
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == domain:
            links.add(full_url)
    return links

async def scrape_documentation(urls):
    """Scrape documentation from provided URLs and their internal links."""
    documents = []
    processed_urls = set()
    
    async with aiohttp.ClientSession() as session:
        async def process_url(url):
            if url in processed_urls:
                return
            processed_urls.add(url)
            
            try:
                logging.info(f"Scraping documentation from: {url}")
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Get all text from the page
                        text = soup.get_text(separator=' ', strip=True)
                        if text:
                            document = {
                                "text": text,
                                "source": f"Documentation: {url}"
                            }
                            documents.append(document)
                            logging.info(f"Added content from {url}")
                            
                        # Get and process all internal links
                        links = await get_all_links(soup, url)
                        for link in links:
                            await process_url(link)
                            
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")
                
        # Process initial URLs
        for url in urls:
            await process_url(url.strip())
                
    logging.info(f"Completed documentation scraping. Found {len(documents)} documents")
    return documents 