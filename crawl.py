import requests
from bs4 import BeautifulSoup
import json
import time
import random
import urllib.robotparser
from urllib.parse import urlparse, urljoin

class Crawler:
    def __init__(self, seed_urls, max_pages=100, delay=1.0):
        """
        Initialize the crawler with seed URLs and constraints.
        
        Args:
            seed_urls (list): Initial URLs to start crawling from
            max_pages (int): Maximum number of pages to crawl
            delay (float): Delay between requests in seconds
        """
        self.seed_urls = seed_urls
        self.max_pages = max_pages
        self.delay = delay
        self.visited = set()
        self.to_visit = set(seed_urls)
        self.crawled_data = []
        self.robot_parsers = {}
    
    def _is_allowed(self, url):
        """Check if URL is allowed by robots.txt"""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self.robot_parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(base_url, "/robots.txt"))
            try:
                rp.read()
                self.robot_parsers[base_url] = rp
            except Exception:
                # If robots.txt can't be read, assume crawling is allowed
                return True
        
        return self.robot_parsers[base_url].can_fetch("*", url)
    
    def _extract_links(self, soup, base_url):
        """Extract links from a page"""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            # Filter for only http/https URLs
            if full_url.startswith(('http://', 'https://')):
                links.add(full_url)
        return links
    
    def crawl(self):
        """Start the crawling process"""
        crawl_count = 0
        
        while self.to_visit and crawl_count < self.max_pages:
            # Get a URL from the queue
            url = self.to_visit.pop()
            if url in self.visited:
                continue
                
            if not self._is_allowed(url):
                print(f"Skipping {url} (disallowed by robots.txt)")
                continue
                
            try:
                # Introduce delay
                time.sleep(self.delay + random.uniform(0.1, 0.5))
                
                # Make the request
                print(f"Crawling: {url}")
                response = requests.get(url, headers={
                    'User-Agent': 'VitaCarnisResearchBot/1.0'
                }, timeout=10)
                
                if response.status_code == 200:
                    # Parse the page
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Store data
                    title = soup.title.string if soup.title else "No title"
                    
                    # Extract main content (simplistic approach)
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                        
                    # Get text
                    content = soup.get_text(separator=' ', strip=True)
                    
                    # Store the crawled data
                    self.crawled_data.append({
                        'url': url,
                        'title': title,
                        'content': content,
                        'timestamp': time.time()
                    })
                    
                    # Find links on the page
                    new_links = self._extract_links(soup, url)
                    
                    # Add new links to the queue
                    for link in new_links:
                        if link not in self.visited:
                            self.to_visit.add(link)
                    
                    # Mark URL as visited
                    self.visited.add(url)
                    crawl_count += 1
                    print(f"Processed {url} ({crawl_count}/{self.max_pages})")
                    
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                
        # Save results
        self.save_results()
        return self.crawled_data
    
    def save_results(self):
        """Save the crawled data to a JSON file in the carnis_data folder"""
        import os
        
        # Ensure the carnis_data directory exists
        os.makedirs('carnis_data', exist_ok=True)

        # Ensure the crawl directory with the crawled_data file exists
        os.makedirs(os.path.join('carnis_data', 'crawl'), exist_ok=True)
        
        # Save the data to the carnis_data folder
        output_path = os.path.join('carnis_data', 'crawl', 'crawled_data.json')
        with open(output_path, 'w') as f:
            json.dump(self.crawled_data, f, indent=4)
        print(f"Crawl complete. Collected data from {len(self.crawled_data)} pages.")
        print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    seeds = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Biology",
        "https://en.wikipedia.org/wiki/Biotechnology"
    ]
    
    crawler = Crawler(seeds, max_pages=50, delay=1.5)
    crawler.crawl()