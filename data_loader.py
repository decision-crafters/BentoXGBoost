import os
import zipfile
import requests
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to handle downloading and processing data from different sources.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory to store downloaded data. If None, a temporary directory will be used.
        """
        if data_dir is None:
            self.data_dir = tempfile.mkdtemp()
            self.temp_dir_created = True
        else:
            self.data_dir = data_dir
            os.makedirs(self.data_dir, exist_ok=True)
            self.temp_dir_created = False
        
        logger.info(f"Data will be stored in {self.data_dir}")
    
    def __del__(self):
        """Clean up temporary directory if created."""
        if hasattr(self, 'temp_dir_created') and self.temp_dir_created:
            try:
                shutil.rmtree(self.data_dir)
                logger.info(f"Removed temporary directory {self.data_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {self.data_dir}: {e}")
    
    def download_file(self, url: str, filename: Optional[str] = None) -> str:
        """
        Download a file from a URL.
        
        Args:
            url: URL to download from
            filename: Name to save the file as. If None, will be derived from the URL.
            
        Returns:
            Path to the downloaded file
        """
        if filename is None:
            filename = url.split('/')[-1]
        
        file_path = os.path.join(self.data_dir, filename)
        
        logger.info(f"Downloading {url} to {file_path}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {url} to {file_path}")
        
        return file_path
    
    def extract_zip(self, zip_path: str, extract_dir: Optional[str] = None) -> str:
        """
        Extract a ZIP file.
        
        Args:
            zip_path: Path to the ZIP file
            extract_dir: Directory to extract to. If None, will extract to a directory with the same name as the ZIP file.
            
        Returns:
            Path to the extracted directory
        """
        if extract_dir is None:
            extract_dir = os.path.join(self.data_dir, os.path.splitext(os.path.basename(zip_path))[0])
        
        os.makedirs(extract_dir, exist_ok=True)
        
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info(f"Extracted {zip_path} to {extract_dir}")
        
        return extract_dir
    
    def download_and_extract_zip(self, url: str, extract_dir: Optional[str] = None) -> str:
        """
        Download a ZIP file and extract it.
        
        Args:
            url: URL to download from
            extract_dir: Directory to extract to. If None, will extract to a directory with the same name as the ZIP file.
            
        Returns:
            Path to the extracted directory
        """
        zip_path = self.download_file(url)
        return self.extract_zip(zip_path, extract_dir)
    
    def fetch_web_content(self, url: str) -> str:
        """
        Fetch content from a web page.
        
        Args:
            url: URL to fetch from
            
        Returns:
            Content of the web page
        """
        logger.info(f"Fetching content from {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        logger.info(f"Fetched content from {url}")
        
        return response.text
    
    def convert_html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to markdown.
        
        Args:
            html_content: HTML content to convert
            
        Returns:
            Markdown content
        """
        try:
            import html2text
        except ImportError:
            logger.warning("html2text not installed. Installing it now...")
            import subprocess
            subprocess.check_call(["pip", "install", "html2text"])
            import html2text
        
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        
        return h.handle(html_content)
    
    def fetch_and_convert_to_markdown(self, url: str) -> str:
        """
        Fetch content from a web page and convert it to markdown.
        
        Args:
            url: URL to fetch from
            
        Returns:
            Markdown content
        """
        html_content = self.fetch_web_content(url)
        return self.convert_html_to_markdown(html_content)
    
    def crawl_website(self, url: str, max_pages: int = 10) -> List[str]:
        """
        Crawl a website and return the content of each page.
        
        Args:
            url: URL to start crawling from
            max_pages: Maximum number of pages to crawl
            
        Returns:
            List of page contents
        """
        try:
            from firecrawl import Crawler
        except ImportError:
            logger.warning("firecrawl not installed. Installing it now...")
            import subprocess
            subprocess.check_call(["pip", "install", "firecrawl"])
            from firecrawl import Crawler
        
        logger.info(f"Crawling {url} (max {max_pages} pages)")
        
        crawler = Crawler(max_pages=max_pages)
        pages = crawler.crawl(url)
        
        logger.info(f"Crawled {len(pages)} pages from {url}")
        
        return [page.content for page in pages]
