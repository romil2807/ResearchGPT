import requests
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import time
from urllib.parse import quote_plus
import re

class ExternalSearchManager:
    """
    Manages external search capabilities for the research system.
    Supports DuckDuckGo search to supplement research paper knowledge.
    """
    
    def __init__(self, config: dict, logger=None):
        """
        Initialize the external search manager with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
            logger: Optional logger for logging
        """
        self.config = config
        self.logger = logger
        
        # Get external search settings
        self.search_config = config.get('external_search', {})
        self.enabled = self.search_config.get('enabled', False)
        self.search_engine = self.search_config.get('search_engine', 'duckduckgo')
        self.max_results = self.search_config.get('max_results', 5)
        self.include_snippets = self.search_config.get('include_snippets', True)
        self.include_urls = self.search_config.get('include_urls', True)
        self.safe_search = self.search_config.get('safe_search', True)
        self.region = self.search_config.get('region', 'wt-wt')
        self.time_limit = self.search_config.get('time_limit', 30)
        
        if self.logger:
            self.logger.info(f"External search manager initialized with engine: {self.search_engine}")
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform an external search using the configured search engine.
        
        Args:
            query: The search query
            
        Returns:
            List of search results as dictionaries
        """
        if not self.enabled:
            if self.logger:
                self.logger.warning("External search is disabled in config")
            return []
            
        if self.search_engine == 'duckduckgo':
            return self._search_duckduckgo(query)
        else:
            if self.logger:
                self.logger.warning(f"Unsupported search engine: {self.search_engine}")
            return []
    
    def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a search using DuckDuckGo.
        
        Args:
            query: The search query
            
        Returns:
            List of search results as dictionaries
        """
        try:
            # Prepare query string
            encoded_query = quote_plus(query)
            
            # Set up parameters
            params = {
                'q': encoded_query,
                'format': 'json',
                'no_html': '1',
                'no_redirect': '1',
                't': 'ResearchGPT',
                'kl': self.region,  # Region parameter
            }
            
            # Add safe search if enabled
            if self.safe_search:
                params['kp'] = '1'
                
            # Make the request
            url = "https://api.duckduckgo.com/"
            start_time = time.time()
            response = requests.get(url, params=params, timeout=self.time_limit)
            
            if response.status_code != 200:
                if self.logger:
                    self.logger.error(f"DuckDuckGo search failed with status code: {response.status_code}")
                return []
                
            # Parse JSON response
            data = response.json()
            
            # Extract results
            results = []
            
            # Add the Abstract/Instant Answer if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Instant Answer'),
                    'snippet': data.get('Abstract'),
                    'url': data.get('AbstractURL'),
                    'source': 'DuckDuckGo Instant Answer'
                })
            
            # Add Related Topics
            for topic in data.get('RelatedTopics', [])[:self.max_results]:
                # Skip category topics
                if 'Topics' in topic:
                    continue
                    
                # Extract title from text (usually formatted as "Title - Description")
                text = topic.get('Text', '')
                title_match = re.match(r'^(.*?) - ', text)
                title = title_match.group(1) if title_match else text[:50]
                
                # Extract snippet (description)
                snippet = text[len(title) + 3:] if title_match else text
                
                results.append({
                    'title': title,
                    'snippet': snippet,
                    'url': topic.get('FirstURL'),
                    'source': 'DuckDuckGo'
                })
                
                if len(results) >= self.max_results:
                    break
            
            # Log search completion
            if self.logger:
                elapsed_time = time.time() - start_time
                self.logger.info(f"DuckDuckGo search completed in {elapsed_time:.2f}s with {len(results)} results")
            
            return results
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in DuckDuckGo search: {str(e)}")
            return []
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for inclusion in prompts.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted results string
        """
        if not results:
            return ""
            
        formatted = "External search results:\n\n"
        
        for i, result in enumerate(results):
            # Add title
            formatted += f"{i+1}. {result.get('title', 'No title')}\n"
            
            # Add URL if configured
            if self.include_urls and result.get('url'):
                formatted += f"   URL: {result.get('url')}\n"
                
            # Add snippet if configured
            if self.include_snippets and result.get('snippet'):
                formatted += f"   {result.get('snippet')}\n"
                
            formatted += "\n"
            
        return formatted
    
    def search_and_format(self, query: str) -> str:
        """
        Search and return formatted results in one step.
        
        Args:
            query: The search query
            
        Returns:
            Formatted search results string
        """
        results = self.search(query)
        return self.format_results(results) 