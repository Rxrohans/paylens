"""
exchange_rate_fetcher.py - Fetch real-time exchange rates
Uses direct web scraping when search engines fail to provide actual rates.

Author: Rohan Singh
Created: March 2026 for PayLens v3
"""

import re
import logging
from typing import Optional, Dict
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("PayLens.exchange_rate")


class ExchangeRateFetcher:
    """Fetches current exchange rates from reliable sources."""
    
    # Free, reliable sources for exchange rates
    SOURCES = {
        "google_finance": "https://www.google.com/finance/quote/{source}-{target}",
        "xe_com": "https://www.xe.com/currencyconverter/convert/?Amount=1&From={source}&To={target}",
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_rate(self, source_currency: str, target_currency: str = "INR") -> Optional[Dict]:
        """
        Fetch exchange rate from multiple sources.
        
        Args:
            source_currency: Source currency code (USD, EUR, GBP)
            target_currency: Target currency code (default: INR)
            
        Returns:
            Dict with {rate: float, source: str, url: str, pair: str} or None
        """
        source_currency = source_currency.upper()
        target_currency = target_currency.upper()
        
        logger.info(f"Fetching exchange rate: {source_currency} → {target_currency}")
        
        # Try Google Finance first (most reliable)
        rate = self._fetch_from_google_finance(source_currency, target_currency)
        if rate:
            return rate
        
        # Fallback to XE.com
        rate = self._fetch_from_xe(source_currency, target_currency)
        if rate:
            return rate
        
        # Fallback to free API
        rate = self._fetch_from_api(source_currency, target_currency)
        if rate:
            return rate
        
        logger.warning(f"Failed to fetch {source_currency}-{target_currency} rate from all sources")
        return None
    
    def _fetch_from_google_finance(self, source: str, target: str) -> Optional[Dict]:
        """Fetch rate from Google Finance by parsing HTML."""
        try:
            url = self.SOURCES["google_finance"].format(source=source, target=target)
            logger.debug(f"Trying Google Finance: {url}")
            
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Google Finance shows rate in a div with class "YMlKec fxKbKc"
            rate_elem = soup.find('div', class_='YMlKec fxKbKc')
            if rate_elem:
                rate_text = rate_elem.text.strip()
                rate = float(rate_text.replace(',', ''))
                
                logger.info(f"✓ Google Finance: 1 {source} = {rate} {target}")
                return {
                    'rate': rate,
                    'source': 'Google Finance',
                    'url': url,
                    'pair': f"{source}/{target}"
                }
            
            # Fallback: Search for pattern "1 USD = X.XX INR" in page text
            text = soup.get_text()
            match = re.search(r'1\s+' + source + r'\s*=\s*([\d,\.]+)\s+' + target, text, re.IGNORECASE)
            if match:
                rate = float(match.group(1).replace(',', ''))
                logger.info(f"✓ Google Finance (fallback): 1 {source} = {rate} {target}")
                return {
                    'rate': rate,
                    'source': 'Google Finance',
                    'url': url,
                    'pair': f"{source}/{target}"
                }
            
            logger.debug(f"Could not parse rate from Google Finance")
            return None
            
        except Exception as e:
            logger.debug(f"Google Finance failed: {e}")
            return None
    
    def _fetch_from_xe(self, source: str, target: str) -> Optional[Dict]:
        """Fetch rate from XE.com by parsing HTML."""
        try:
            url = self.SOURCES["xe_com"].format(source=source, target=target)
            logger.debug(f"Trying XE.com: {url}")
            
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # XE.com shows the rate - look for numeric values
            # Pattern: "1 USD = X.XX INR"
            text = soup.get_text()
            match = re.search(r'1\s+' + source + r'\s*=\s*([\d,\.]+)\s+' + target, text, re.IGNORECASE)
            if match:
                rate = float(match.group(1).replace(',', ''))
                logger.info(f"✓ XE.com: 1 {source} = {rate} {target}")
                return {
                    'rate': rate,
                    'source': 'XE.com',
                    'url': url,
                    'pair': f"{source}/{target}"
                }
            
            logger.debug(f"Could not parse rate from XE.com")
            return None
            
        except Exception as e:
            logger.debug(f"XE.com failed: {e}")
            return None
    
    def _fetch_from_api(self, source: str, target: str) -> Optional[Dict]:
        """
        Fetch from free exchangerate-api.com (no API key required).
        This is the most reliable fallback.
        """
        try:
            url = f"https://open.er-api.com/v6/latest/{source}"
            logger.debug(f"Trying ExchangeRate-API: {url}")
            
            response = self.session.get(url, timeout=5)
            data = response.json()
            
            if data.get('result') == 'success':
                rates = data.get('rates', {})
                rate = rates.get(target)
                
                if rate:
                    logger.info(f"✓ ExchangeRate-API: 1 {source} = {rate} {target}")
                    return {
                        'rate': float(rate),
                        'source': 'ExchangeRate-API',
                        'url': url,
                        'pair': f"{source}/{target}"
                    }
            
            logger.debug(f"Could not get rate from ExchangeRate-API")
            return None
            
        except Exception as e:
            logger.debug(f"ExchangeRate-API failed: {e}")
            return None


# Utility function for easy import
def get_exchange_rate(source_currency: str, target_currency: str = "INR") -> Optional[Dict]:
    """
    Quick function to get exchange rate.
    
    Usage:
        >>> rate_info = get_exchange_rate("USD", "INR")
        >>> if rate_info:
        ...     print(f"1 USD = {rate_info['rate']} INR (from {rate_info['source']})")
    
    Args:
        source_currency: Source currency code (USD, EUR, GBP, etc.)
        target_currency: Target currency code (default: INR)
        
    Returns:
        Dict with rate info or None if fetch failed
    """
    fetcher = ExchangeRateFetcher()
    return fetcher.fetch_rate(source_currency, target_currency)


# Test if run directly
if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s | %(message)s'
    )
    
    print("\n" + "="*70)
    print("Exchange Rate Fetcher Test")
    print("="*70 + "\n")
    
    test_pairs = [
        ("USD", "INR"),
        ("EUR", "INR"),
        ("GBP", "INR"),
    ]
    
    results = []
    
    for source, target in test_pairs:
        print(f"Fetching {source} → {target}...")
        rate_info = get_exchange_rate(source, target)
        
        if rate_info:
            print(f"  ✓ SUCCESS: 1 {source} = {rate_info['rate']:.4f} {target}")
            print(f"  Source: {rate_info['source']}")
            print(f"  URL: {rate_info['url'][:60]}...")
            results.append(True)
        else:
            print(f"  ✗ FAILED to fetch {source}/{target}")
            results.append(False)
        print()
    
    # Summary
    success_count = sum(results)
    total_count = len(results)
    
    print("="*70)
    print(f"Results: {success_count}/{total_count} successful")
    print("="*70)
    
    # Exit code: 0 if all succeeded, 1 if any failed
    sys.exit(0 if all(results) else 1)