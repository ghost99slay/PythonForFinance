"""
Index Service
Handles fetching and validating index constituents for portfolio analysis.
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexService:
    """Service for fetching and validating index constituents."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        
        # Index information and URLs for web scraping
        self.index_info = {
            'SP500': {
                'name': 'S&P 500 Index',
                'description': 'Standard & Poor\'s 500 Index - 500 largest US companies by market cap',
                'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                'table_id': 'constituents',
                'ticker_column': 0  # First column contains tickers
            },
            'NASDAQ100': {
                'name': 'NASDAQ-100 Index', 
                'description': 'NASDAQ-100 Index - 100 largest non-financial companies on NASDAQ',
                'url': 'https://en.wikipedia.org/wiki/NASDAQ-100',
                'table_class': 'wikitable sortable',
                'ticker_column': 1  # Second column contains tickers
            },
            'DOW30': {
                'name': 'Dow Jones Industrial Average',
                'description': 'Dow Jones Industrial Average - 30 large US companies',
                'url': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
                'table_class': 'wikitable sortable',
                'ticker_column': 0  # First column contains tickers
            },
            'RUSSELL2000': {
                'name': 'Russell 2000 Index',
                'description': 'Russell 2000 Index - Small-cap US companies (approximately 2000 stocks)',
                'fallback_tickers': [
                    'AMC', 'SIRI', 'PLUG', 'IRM', 'FUBO', 'RIG', 'GEVO', 'BLNK', 'SPCE', 'RIDE',
                    'WKHS', 'CLNE', 'CLOV', 'WISH', 'ROOT', 'OPEN', 'RBLX', 'HOOD', 'SOFI', 'LCID',
                    'RIVN', 'AFRM', 'UPST', 'COIN', 'DKNG', 'PLTR', 'SNOW', 'ZM', 'PTON', 'ROKU',
                    'SQ', 'SHOP', 'SPOT', 'UBER', 'LYFT', 'DASH', 'ABNB', 'PINS', 'BYND', 'PETN'
                ]
            },
            'RUSSELL1000': {
                'name': 'Russell 1000 Index', 
                'description': 'Russell 1000 Index - Large-cap US companies (approximately 1000 stocks)',
                'fallback_tickers': [
                    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'TSLA', 'META', 'UNH', 'JNJ',
                    'XOM', 'JPM', 'V', 'PG', 'MA', 'CVX', 'HD', 'ABBV', 'LLY', 'PFE',
                    'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'MRK'
                ]
            }
        }
    
    def get_available_indices(self) -> Dict[str, Dict[str, str]]:
        """
        Get list of available indices for selection.
        
        Returns:
            Dictionary of index symbols and their metadata
        """
        return {
            symbol: {
                'name': data['name'],
                'description': data['description']
            }
            for symbol, data in self.index_info.items()
        }
    
    def _scrape_sp500_constituents(self) -> List[str]:
        """Scrape S&P 500 constituents from Wikipedia."""
        try:
            logger.info("Scraping S&P 500 constituents from Wikipedia")
            
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(self.index_info['SP500']['url'], headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            
            if not table:
                raise ValueError("Could not find S&P 500 constituents table")
            
            tickers = []
            for row in table.find_all('tr')[1:]:  # Skip header
                cells = row.find_all('td')
                if cells:
                    ticker = cells[0].text.strip()
                    # Clean up ticker (remove any extra characters)
                    ticker = re.sub(r'[^A-Z.-]', '', ticker.upper())
                    if ticker and len(ticker) <= 5:  # Valid ticker length
                        tickers.append(ticker)
            
            logger.info(f"Successfully scraped {len(tickers)} S&P 500 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to scrape S&P 500 constituents: {e}")
            # Return a comprehensive fallback list of major S&P 500 stocks
            return [
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'META', 'UNH', 'JNJ', 'XOM',
                'JPM', 'V', 'PG', 'MA', 'CVX', 'HD', 'ABBV', 'LLY', 'PFE', 'BAC',
                'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'MRK', 'NFLX',
                'VZ', 'ADBE', 'CMCSA', 'NKE', 'CRM', 'ACN', 'DHR', 'TXN', 'NEE', 'RTX',
                'QCOM', 'LIN', 'PM', 'UPS', 'HON', 'ORCL', 'T', 'LOW', 'SPGI', 'INTU',
                'CAT', 'GS', 'AXP', 'BKNG', 'DE', 'AMD', 'BLK', 'ELV', 'GILD', 'MDLZ',
                'ADP', 'TJX', 'VRTX', 'ADI', 'LRCX', 'SYK', 'PANW', 'AMAT', 'C', 'MU',
                'PYPL', 'SCHW', 'TMUS', 'ISRG', 'NOW', 'ZTS', 'CB', 'BSX', 'REGN', 'SO',
                'PLD', 'BMY', 'ITW', 'DUK', 'EOG', 'WM', 'MMC', 'CSX', 'PNC', 'CL',
                'APH', 'FI', 'EMR', 'USB', 'NSC', 'AON', 'GE', 'PSA', 'WELL', 'COP',
                'MPC', 'SHW', 'FCX', 'GM', 'TGT', 'F', 'AIG', 'JCI', 'PCAR', 'NEM',
                'KMB', 'VLO', 'PAYX', 'GIS', 'OXY', 'DFS', 'BDX', 'GLW', 'CTAS', 'CMG',
                'ADSK', 'ADM', 'ROP', 'ROST', 'NXPI', 'KHC', 'MCHP', 'EW', 'FAST', 'CTSH',
                'DXCM', 'EA', 'VRSK', 'LULU', 'ODFL', 'EXC', 'AEP', 'IDXX', 'FANG', 'XEL',
                'KMI', 'CCEP', 'CSGP', 'ON', 'ANSS', 'WBD', 'GPN', 'BIIB', 'IQV', 'FTNT'
            ]
    
    def _scrape_nasdaq100_constituents(self) -> List[str]:
        """Scrape NASDAQ-100 constituents from Wikipedia."""
        try:
            logger.info("Scraping NASDAQ-100 constituents from Wikipedia")
            
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(self.index_info['NASDAQ100']['url'], headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table', class_='wikitable sortable')
            
            tickers = []
            for table in tables:
                for row in table.find_all('tr')[1:]:  # Skip header
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        ticker = cells[1].text.strip()  # Second column typically has ticker
                        # Clean up ticker
                        ticker = re.sub(r'[^A-Z.-]', '', ticker.upper())
                        if ticker and len(ticker) <= 5:
                            tickers.append(ticker)
            
            # Remove duplicates and sort
            tickers = sorted(list(set(tickers)))
            logger.info(f"Successfully scraped {len(tickers)} NASDAQ-100 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to scrape NASDAQ-100 constituents: {e}")
            # Return fallback list
            return [
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'META', 'AVGO', 'COST', 'NFLX',
                'ADBE', 'PEP', 'TMUS', 'CSCO', 'CMCSA', 'TXN', 'QCOM', 'INTC', 'INTU', 'AMD',
                'AMAT', 'ISRG', 'BKNG', 'MU', 'ADI', 'GILD', 'LRCX', 'MELI', 'REGN', 'PYPL',
                'MDLZ', 'KLAC', 'CDNS', 'SNPS', 'MRVL', 'ORLY', 'CTAS', 'DXCM', 'SBUX', 'PCAR',
                'FTNT', 'MNST', 'PAYX', 'FAST', 'ODFL', 'ROST', 'EA', 'CTSH', 'LULU', 'VRSK',
                'EXC', 'XEL', 'DLTR', 'CSGP', 'ANSS', 'BIIB', 'IDXX', 'WBD', 'ZM', 'MRNA'
            ]
    
    def _scrape_dow30_constituents(self) -> List[str]:
        """Scrape Dow 30 constituents from Wikipedia."""
        try:
            logger.info("Scraping Dow 30 constituents from Wikipedia")
            
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(self.index_info['DOW30']['url'], headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table', class_='wikitable sortable')
            
            tickers = []
            for table in tables:
                for row in table.find_all('tr')[1:]:  # Skip header
                    cells = row.find_all('td')
                    if cells:
                        ticker = cells[0].text.strip()  # First column has ticker
                        # Clean up ticker
                        ticker = re.sub(r'[^A-Z.-]', '', ticker.upper())
                        if ticker and len(ticker) <= 5:
                            tickers.append(ticker)
            
            # Remove duplicates and sort
            tickers = sorted(list(set(tickers)))
            logger.info(f"Successfully scraped {len(tickers)} Dow 30 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to scrape Dow 30 constituents: {e}")
            # Return fallback list
            return [
                'UNH', 'GS', 'HD', 'MSFT', 'AMGN', 'MCD', 'V', 'CRM', 'HON', 'CAT',
                'AXP', 'BA', 'TRV', 'AAPL', 'IBM', 'JPM', 'JNJ', 'WMT', 'CVX', 'NKE',
                'PG', 'MRK', 'DIS', 'KO', 'DOW', 'CSCO', 'VZ', 'INTC', 'WBA', 'MMM'
            ]
    
    def get_index_constituents(self, index_symbol: str, max_stocks: int = 1000, skip_validation: bool = False, 
                             start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Get constituents of an index with data validation.
        
        Args:
            index_symbol: Symbol of the index (e.g., 'SP500', 'NASDAQ100')
            max_stocks: Maximum number of stocks to return (for performance)
            skip_validation: If True, skip validation and return tickers directly
            start_date: Start date for data validation
            end_date: End date for data validation
            
        Returns:
            Tuple of (valid_tickers, metadata)
        """
        try:
            # Check cache first (but skip cache if skip_validation is True to avoid cached failures)
            cache_key = f"{index_symbol}_{max_stocks}_{skip_validation}"
            if not skip_validation and cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_timeout:
                    logger.info(f"Returning cached data for {index_symbol}")
                    return data
            
            logger.info(f"Fetching constituents for {index_symbol}")
            
            # Get index data
            if index_symbol not in self.index_info:
                raise ValueError(f"Index {index_symbol} not supported")
            
            index_data = self.index_info[index_symbol]
            
            # Fetch constituents based on index type
            if index_symbol == 'SP500':
                all_tickers = self._scrape_sp500_constituents()
            elif index_symbol == 'NASDAQ100':
                all_tickers = self._scrape_nasdaq100_constituents()
            elif index_symbol == 'DOW30':
                all_tickers = self._scrape_dow30_constituents()
            elif index_symbol in ['RUSSELL2000', 'RUSSELL1000']:
                # Use fallback for Russell indices (too many to scrape efficiently)
                all_tickers = index_data.get('fallback_tickers', [])
            else:
                raise ValueError(f"No scraping method implemented for {index_symbol}")
            
            # Limit for performance if requested
            if max_stocks and max_stocks < len(all_tickers):
                all_tickers = all_tickers[:max_stocks]
            
            # Skip validation if requested (for testing or when validation is problematic)
            if skip_validation:
                logger.info(f"Skipping validation for {index_symbol} as requested")
                valid_tickers = all_tickers
                validation_results = {
                    'valid_tickers': valid_tickers,
                    'invalid_tickers': [],
                    'validation_details': {ticker: {'valid': True, 'reason': 'Validation skipped'} for ticker in valid_tickers},
                    'success_rate': 1.0,
                    'validation_skipped': True
                }
            else:
                # Validate tickers and filter out invalid ones
                valid_tickers, validation_results = self._validate_tickers_with_date_range(
                    all_tickers, start_date, end_date
                )
            
            # Fallback: if validation fails completely, provide a basic working set
            if len(valid_tickers) == 0:
                logger.warning(f"Validation failed for all tickers in {index_symbol}, using fallback set")
                fallback_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
                if skip_validation:
                    valid_tickers = fallback_tickers
                else:
                    valid_tickers, validation_results = self._validate_tickers_with_date_range(
                        fallback_tickers[:10], start_date, end_date
                    )
                
                # If even fallback fails, just return the basic tickers without validation
                if len(valid_tickers) == 0:
                    logger.error(f"Even fallback validation failed for {index_symbol}, returning basic tickers")
                    valid_tickers = fallback_tickers[:10]
                    validation_results = {
                        'valid_tickers': valid_tickers,
                        'invalid_tickers': [],
                        'validation_details': {ticker: {'valid': True, 'reason': 'Fallback (not validated)'} for ticker in valid_tickers},
                        'success_rate': 1.0,
                        'fallback_used': True
                    }
            
            metadata = {
                'index_name': index_data['name'],
                'index_description': index_data['description'],
                'total_constituents': len(all_tickers),
                'requested_count': len(all_tickers),
                'valid_count': len(valid_tickers),
                'invalid_count': len(all_tickers) - len(valid_tickers),
                'validation_results': validation_results,
                'last_updated': datetime.now().isoformat(),
                'fallback_used': validation_results.get('fallback_used', False),
                'date_range_used': f"{start_date} to {end_date}" if start_date and end_date else "Not specified"
            }
            
            result = (valid_tickers, metadata)
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), result)
            
            logger.info(f"Successfully processed {index_symbol}: {len(valid_tickers)}/{len(all_tickers)} valid tickers")
            return result
            
        except Exception as e:
            logger.error(f"Error getting constituents for {index_symbol}: {e}")
            raise
    
    def _validate_tickers_with_date_range(self, tickers: List[str], start_date: Optional[datetime] = None, 
                                        end_date: Optional[datetime] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Validate tickers with optional date range filtering.
        
        Args:
            tickers: List of ticker symbols to validate
            start_date: Start date for data availability check
            end_date: End date for data availability check
            
        Returns:
            Tuple of (valid_tickers, validation_results)
        """
        if not tickers:
            return [], {'valid_tickers': [], 'invalid_tickers': [], 'validation_details': {}, 'success_rate': 0.0}
        
        logger.info(f"Validating {len(tickers)} tickers with date range {start_date} to {end_date}")
        
        valid_tickers = []
        invalid_tickers = []
        validation_details = {}
        
        # Process tickers in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"Validating batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
            
            for ticker in batch:
                try:
                    # Get stock data
                    stock = yf.Ticker(ticker)
                    
                    # Determine date range for validation
                    if start_date and end_date:
                        # Use user-specified date range
                        hist = stock.history(start=start_date, end=end_date)
                    else:
                        # Default: check last 30 days
                        hist = stock.history(period='1mo')
                    
                    if hist.empty:
                        validation_details[ticker] = {'valid': False, 'reason': 'No price history available'}
                        invalid_tickers.append(ticker)
                        continue
                    
                    # Check if we have sufficient data
                    if len(hist) < 5:  # Need at least 5 data points
                        validation_details[ticker] = {'valid': False, 'reason': f'Insufficient data ({len(hist)} days)'}
                        invalid_tickers.append(ticker)
                        continue
                    
                    # Check for reasonable price (more permissive criteria)
                    last_price = float(hist['Close'].iloc[-1])
                    if last_price < 0.10:  # Very permissive - only reject extremely low prices
                        validation_details[ticker] = {'valid': False, 'reason': f'Price too low (${last_price:.2f})'}
                        invalid_tickers.append(ticker)
                        continue
                    
                    if last_price > 100000:  # Very high threshold
                        validation_details[ticker] = {'valid': False, 'reason': f'Price too high (${last_price:.2f})'}
                        invalid_tickers.append(ticker)
                        continue
                    
                    # If we get here, the ticker is valid
                    validation_details[ticker] = {'valid': True, 'reason': f'Valid (${last_price:.2f}, {len(hist)} days)'}
                    valid_tickers.append(ticker)
                    
                except Exception as e:
                    validation_details[ticker] = {'valid': False, 'reason': f'Error: {str(e)[:50]}'}
                    invalid_tickers.append(ticker)
            
            # Small delay between batches to be respectful to the API
            if i + batch_size < len(tickers):
                time.sleep(0.1)
        
        success_rate = len(valid_tickers) / len(tickers) if tickers else 0
        
        validation_results = {
            'valid_tickers': valid_tickers,
            'invalid_tickers': invalid_tickers,
            'validation_details': validation_details,
            'success_rate': success_rate
        }
        
        logger.info(f"Validation complete: {len(valid_tickers)}/{len(tickers)} tickers valid ({success_rate:.1%})")
        return valid_tickers, validation_results