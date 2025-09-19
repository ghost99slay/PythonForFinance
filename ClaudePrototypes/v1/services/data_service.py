"""
Financial Data Service
Handles data retrieval from various financial data sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching and processing financial data."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_stock_data(self, ticker: str, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None, period: str = "5y") -> Dict[str, Any]:
        """
        Fetch stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            period: Period for data if dates not specified
            
        Returns:
            Dictionary containing stock data and basic statistics
        """
        try:
            # Use cache key based on parameters
            cache_key = f"{ticker}_{start_date}_{end_date}_{period}"
            
            if cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_timeout:
                    return data
            
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                hist_data = stock.history(start=start_date, end=end_date, auto_adjust=False)
            else:
                hist_data = stock.history(period=period, auto_adjust=False)
            
            if hist_data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Calculate returns
            hist_data['Simple_Return'] = hist_data['Adj Close'].pct_change()
            hist_data['Log_Return'] = np.log(hist_data['Adj Close'] / hist_data['Adj Close'].shift(1))
            
            # Prepare response data
            result = {
                'ticker': ticker,
                'data': hist_data.to_dict('records'),
                'dates': [d.strftime('%Y-%m-%d') for d in hist_data.index],
                'prices': hist_data['Adj Close'].tolist(),
                'returns': hist_data['Simple_Return'].dropna().tolist(),
                'log_returns': hist_data['Log_Return'].dropna().tolist(),
                'statistics': {
                    'current_price': float(hist_data['Adj Close'].iloc[-1]),
                    'avg_daily_return': float(hist_data['Simple_Return'].mean()),
                    'avg_annual_return': float(hist_data['Simple_Return'].mean() * 252),
                    'daily_volatility': float(hist_data['Simple_Return'].std()),
                    'annual_volatility': float(hist_data['Simple_Return'].std() * np.sqrt(252)),
                    'max_price': float(hist_data['Adj Close'].max()),
                    'min_price': float(hist_data['Adj Close'].min()),
                    'total_return': float((hist_data['Adj Close'].iloc[-1] / hist_data['Adj Close'].iloc[0]) - 1)
                }
            }
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise e
    
    def get_multiple_stocks_data(self, tickers: List[str], start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None, period: str = "5y") -> Dict[str, Any]:
        """
        Fetch data for multiple stocks and return combined DataFrame.
        Uses batch downloading for improved performance with large portfolios.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            period: Period for data if dates not specified
            
        Returns:
            Dictionary containing combined data and individual stock statistics
        """
        try:
            logger.info(f"Fetching data for {len(tickers)} tickers using batch download")
            
            # Use yfinance batch download for much faster performance
            if start_date and end_date:
                combined_data = yf.download(tickers, start=start_date, end=end_date, 
                                          group_by='ticker', progress=False, threads=True)
            else:
                combined_data = yf.download(tickers, period=period, 
                                          group_by='ticker', progress=False, threads=True)
            
            # Handle single ticker case (yfinance returns different structure)
            if len(tickers) == 1:
                ticker = tickers[0]
                combined_data.columns = pd.MultiIndex.from_product([[ticker], combined_data.columns])
            
            # Extract closing prices for all tickers using efficient concatenation
            price_series = []
            individual_stats = {}
            
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        ticker_data = combined_data[ticker]
                    else:
                        ticker_data = combined_data[ticker] if ticker in combined_data.columns.get_level_values(0) else None
                    
                    if ticker_data is not None and not ticker_data.empty:
                        # Get closing prices
                        close_prices = ticker_data['Close'].dropna()
                        if not close_prices.empty:
                            close_prices.name = ticker
                            price_series.append(close_prices)
                            
                            # Calculate individual statistics
                            returns = close_prices.pct_change().dropna()
                            individual_stats[ticker] = {
                                'current_price': float(close_prices.iloc[-1]),
                                'daily_return': float(returns.iloc[-1]) if len(returns) > 0 else 0.0,
                                'volatility': float(returns.std() * np.sqrt(252)),
                                'avg_return': float(returns.mean() * 252),
                                'total_return': float((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) if len(close_prices) > 1 else 0.0,
                                'data_points': len(close_prices)
                            }
                        else:
                            logger.warning(f"No valid data for {ticker}")
                    else:
                        logger.warning(f"No data available for {ticker}")
                        
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                    continue
            
            # Create DataFrame efficiently using concat
            if not price_series:
                raise ValueError("No valid data found for any tickers")
            
            price_data = pd.concat(price_series, axis=1)
            
            # Calculate returns for the combined dataset using price_data
            if price_data.empty:
                raise ValueError("No valid data found for any tickers")
            
            returns_data = price_data.pct_change().dropna()
            log_returns_data = np.log(price_data / price_data.shift(1)).dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_data.corr()
            covariance_matrix = returns_data.cov() * 252  # Annualized
            
            logger.info(f"Successfully processed {len(price_data.columns)}/{len(tickers)} tickers")
            
            return {
                'tickers': list(price_data.columns),  # Only successful tickers
                'prices': price_data.to_dict('series'),
                'returns': returns_data.to_dict('series'),
                'log_returns': log_returns_data.to_dict('series'),
                'dates': [d.strftime('%Y-%m-%d') for d in price_data.index],
                'correlation_matrix': correlation_matrix.to_dict(),
                'covariance_matrix': covariance_matrix.to_dict(),
                'individual_statistics': individual_stats,
                'combined_statistics': {
                    'avg_daily_returns': (returns_data.mean() * 252).to_dict(),
                    'annual_volatilities': (returns_data.std() * np.sqrt(252)).to_dict(),
                    'total_returns': ((price_data.iloc[-1] / price_data.iloc[0]) - 1).to_dict()
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching multiple stocks data: {str(e)}")
            raise e
    
    def get_market_data(self, market_ticker: str = "^GSPC", start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None, period: str = "5y") -> Dict[str, Any]:
        """
        Fetch market index data (default S&P 500).
        
        Args:
            market_ticker: Market index ticker (default S&P 500)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            period: Period for data if dates not specified
            
        Returns:
            Dictionary containing market data
        """
        return self.get_stock_data(market_ticker, start_date, end_date, period)
    
    def get_risk_free_rate(self, rate: Optional[float] = None) -> float:
        """
        Get risk-free rate. If not provided, attempts to fetch 10-year Treasury rate.
        
        Args:
            rate: Manual risk-free rate (as decimal, e.g., 0.025 for 2.5%)
            
        Returns:
            Risk-free rate as decimal
        """
        if rate is not None:
            return rate
        
        try:
            # Try to fetch 10-year Treasury rate
            treasury = yf.Ticker("^TNX")
            treasury_data = treasury.history(period="1d")
            if not treasury_data.empty:
                return float(treasury_data['Close'].iloc[-1] / 100)  # Convert percentage to decimal
        except Exception as e:
            logger.warning(f"Could not fetch Treasury rate: {str(e)}")
        
        # Default to 2.5% if unable to fetch
        return 0.025
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol exists and has data.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Boolean indicating if ticker is valid
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            return not hist.empty
        except:
            return False
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get company information for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', None),
                'pe_ratio': info.get('forwardPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'description': info.get('longBusinessSummary', 'N/A')[:500] + '...' if info.get('longBusinessSummary') else 'N/A'
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            return {
                'symbol': ticker,
                'name': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'beta': None,
                'pe_ratio': None,
                'dividend_yield': None,
                'description': 'N/A'
            }
