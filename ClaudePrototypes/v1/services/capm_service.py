"""
Capital Asset Pricing Model (CAPM) Service
Implements CAPM analysis including beta calculation, expected returns, and alpha.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from scipy import stats
import matplotlib.pyplot as plt
from .data_service import DataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CAPMService:
    """Service for CAPM analysis and related calculations."""
    
    def __init__(self):
        self.data_service = DataService()
    
    def analyze_capm(self, ticker: str, market_ticker: str, start_date: datetime,
                    end_date: datetime, risk_free_rate: float = 0.025) -> Dict[str, Any]:
        """
        Perform comprehensive CAPM analysis for a stock.
        
        Args:
            ticker: Stock ticker symbol
            market_ticker: Market index ticker (e.g., ^GSPC for S&P 500)
            start_date: Start date for analysis
            end_date: End date for analysis
            risk_free_rate: Risk-free rate (as decimal, e.g., 0.025 for 2.5%)
            
        Returns:
            Dictionary containing CAPM analysis results
        """
        try:
            # Get stock and market data
            stock_data = self.data_service.get_stock_data(ticker, start_date, end_date)
            market_data = self.data_service.get_market_data(market_ticker, start_date, end_date)
            
            # Extract returns
            stock_returns = pd.Series(stock_data['log_returns'])
            market_returns = pd.Series(market_data['log_returns'])
            
            # Align the data (in case of different lengths)
            min_length = min(len(stock_returns), len(market_returns))
            stock_returns = stock_returns[:min_length]
            market_returns = market_returns[:min_length]
            
            # Calculate beta
            beta_results = self._calculate_beta(stock_returns, market_returns)
            
            # Calculate expected return using CAPM
            expected_return = risk_free_rate + beta_results['beta'] * (
                market_data['statistics']['avg_annual_return'] - risk_free_rate
            )
            
            # Calculate alpha
            actual_return = stock_data['statistics']['avg_annual_return']
            alpha = actual_return - expected_return
            
            # Calculate Sharpe ratio
            stock_volatility = stock_data['statistics']['annual_volatility']
            sharpe_ratio = (actual_return - risk_free_rate) / stock_volatility if stock_volatility > 0 else 0
            
            # Calculate Treynor ratio
            treynor_ratio = (actual_return - risk_free_rate) / beta_results['beta'] if beta_results['beta'] != 0 else 0
            
            # Get company information
            company_info = self.data_service.get_company_info(ticker)
            
            # Generate insights
            insights = self._generate_capm_insights(
                ticker, beta_results['beta'], alpha, expected_return, 
                actual_return, sharpe_ratio, company_info
            )
            
            return {
                'ticker': ticker,
                'market_ticker': market_ticker,
                'company_info': company_info,
                'analysis_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                },
                'capm_metrics': {
                    'beta': float(beta_results['beta']),
                    'beta_confidence_interval': beta_results['confidence_interval'],
                    'r_squared': float(beta_results['r_squared']),
                    'p_value': float(beta_results['p_value']),
                    'alpha': float(alpha),
                    'expected_return': float(expected_return),
                    'actual_return': float(actual_return),
                    'risk_free_rate': float(risk_free_rate),
                    'market_return': float(market_data['statistics']['avg_annual_return']),
                    'equity_risk_premium': float(market_data['statistics']['avg_annual_return'] - risk_free_rate)
                },
                'performance_metrics': {
                    'sharpe_ratio': float(sharpe_ratio),
                    'treynor_ratio': float(treynor_ratio),
                    'volatility': float(stock_volatility),
                    'market_volatility': float(market_data['statistics']['annual_volatility'])
                },
                'regression_data': {
                    'stock_returns': stock_returns.tolist(),
                    'market_returns': market_returns.tolist(),
                    'regression_line': beta_results['regression_line']
                },
                'insights': insights,
                'chart_data': self._create_capm_chart_data(stock_returns, market_returns, beta_results)
            }
            
        except Exception as e:
            logger.error(f"Error in CAPM analysis: {str(e)}")
            raise e
    
    def _calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> Dict[str, Any]:
        """Calculate beta and related regression statistics."""
        try:
            # Remove any NaN values
            combined_data = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()
            stock_clean = combined_data['stock']
            market_clean = combined_data['market']
            
            # Calculate covariance and variance
            covariance = np.cov(stock_clean, market_clean)[0, 1] * 252  # Annualized
            market_variance = np.var(market_clean) * 252  # Annualized
            
            # Calculate beta
            beta = covariance / market_variance if market_variance != 0 else 0
            
            # Perform linear regression for additional statistics
            slope, intercept, r_value, p_value, std_err = stats.linregress(market_clean, stock_clean)
            
            # Calculate confidence interval for beta (95% confidence)
            confidence_interval = {
                'lower': float(slope - 1.96 * std_err),
                'upper': float(slope + 1.96 * std_err)
            }
            
            # Generate regression line data
            market_range = np.linspace(market_clean.min(), market_clean.max(), 100)
            regression_line = {
                'x': market_range.tolist(),
                'y': (slope * market_range + intercept).tolist()
            }
            
            return {
                'beta': float(beta),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'std_error': float(std_err),
                'confidence_interval': confidence_interval,
                'regression_line': regression_line,
                'intercept': float(intercept)
            }
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return {
                'beta': 0.0,
                'r_squared': 0.0,
                'p_value': 1.0,
                'std_error': 0.0,
                'confidence_interval': {'lower': 0.0, 'upper': 0.0},
                'regression_line': {'x': [], 'y': []},
                'intercept': 0.0
            }
    
    def _generate_capm_insights(self, ticker: str, beta: float, alpha: float,
                              expected_return: float, actual_return: float,
                              sharpe_ratio: float, company_info: Dict[str, Any]) -> list[str]:
        """Generate insights about the CAPM analysis."""
        insights = []
        
        try:
            # Beta interpretation
            if beta < 0.5:
                insights.append(f"{ticker} is a defensive stock (β={beta:.3f}), moving less than the market.")
            elif beta < 1.0:
                insights.append(f"{ticker} is moderately defensive (β={beta:.3f}), less volatile than the market.")
            elif beta < 1.5:
                insights.append(f"{ticker} is moderately aggressive (β={beta:.3f}), more volatile than the market.")
            else:
                insights.append(f"{ticker} is highly aggressive (β={beta:.3f}), significantly more volatile than the market.")
            
            # Alpha interpretation
            if alpha > 0.02:  # 2% positive alpha
                insights.append(f"Strong positive alpha of {alpha:.1%} suggests excellent risk-adjusted performance.")
            elif alpha > 0.005:  # 0.5% positive alpha
                insights.append(f"Positive alpha of {alpha:.1%} indicates outperformance relative to systematic risk.")
            elif alpha < -0.02:  # -2% negative alpha
                insights.append(f"Negative alpha of {alpha:.1%} suggests poor risk-adjusted performance.")
            else:
                insights.append(f"Alpha near zero ({alpha:.1%}) indicates performance in line with market expectations.")
            
            # Expected vs actual return
            return_diff = actual_return - expected_return
            if abs(return_diff) > 0.02:  # 2% difference
                direction = "outperformed" if return_diff > 0 else "underperformed"
                insights.append(f"{ticker} {direction} CAPM expectations by {abs(return_diff):.1%}.")
            
            # Sharpe ratio interpretation
            if sharpe_ratio > 1.0:
                insights.append(f"Excellent risk-adjusted returns with Sharpe ratio of {sharpe_ratio:.2f}.")
            elif sharpe_ratio > 0.5:
                insights.append(f"Good risk-adjusted returns with Sharpe ratio of {sharpe_ratio:.2f}.")
            elif sharpe_ratio < 0:
                insights.append(f"Poor risk-adjusted returns with negative Sharpe ratio of {sharpe_ratio:.2f}.")
            
            # Sector context
            if company_info['sector'] != 'N/A':
                insights.append(f"As a {company_info['sector']} company, consider sector-specific risk factors.")
            
            # Beta stability suggestion
            insights.append("Beta is calculated from historical data and may not predict future market sensitivity.")
            
        except Exception as e:
            logger.error(f"Error generating CAPM insights: {str(e)}")
            insights.append("Unable to generate detailed insights due to calculation error.")
        
        return insights
    
    def _create_capm_chart_data(self, stock_returns: pd.Series, market_returns: pd.Series,
                              beta_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create chart data for CAPM visualization."""
        try:
            # Prepare scatter plot data
            combined_data = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()
            
            chart_data = {
                'scatter_plot': {
                    'x': combined_data['market'].tolist(),
                    'y': combined_data['stock'].tolist(),
                    'name': 'Returns Relationship'
                },
                'regression_line': beta_results['regression_line'],
                'beta_value': beta_results['beta'],
                'r_squared': beta_results['r_squared']
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating CAPM chart data: {str(e)}")
            return {}
    
    def calculate_security_market_line(self, beta_range: tuple = (0, 2), 
                                     risk_free_rate: float = 0.025,
                                     market_return: float = 0.10) -> Dict[str, Any]:
        """
        Calculate the Security Market Line (SML) for CAPM visualization.
        
        Args:
            beta_range: Range of beta values for the line
            risk_free_rate: Risk-free rate
            market_return: Expected market return
            
        Returns:
            Dictionary containing SML data
        """
        try:
            beta_values = np.linspace(beta_range[0], beta_range[1], 100)
            expected_returns = risk_free_rate + beta_values * (market_return - risk_free_rate)
            
            return {
                'beta_values': beta_values.tolist(),
                'expected_returns': expected_returns.tolist(),
                'risk_free_rate': risk_free_rate,
                'market_return': market_return,
                'market_beta': 1.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating Security Market Line: {str(e)}")
            return {}
    
    def compare_stocks_capm(self, tickers: list[str], market_ticker: str,
                          start_date: datetime, end_date: datetime,
                          risk_free_rate: float = 0.025) -> Dict[str, Any]:
        """
        Compare multiple stocks using CAPM analysis.
        
        Args:
            tickers: List of stock ticker symbols
            market_ticker: Market index ticker
            start_date: Start date for analysis
            end_date: End date for analysis
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            comparison_results = {}
            
            for ticker in tickers:
                try:
                    capm_result = self.analyze_capm(ticker, market_ticker, start_date, end_date, risk_free_rate)
                    comparison_results[ticker] = {
                        'beta': capm_result['capm_metrics']['beta'],
                        'alpha': capm_result['capm_metrics']['alpha'],
                        'expected_return': capm_result['capm_metrics']['expected_return'],
                        'actual_return': capm_result['capm_metrics']['actual_return'],
                        'sharpe_ratio': capm_result['performance_metrics']['sharpe_ratio'],
                        'r_squared': capm_result['capm_metrics']['r_squared']
                    }
                except Exception as e:
                    logger.warning(f"Failed to analyze {ticker}: {str(e)}")
                    continue
            
            # Generate comparison insights
            if comparison_results:
                best_alpha = max(comparison_results.items(), key=lambda x: x[1]['alpha'])
                best_sharpe = max(comparison_results.items(), key=lambda x: x[1]['sharpe_ratio'])
                lowest_beta = min(comparison_results.items(), key=lambda x: x[1]['beta'])
                
                insights = [
                    f"Best alpha: {best_alpha[0]} ({best_alpha[1]['alpha']:.2%})",
                    f"Best Sharpe ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})",
                    f"Most defensive (lowest beta): {lowest_beta[0]} ({lowest_beta[1]['beta']:.3f})"
                ]
            else:
                insights = ["No successful analyses to compare"]
            
            return {
                'comparison_data': comparison_results,
                'insights': insights,
                'analysis_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in CAPM comparison: {str(e)}")
            raise e
