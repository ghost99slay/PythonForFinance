"""
Portfolio Theory Service
Implements Markowitz Portfolio Theory and Efficient Frontier calculations.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import io
import base64
from .data_service import DataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioService:
    """Service for portfolio analysis using Markowitz Portfolio Theory."""
    
    def __init__(self):
        self.data_service = DataService()
    
    def analyze_portfolio(self, tickers: List[str], start_date: datetime, 
                         end_date: datetime) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis including efficient frontier.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary containing portfolio analysis results
        """
        try:
            # Get data for all stocks
            portfolio_data = self.data_service.get_multiple_stocks_data(
                tickers, start_date, end_date
            )
            
            # Use only the tickers that actually have data
            valid_tickers = portfolio_data['tickers']  # This contains only successful tickers
            logger.info(f"Portfolio analysis proceeding with {len(valid_tickers)}/{len(tickers)} tickers with valid data")
            
            # Extract returns data
            returns_df = pd.DataFrame(portfolio_data['returns'])
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Generate efficient frontier
            efficient_portfolios = self._generate_efficient_frontier(expected_returns, cov_matrix)
            
            # Find optimal portfolios
            min_vol_portfolio = self._find_minimum_volatility_portfolio(expected_returns, cov_matrix)
            max_sharpe_portfolio = self._find_maximum_sharpe_portfolio(expected_returns, cov_matrix)
            
            # Calculate equal weight portfolio for comparison (using valid tickers)
            equal_weights = np.array([1/len(valid_tickers)] * len(valid_tickers))
            equal_weight_return = np.sum(equal_weights * expected_returns)
            equal_weight_vol = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            
            # Generate portfolio visualization
            chart_data = self._create_portfolio_chart(
                efficient_portfolios, min_vol_portfolio, max_sharpe_portfolio,
                equal_weight_return, equal_weight_vol, valid_tickers
            )
            
            return {
                'tickers': valid_tickers,  # Return only valid tickers
                'original_tickers_requested': len(tickers),
                'valid_tickers_analyzed': len(valid_tickers),
                'analysis_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                },
                'expected_returns': expected_returns.to_dict(),
                'correlation_matrix': portfolio_data['correlation_matrix'],
                'covariance_matrix': cov_matrix.to_dict(),
                'efficient_frontier': efficient_portfolios,
                'optimal_portfolios': {
                    'minimum_volatility': min_vol_portfolio,
                    'maximum_sharpe': max_sharpe_portfolio,
                    'equal_weight': {
                        'weights': dict(zip(valid_tickers, equal_weights)),
                        'expected_return': float(equal_weight_return),
                        'volatility': float(equal_weight_vol),
                        'sharpe_ratio': float((equal_weight_return - 0.025) / equal_weight_vol)
                    }
                },
                'individual_statistics': portfolio_data['individual_statistics'],
                'chart_data': chart_data,
                'insights': self._generate_portfolio_insights(
                    valid_tickers, expected_returns, cov_matrix, 
                    min_vol_portfolio, max_sharpe_portfolio
                )
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {str(e)}")
            raise e
    
    def _generate_efficient_frontier(self, expected_returns: pd.Series, 
                                   cov_matrix: pd.DataFrame, num_portfolios: int = None) -> Dict[str, Any]:
        """Generate the efficient frontier."""
        try:
            num_assets = len(expected_returns)
            
            # Adjust number of portfolios based on portfolio size for performance
            if num_portfolios is None:
                if num_assets > 200:
                    num_portfolios = 20  # Fewer points for very large portfolios
                elif num_assets > 100:
                    num_portfolios = 50  # Moderate points for large portfolios
                else:
                    num_portfolios = 100  # Full resolution for smaller portfolios
            
            logger.info(f"Generating efficient frontier with {num_portfolios} points for {num_assets} assets")
            
            # Define the range of target returns
            min_ret = expected_returns.min()
            max_ret = expected_returns.max()
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            efficient_portfolios = {
                'returns': [],
                'volatilities': [],
                'weights': []
            }
            
            for target_return in target_returns:
                # Constraints
                constraints = (
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                    {'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return}  # target return
                )
                
                # Bounds (weights between 0 and 1)
                bounds = tuple((0, 1) for _ in range(num_assets))
                
                # Initial guess (equal weights)
                initial_guess = np.array([1/num_assets] * num_assets)
                
                # Objective function (minimize portfolio variance)
                def objective(weights):
                    return np.dot(weights.T, np.dot(cov_matrix, weights))
                
                # Optimize with settings optimized for large portfolios
                options = {'ftol': 1e-6, 'maxiter': 200} if num_assets > 100 else {}
                result = minimize(objective, initial_guess, method='SLSQP', 
                                bounds=bounds, constraints=constraints, options=options)
                
                if result.success:
                    weights = result.x
                    portfolio_return = np.sum(weights * expected_returns)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    efficient_portfolios['returns'].append(float(portfolio_return))
                    efficient_portfolios['volatilities'].append(float(portfolio_vol))
                    efficient_portfolios['weights'].append(weights.tolist())
            
            return efficient_portfolios
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            return {'returns': [], 'volatilities': [], 'weights': []}
    
    def _find_minimum_volatility_portfolio(self, expected_returns: pd.Series, 
                                         cov_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Find the minimum volatility portfolio."""
        try:
            num_assets = len(expected_returns)
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Initial guess
            initial_guess = np.array([1/num_assets] * num_assets)
            
            # Objective function (minimize portfolio variance)
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Optimize with settings optimized for large portfolios
            options = {'ftol': 1e-6, 'maxiter': 200} if num_assets > 100 else {}
            result = minimize(objective, initial_guess, method='SLSQP', 
                            bounds=bounds, constraints=constraints, options=options)
            
            if result.success:
                weights = result.x
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - 0.025) / portfolio_vol  # Assuming 2.5% risk-free rate
                
                return {
                    'weights': dict(zip(expected_returns.index, weights)),
                    'expected_return': float(portfolio_return),
                    'volatility': float(portfolio_vol),
                    'sharpe_ratio': float(sharpe_ratio)
                }
            else:
                raise ValueError("Optimization failed for minimum volatility portfolio")
                
        except Exception as e:
            logger.error(f"Error finding minimum volatility portfolio: {str(e)}")
            raise e
    
    def _find_maximum_sharpe_portfolio(self, expected_returns: pd.Series, 
                                     cov_matrix: pd.DataFrame, risk_free_rate: float = 0.025) -> Dict[str, Any]:
        """Find the maximum Sharpe ratio portfolio."""
        try:
            num_assets = len(expected_returns)
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Initial guess
            initial_guess = np.array([1/num_assets] * num_assets)
            
            # Objective function (maximize Sharpe ratio = minimize negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                if portfolio_vol == 0:
                    return -np.inf
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            # Optimize with settings optimized for large portfolios
            options = {'ftol': 1e-6, 'maxiter': 200} if num_assets > 100 else {}
            result = minimize(objective, initial_guess, method='SLSQP', 
                            bounds=bounds, constraints=constraints, options=options)
            
            if result.success:
                weights = result.x
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
                
                return {
                    'weights': dict(zip(expected_returns.index, weights)),
                    'expected_return': float(portfolio_return),
                    'volatility': float(portfolio_vol),
                    'sharpe_ratio': float(sharpe_ratio)
                }
            else:
                raise ValueError("Optimization failed for maximum Sharpe ratio portfolio")
                
        except Exception as e:
            logger.error(f"Error finding maximum Sharpe portfolio: {str(e)}")
            raise e
    
    def _create_portfolio_chart(self, efficient_portfolios: Dict[str, Any], 
                              min_vol_portfolio: Dict[str, Any], 
                              max_sharpe_portfolio: Dict[str, Any],
                              equal_weight_return: float, equal_weight_vol: float,
                              tickers: List[str]) -> Dict[str, Any]:
        """Create portfolio visualization chart data."""
        try:
            # Prepare data for frontend charting
            chart_data = {
                'efficient_frontier': {
                    'x': efficient_portfolios['volatilities'],
                    'y': efficient_portfolios['returns'],
                    'name': 'Efficient Frontier'
                },
                'min_volatility': {
                    'x': [min_vol_portfolio['volatility']],
                    'y': [min_vol_portfolio['expected_return']],
                    'name': 'Minimum Volatility Portfolio'
                },
                'max_sharpe': {
                    'x': [max_sharpe_portfolio['volatility']],
                    'y': [max_sharpe_portfolio['expected_return']],
                    'name': 'Maximum Sharpe Ratio Portfolio'
                },
                'equal_weight': {
                    'x': [equal_weight_vol],
                    'y': [equal_weight_return],
                    'name': 'Equal Weight Portfolio'
                },
                'individual_stocks': {
                    'tickers': tickers,
                    'x': [],
                    'y': []
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating portfolio chart: {str(e)}")
            return {}
    
    def _generate_portfolio_insights(self, tickers: List[str], expected_returns: pd.Series,
                                   cov_matrix: pd.DataFrame, min_vol_portfolio: Dict[str, Any],
                                   max_sharpe_portfolio: Dict[str, Any]) -> List[str]:
        """Generate insights about the portfolio analysis."""
        insights = []
        
        try:
            # Best performing stock
            best_stock = expected_returns.idxmax()
            best_return = expected_returns.max()
            insights.append(f"Best performing stock: {best_stock} with {best_return:.2%} expected annual return")
            
            # Most volatile stock
            volatilities = np.sqrt(np.diag(cov_matrix))
            most_volatile_idx = np.argmax(volatilities)
            most_volatile_stock = tickers[most_volatile_idx]
            insights.append(f"Most volatile stock: {most_volatile_stock} with {volatilities[most_volatile_idx]:.2%} annual volatility")
            
            # Correlation insights
            corr_matrix = cov_matrix.corr() if hasattr(cov_matrix, 'corr') else pd.DataFrame(cov_matrix).corr()
            
            # Find highest correlation pair
            corr_values = []
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    corr_values.append((tickers[i], tickers[j], corr_matrix.iloc[i, j]))
            
            if corr_values:
                highest_corr = max(corr_values, key=lambda x: x[2])
                insights.append(f"Highest correlation: {highest_corr[0]} and {highest_corr[1]} ({highest_corr[2]:.3f})")
            
            # Portfolio insights
            min_vol_top_weight = max(min_vol_portfolio['weights'].items(), key=lambda x: x[1])
            insights.append(f"Minimum volatility portfolio is dominated by {min_vol_top_weight[0]} ({min_vol_top_weight[1]:.1%})")
            
            max_sharpe_top_weight = max(max_sharpe_portfolio['weights'].items(), key=lambda x: x[1])
            insights.append(f"Maximum Sharpe ratio portfolio is dominated by {max_sharpe_top_weight[0]} ({max_sharpe_top_weight[1]:.1%})")
            
            # Diversification benefit
            equal_weight_vol = np.sqrt(np.sum(np.diag(cov_matrix)) / len(tickers)**2)
            actual_min_vol = min_vol_portfolio['volatility']
            diversification_benefit = (equal_weight_vol - actual_min_vol) / equal_weight_vol
            insights.append(f"Diversification reduces risk by {diversification_benefit:.1%} compared to naive equal weighting")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("Unable to generate detailed insights due to calculation error")
        
        return insights
    
    def calculate_portfolio_performance(self, weights: Dict[str, float], tickers: List[str],
                                      start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate performance metrics for a given portfolio."""
        try:
            # Get portfolio data
            portfolio_data = self.data_service.get_multiple_stocks_data(tickers, start_date, end_date)
            returns_df = pd.DataFrame(portfolio_data['returns'])
            
            # Calculate portfolio returns
            weight_array = np.array([weights.get(ticker, 0) for ticker in tickers])
            portfolio_returns = returns_df.dot(weight_array)
            
            # Calculate performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.025) / annual_vol if annual_vol > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'annual_volatility': float(annual_vol),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'portfolio_returns': portfolio_returns.tolist(),
                'cumulative_returns': cumulative_returns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {str(e)}")
            raise e
