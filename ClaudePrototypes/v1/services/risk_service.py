"""
Risk Analysis Service
Implements investment risk measurement tools including variance, correlation, and diversification analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging
from .data_service import DataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskService:
    """Service for comprehensive risk analysis of investments and portfolios."""
    
    def __init__(self):
        self.data_service = DataService()
    
    def analyze_risk(self, tickers: List[str], start_date: datetime, 
                    end_date: datetime) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis for a portfolio of stocks.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary containing risk analysis results
        """
        try:
            # Get data for all stocks
            portfolio_data = self.data_service.get_multiple_stocks_data(
                tickers, start_date, end_date
            )
            
            # Extract returns data
            returns_df = pd.DataFrame(portfolio_data['returns'])
            
            # Calculate risk metrics for individual stocks
            individual_risks = self._calculate_individual_risks(returns_df)
            
            # Calculate correlation and covariance analysis
            correlation_analysis = self._analyze_correlations(returns_df)
            
            # Calculate diversification benefits
            diversification_analysis = self._analyze_diversification(returns_df)
            
            # Calculate Value at Risk (VaR) and Conditional VaR
            var_analysis = self._calculate_var(returns_df)
            
            # Generate risk insights
            insights = self._generate_risk_insights(
                tickers, individual_risks, correlation_analysis, diversification_analysis
            )
            
            return {
                'tickers': tickers,
                'analysis_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                },
                'individual_risks': individual_risks,
                'correlation_analysis': correlation_analysis,
                'diversification_analysis': diversification_analysis,
                'var_analysis': var_analysis,
                'portfolio_statistics': portfolio_data['combined_statistics'],
                'insights': insights,
                'chart_data': self._create_risk_chart_data(returns_df, correlation_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            raise e
    
    def _calculate_individual_risks(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for individual stocks."""
        try:
            individual_risks = {}
            
            for ticker in returns_df.columns:
                returns = returns_df[ticker].dropna()
                
                # Basic risk metrics
                daily_vol = returns.std()
                annual_vol = daily_vol * np.sqrt(252)
                
                # Skewness and kurtosis
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                
                # Downside deviation (semi-deviation)
                negative_returns = returns[returns < 0]
                downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
                
                # Maximum drawdown
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Percentiles
                percentiles = {
                    '1%': float(returns.quantile(0.01)),
                    '5%': float(returns.quantile(0.05)),
                    '95%': float(returns.quantile(0.95)),
                    '99%': float(returns.quantile(0.99))
                }
                
                individual_risks[ticker] = {
                    'daily_volatility': float(daily_vol),
                    'annual_volatility': float(annual_vol),
                    'downside_deviation': float(downside_deviation),
                    'max_drawdown': float(max_drawdown),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'percentiles': percentiles
                }
            
            return individual_risks
            
        except Exception as e:
            logger.error(f"Error calculating individual risks: {str(e)}")
            return {}
    
    def _analyze_correlations(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between stocks."""
        try:
            # Correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Covariance matrix (annualized)
            covariance_matrix = returns_df.cov() * 252
            
            # Find highest and lowest correlations
            correlations = []
            tickers = returns_df.columns.tolist()
            
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    corr_value = correlation_matrix.iloc[i, j]
                    correlations.append({
                        'pair': f"{tickers[i]} - {tickers[j]}",
                        'correlation': float(corr_value)
                    })
            
            # Sort by correlation
            correlations.sort(key=lambda x: x['correlation'], reverse=True)
            
            # Calculate average correlation
            avg_correlation = np.mean([c['correlation'] for c in correlations])
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'covariance_matrix': covariance_matrix.to_dict(),
                'pairwise_correlations': correlations,
                'highest_correlation': correlations[0] if correlations else None,
                'lowest_correlation': correlations[-1] if correlations else None,
                'average_correlation': float(avg_correlation)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {}
    
    def _analyze_diversification(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze diversification benefits."""
        try:
            num_assets = len(returns_df.columns)
            
            # Equal weight portfolio
            equal_weights = np.array([1/num_assets] * num_assets)
            portfolio_returns = returns_df.dot(equal_weights)
            
            # Portfolio risk metrics
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Individual stock volatilities
            individual_vols = returns_df.std() * np.sqrt(252)
            avg_individual_vol = individual_vols.mean()
            
            # Diversification ratio
            diversification_ratio = avg_individual_vol / portfolio_vol if portfolio_vol > 0 else 1
            
            # Risk reduction from diversification
            naive_portfolio_vol = np.sqrt(np.sum(individual_vols**2) / num_assets**2)
            risk_reduction = (naive_portfolio_vol - portfolio_vol) / naive_portfolio_vol if naive_portfolio_vol > 0 else 0
            
            # Systematic vs idiosyncratic risk decomposition
            covariance_matrix = returns_df.cov() * 252
            
            # Portfolio variance
            portfolio_variance = np.dot(equal_weights.T, np.dot(covariance_matrix, equal_weights))
            
            # Weighted average of individual variances (idiosyncratic risk)
            individual_variances = np.diag(covariance_matrix)
            weighted_avg_variance = np.sum(equal_weights**2 * individual_variances)
            
            # Diversifiable risk (covariance effect)
            diversifiable_risk = portfolio_variance - weighted_avg_variance
            
            # Non-diversifiable risk (systematic risk)
            non_diversifiable_risk = weighted_avg_variance
            
            return {
                'portfolio_volatility': float(portfolio_vol),
                'average_individual_volatility': float(avg_individual_vol),
                'diversification_ratio': float(diversification_ratio),
                'risk_reduction_percentage': float(risk_reduction * 100),
                'diversifiable_risk': float(diversifiable_risk),
                'non_diversifiable_risk': float(non_diversifiable_risk),
                'systematic_risk_percentage': float((non_diversifiable_risk / portfolio_variance) * 100) if portfolio_variance > 0 else 0,
                'idiosyncratic_risk_percentage': float((diversifiable_risk / portfolio_variance) * 100) if portfolio_variance > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing diversification: {str(e)}")
            return {}
    
    def _calculate_var(self, returns_df: pd.DataFrame, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) and Conditional VaR."""
        try:
            var_results = {}
            
            # Equal weight portfolio returns
            num_assets = len(returns_df.columns)
            equal_weights = np.array([1/num_assets] * num_assets)
            portfolio_returns = returns_df.dot(equal_weights)
            
            for confidence_level in confidence_levels:
                # Historical VaR
                var_historical = portfolio_returns.quantile(1 - confidence_level)
                
                # Parametric VaR (assuming normal distribution)
                mean_return = portfolio_returns.mean()
                std_return = portfolio_returns.std()
                from scipy.stats import norm
                var_parametric = mean_return + norm.ppf(1 - confidence_level) * std_return
                
                # Conditional VaR (Expected Shortfall)
                tail_returns = portfolio_returns[portfolio_returns <= var_historical]
                cvar = tail_returns.mean() if len(tail_returns) > 0 else var_historical
                
                var_results[f'{int(confidence_level*100)}%'] = {
                    'var_historical': float(var_historical),
                    'var_parametric': float(var_parametric),
                    'conditional_var': float(cvar),
                    'var_historical_annual': float(var_historical * np.sqrt(252)),
                    'var_parametric_annual': float(var_parametric * np.sqrt(252)),
                    'conditional_var_annual': float(cvar * np.sqrt(252))
                }
            
            return var_results
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return {}
    
    def _generate_risk_insights(self, tickers: List[str], individual_risks: Dict[str, Any],
                              correlation_analysis: Dict[str, Any], 
                              diversification_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights about the risk analysis."""
        insights = []
        
        try:
            # Most and least risky stocks
            if individual_risks:
                volatilities = {ticker: data['annual_volatility'] for ticker, data in individual_risks.items()}
                most_risky = max(volatilities, key=volatilities.get)
                least_risky = min(volatilities, key=volatilities.get)
                
                insights.append(f"Most risky stock: {most_risky} ({volatilities[most_risky]:.1%} annual volatility)")
                insights.append(f"Least risky stock: {least_risky} ({volatilities[least_risky]:.1%} annual volatility)")
            
            # Correlation insights
            if correlation_analysis.get('highest_correlation'):
                highest_corr = correlation_analysis['highest_correlation']
                insights.append(f"Highest correlation: {highest_corr['pair']} ({highest_corr['correlation']:.3f})")
            
            if correlation_analysis.get('average_correlation'):
                avg_corr = correlation_analysis['average_correlation']
                if avg_corr > 0.7:
                    insights.append(f"High average correlation ({avg_corr:.3f}) limits diversification benefits")
                elif avg_corr < 0.3:
                    insights.append(f"Low average correlation ({avg_corr:.3f}) provides good diversification")
                else:
                    insights.append(f"Moderate average correlation ({avg_corr:.3f}) provides some diversification")
            
            # Diversification insights
            if diversification_analysis.get('risk_reduction_percentage'):
                risk_reduction = diversification_analysis['risk_reduction_percentage']
                insights.append(f"Diversification reduces portfolio risk by {risk_reduction:.1f}%")
            
            if diversification_analysis.get('diversification_ratio'):
                div_ratio = diversification_analysis['diversification_ratio']
                if div_ratio > 1.5:
                    insights.append(f"Excellent diversification ratio of {div_ratio:.2f}")
                elif div_ratio > 1.2:
                    insights.append(f"Good diversification ratio of {div_ratio:.2f}")
                else:
                    insights.append(f"Limited diversification ratio of {div_ratio:.2f}")
            
            # Systematic vs idiosyncratic risk
            if diversification_analysis.get('systematic_risk_percentage'):
                sys_risk = diversification_analysis['systematic_risk_percentage']
                insights.append(f"Systematic (non-diversifiable) risk accounts for {sys_risk:.1f}% of total portfolio risk")
            
            # Risk distribution insights
            if len(tickers) < 5:
                insights.append("Consider adding more assets to improve diversification")
            elif len(tickers) > 20:
                insights.append("Large number of holdings may lead to over-diversification")
            
        except Exception as e:
            logger.error(f"Error generating risk insights: {str(e)}")
            insights.append("Unable to generate detailed insights due to calculation error")
        
        return insights
    
    def _create_risk_chart_data(self, returns_df: pd.DataFrame, 
                              correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create chart data for risk visualization."""
        try:
            # Volatility comparison
            volatilities = (returns_df.std() * np.sqrt(252)).to_dict()
            
            # Returns distribution data
            returns_distributions = {}
            for ticker in returns_df.columns:
                returns_distributions[ticker] = returns_df[ticker].tolist()
            
            chart_data = {
                'volatility_comparison': {
                    'labels': list(volatilities.keys()),
                    'values': list(volatilities.values())
                },
                'correlation_heatmap': correlation_analysis.get('correlation_matrix', {}),
                'returns_distributions': returns_distributions,
                'risk_return_scatter': {
                    'tickers': returns_df.columns.tolist(),
                    'returns': (returns_df.mean() * 252).tolist(),
                    'volatilities': (returns_df.std() * np.sqrt(252)).tolist()
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating risk chart data: {str(e)}")
            return {}
    
    def calculate_portfolio_risk(self, weights: Dict[str, float], tickers: List[str],
                               start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate risk metrics for a specific portfolio allocation."""
        try:
            # Get portfolio data
            portfolio_data = self.data_service.get_multiple_stocks_data(tickers, start_date, end_date)
            returns_df = pd.DataFrame(portfolio_data['returns'])
            
            # Convert weights to array
            weight_array = np.array([weights.get(ticker, 0) for ticker in tickers])
            
            # Calculate portfolio returns
            portfolio_returns = returns_df.dot(weight_array)
            
            # Risk metrics
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            downside_vol = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252) if (portfolio_returns < 0).any() else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # VaR calculations
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)
            
            return {
                'portfolio_volatility': float(portfolio_vol),
                'downside_volatility': float(downside_vol),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'skewness': float(portfolio_returns.skew()),
                'kurtosis': float(portfolio_returns.kurtosis())
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            raise e
