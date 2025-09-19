"""
Monte Carlo Simulation Service
Implements Monte Carlo simulations for stock price forecasting, options pricing, and scenario analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from scipy.stats import norm
from .data_service import DataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloService:
    """Service for Monte Carlo simulations in financial analysis."""
    
    def __init__(self):
        self.data_service = DataService()
    
    def run_simulation(self, ticker: str, start_date: datetime, end_date: datetime,
                      time_horizon: int = 252, simulations: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive Monte Carlo simulation for stock price forecasting.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            time_horizon: Number of days to simulate forward
            simulations: Number of simulation runs
            
        Returns:
            Dictionary containing simulation results
        """
        try:
            # Get historical data
            stock_data = self.data_service.get_stock_data(ticker, start_date, end_date)
            
            # Calculate parameters from historical data
            log_returns = pd.Series(stock_data['log_returns']).dropna()
            
            # Parameters for geometric Brownian motion
            mu = log_returns.mean()  # Drift (daily)
            sigma = log_returns.std()  # Volatility (daily)
            
            # Current stock price
            current_price = stock_data['statistics']['current_price']
            
            # Run stock price simulation
            price_paths = self._simulate_stock_prices(
                current_price, mu, sigma, time_horizon, simulations
            )
            
            # Calculate simulation statistics
            simulation_stats = self._calculate_simulation_statistics(price_paths, current_price)
            
            # Options pricing simulations
            options_analysis = self._simulate_options_pricing(
                price_paths, current_price, time_horizon
            )
            
            # Revenue simulation (business application)
            revenue_simulation = self._simulate_business_revenue(
                base_revenue=170, growth_mean=0.05, growth_std=0.15, simulations=simulations
            )
            
            # Generate insights
            insights = self._generate_monte_carlo_insights(
                ticker, simulation_stats, options_analysis, time_horizon
            )
            
            return {
                'ticker': ticker,
                'simulation_parameters': {
                    'time_horizon_days': time_horizon,
                    'number_of_simulations': simulations,
                    'current_price': current_price,
                    'historical_drift': float(mu),
                    'historical_volatility': float(sigma),
                    'annualized_drift': float(mu * 252),
                    'annualized_volatility': float(sigma * np.sqrt(252))
                },
                'price_simulation': {
                    'final_prices': price_paths[:, -1].tolist(),
                    'price_paths': price_paths.tolist(),
                    'statistics': simulation_stats
                },
                'options_analysis': options_analysis,
                'revenue_simulation': revenue_simulation,
                'insights': insights,
                'chart_data': self._create_monte_carlo_chart_data(price_paths, current_price, time_horizon)
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise e
    
    def _simulate_stock_prices(self, initial_price: float, mu: float, sigma: float,
                              time_horizon: int, simulations: int) -> np.ndarray:
        """Simulate stock prices using geometric Brownian motion."""
        try:
            # Initialize array to store price paths
            price_paths = np.zeros((simulations, time_horizon + 1))
            price_paths[:, 0] = initial_price
            
            # Generate random shocks
            random_shocks = np.random.normal(0, 1, (simulations, time_horizon))
            
            # Calculate daily returns using geometric Brownian motion
            dt = 1/252  # Daily time step (assuming 252 trading days per year)
            
            for t in range(1, time_horizon + 1):
                # Geometric Brownian Motion: dS = μS dt + σS dW
                # Discrete form: S(t+1) = S(t) * exp((μ - σ²/2) * dt + σ * sqrt(dt) * Z)
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[:, t-1]
                )
            
            return price_paths
            
        except Exception as e:
            logger.error(f"Error simulating stock prices: {str(e)}")
            return np.array([[initial_price]])
    
    def _calculate_simulation_statistics(self, price_paths: np.ndarray, 
                                       initial_price: float) -> Dict[str, Any]:
        """Calculate statistics from price simulation results."""
        try:
            final_prices = price_paths[:, -1]
            returns = (final_prices - initial_price) / initial_price
            
            statistics = {
                'final_price_mean': float(np.mean(final_prices)),
                'final_price_std': float(np.std(final_prices)),
                'final_price_min': float(np.min(final_prices)),
                'final_price_max': float(np.max(final_prices)),
                'final_price_median': float(np.median(final_prices)),
                'total_return_mean': float(np.mean(returns)),
                'total_return_std': float(np.std(returns)),
                'probability_positive': float(np.mean(returns > 0)),
                'probability_loss_5pct': float(np.mean(returns < -0.05)),
                'probability_loss_10pct': float(np.mean(returns < -0.10)),
                'probability_gain_10pct': float(np.mean(returns > 0.10)),
                'probability_gain_20pct': float(np.mean(returns > 0.20)),
                'percentiles': {
                    '5%': float(np.percentile(final_prices, 5)),
                    '10%': float(np.percentile(final_prices, 10)),
                    '25%': float(np.percentile(final_prices, 25)),
                    '75%': float(np.percentile(final_prices, 75)),
                    '90%': float(np.percentile(final_prices, 90)),
                    '95%': float(np.percentile(final_prices, 95))
                }
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating simulation statistics: {str(e)}")
            return {}
    
    def _simulate_options_pricing(self, price_paths: np.ndarray, current_price: float,
                                time_horizon: int, risk_free_rate: float = 0.025) -> Dict[str, Any]:
        """Simulate options pricing using Monte Carlo method."""
        try:
            final_prices = price_paths[:, -1]
            time_to_expiry = time_horizon / 252  # Convert to years
            
            # Define strike prices around current price
            strike_prices = [
                current_price * 0.9,   # 10% out of the money put / in the money call
                current_price * 0.95,  # 5% out of the money put / in the money call  
                current_price,         # At the money
                current_price * 1.05,  # 5% out of the money call / in the money put
                current_price * 1.1    # 10% out of the money call / in the money put
            ]
            
            options_results = {}
            
            for strike in strike_prices:
                # Call option payoffs
                call_payoffs = np.maximum(final_prices - strike, 0)
                call_price = np.mean(call_payoffs) * np.exp(-risk_free_rate * time_to_expiry)
                
                # Put option payoffs
                put_payoffs = np.maximum(strike - final_prices, 0)
                put_price = np.mean(put_payoffs) * np.exp(-risk_free_rate * time_to_expiry)
                
                # Calculate probabilities
                call_prob_itm = np.mean(final_prices > strike)  # Probability of finishing in the money
                put_prob_itm = np.mean(final_prices < strike)
                
                strike_key = f"strike_{strike:.2f}"
                options_results[strike_key] = {
                    'strike_price': float(strike),
                    'call_price': float(call_price),
                    'put_price': float(put_price),
                    'call_probability_itm': float(call_prob_itm),
                    'put_probability_itm': float(put_prob_itm),
                    'moneyness': float(strike / current_price)
                }
            
            # Black-Scholes comparison for at-the-money option
            bs_comparison = self._black_scholes_comparison(
                current_price, current_price, time_to_expiry, risk_free_rate,
                np.std(np.log(price_paths[:, 1:] / price_paths[:, :-1])) * np.sqrt(252)
            )
            
            return {
                'options_prices': options_results,
                'black_scholes_comparison': bs_comparison,
                'simulation_parameters': {
                    'risk_free_rate': risk_free_rate,
                    'time_to_expiry': time_to_expiry
                }
            }
            
        except Exception as e:
            logger.error(f"Error in options pricing simulation: {str(e)}")
            return {}
    
    def _black_scholes_comparison(self, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, Any]:
        """Calculate Black-Scholes option prices for comparison."""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return {
                'call_price': float(call_price),
                'put_price': float(put_price),
                'd1': float(d1),
                'd2': float(d2)
            }
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {str(e)}")
            return {}
    
    def _simulate_business_revenue(self, base_revenue: float = 170, growth_mean: float = 0.05,
                                 growth_std: float = 0.15, simulations: int = 1000) -> Dict[str, Any]:
        """Simulate business revenue scenarios (example from notebooks)."""
        try:
            # Simulate revenue with normal distribution
            revenues = np.random.normal(base_revenue, base_revenue * 0.12, simulations)  # 12% volatility
            
            # Simulate COGS (Cost of Goods Sold) as percentage of revenue
            cogs_percentage = np.random.normal(0.6, 0.1, simulations)  # 60% ± 10%
            cogs = -(revenues * cogs_percentage)  # Negative because it's a cost
            
            # Calculate gross profit
            gross_profit = revenues + cogs  # cogs is negative
            
            # Simulate operating expenses
            opex_percentage = np.random.normal(0.25, 0.05, simulations)  # 25% ± 5%
            opex = -(revenues * opex_percentage)
            
            # Calculate operating profit
            operating_profit = gross_profit + opex
            
            return {
                'revenue_statistics': {
                    'mean': float(np.mean(revenues)),
                    'std': float(np.std(revenues)),
                    'min': float(np.min(revenues)),
                    'max': float(np.max(revenues)),
                    'percentiles': {
                        '10%': float(np.percentile(revenues, 10)),
                        '25%': float(np.percentile(revenues, 25)),
                        '75%': float(np.percentile(revenues, 75)),
                        '90%': float(np.percentile(revenues, 90))
                    }
                },
                'gross_profit_statistics': {
                    'mean': float(np.mean(gross_profit)),
                    'std': float(np.std(gross_profit)),
                    'probability_positive': float(np.mean(gross_profit > 0))
                },
                'operating_profit_statistics': {
                    'mean': float(np.mean(operating_profit)),
                    'std': float(np.std(operating_profit)),
                    'probability_positive': float(np.mean(operating_profit > 0))
                },
                'simulation_data': {
                    'revenues': revenues.tolist(),
                    'gross_profits': gross_profit.tolist(),
                    'operating_profits': operating_profit.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in business revenue simulation: {str(e)}")
            return {}
    
    def _generate_monte_carlo_insights(self, ticker: str, simulation_stats: Dict[str, Any],
                                     options_analysis: Dict[str, Any], time_horizon: int) -> List[str]:
        """Generate insights from Monte Carlo simulation results."""
        insights = []
        
        try:
            # Price prediction insights
            if simulation_stats:
                mean_price = simulation_stats['final_price_mean']
                current_implied = simulation_stats.get('current_price', mean_price)
                expected_return = (mean_price - current_implied) / current_implied
                
                insights.append(f"Expected price after {time_horizon} days: ${mean_price:.2f}")
                insights.append(f"Expected total return: {expected_return:.1%}")
                
                prob_positive = simulation_stats['probability_positive']
                insights.append(f"Probability of positive return: {prob_positive:.1%}")
                
                if simulation_stats['probability_loss_10pct'] > 0.2:
                    insights.append(f"High risk: {simulation_stats['probability_loss_10pct']:.1%} chance of >10% loss")
                
                if simulation_stats['probability_gain_20pct'] > 0.3:
                    insights.append(f"High upside potential: {simulation_stats['probability_gain_20pct']:.1%} chance of >20% gain")
            
            # Volatility insights
            if 'final_price_std' in simulation_stats:
                price_volatility = simulation_stats['final_price_std'] / simulation_stats['final_price_mean']
                if price_volatility > 0.3:
                    insights.append(f"High price uncertainty with {price_volatility:.1%} coefficient of variation")
                elif price_volatility < 0.1:
                    insights.append(f"Relatively stable price projection with {price_volatility:.1%} coefficient of variation")
            
            # Options insights
            if options_analysis and 'options_prices' in options_analysis:
                atm_options = None
                for key, option_data in options_analysis['options_prices'].items():
                    if abs(option_data['moneyness'] - 1.0) < 0.01:  # At-the-money
                        atm_options = option_data
                        break
                
                if atm_options:
                    insights.append(f"At-the-money call option fair value: ${atm_options['call_price']:.2f}")
                    insights.append(f"At-the-money put option fair value: ${atm_options['put_price']:.2f}")
            
            # Risk management insights
            if 'percentiles' in simulation_stats:
                worst_5pct = simulation_stats['percentiles']['5%']
                current_price = simulation_stats.get('current_price', simulation_stats['final_price_mean'])
                worst_case_loss = (current_price - worst_5pct) / current_price
                insights.append(f"Worst-case scenario (5th percentile): ${worst_5pct:.2f} ({worst_case_loss:.1%} loss)")
            
            # Simulation reliability
            insights.append(f"Simulation based on historical volatility and assumes geometric Brownian motion")
            insights.append(f"Results are probabilistic estimates, not guaranteed outcomes")
            
        except Exception as e:
            logger.error(f"Error generating Monte Carlo insights: {str(e)}")
            insights.append("Unable to generate detailed insights due to calculation error")
        
        return insights
    
    def _create_monte_carlo_chart_data(self, price_paths: np.ndarray, current_price: float,
                                     time_horizon: int) -> Dict[str, Any]:
        """Create chart data for Monte Carlo visualization."""
        try:
            # Sample a subset of paths for visualization (to avoid overcrowding)
            sample_size = min(100, price_paths.shape[0])
            sample_indices = np.random.choice(price_paths.shape[0], sample_size, replace=False)
            sample_paths = price_paths[sample_indices]
            
            # Create time axis
            time_axis = list(range(time_horizon + 1))
            
            # Final price distribution
            final_prices = price_paths[:, -1]
            
            chart_data = {
                'price_paths': {
                    'time_axis': time_axis,
                    'sample_paths': sample_paths.tolist(),
                    'mean_path': np.mean(price_paths, axis=0).tolist(),
                    'percentile_5': np.percentile(price_paths, 5, axis=0).tolist(),
                    'percentile_95': np.percentile(price_paths, 95, axis=0).tolist()
                },
                'final_price_distribution': {
                    'prices': final_prices.tolist(),
                    'current_price': current_price
                },
                'return_distribution': {
                    'returns': ((final_prices - current_price) / current_price).tolist()
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating Monte Carlo chart data: {str(e)}")
            return {}
    
    def simulate_portfolio_scenarios(self, tickers: List[str], weights: Dict[str, float],
                                   start_date: datetime, end_date: datetime,
                                   time_horizon: int = 252, simulations: int = 1000) -> Dict[str, Any]:
        """Simulate portfolio performance scenarios."""
        try:
            # Get portfolio data
            portfolio_data = self.data_service.get_multiple_stocks_data(tickers, start_date, end_date)
            returns_df = pd.DataFrame(portfolio_data['returns'])
            
            # Calculate portfolio parameters
            weight_array = np.array([weights.get(ticker, 0) for ticker in tickers])
            
            # Portfolio returns and covariance
            portfolio_returns = returns_df.dot(weight_array)
            mu_portfolio = portfolio_returns.mean()
            sigma_portfolio = portfolio_returns.std()
            
            # Current portfolio value (assuming $10,000 initial investment)
            initial_value = 10000
            
            # Simulate portfolio values
            portfolio_paths = self._simulate_stock_prices(
                initial_value, mu_portfolio, sigma_portfolio, time_horizon, simulations
            )
            
            # Calculate statistics
            final_values = portfolio_paths[:, -1]
            returns = (final_values - initial_value) / initial_value
            
            return {
                'portfolio_simulation': {
                    'initial_value': initial_value,
                    'final_values': final_values.tolist(),
                    'returns': returns.tolist(),
                    'value_paths': portfolio_paths.tolist(),
                    'statistics': {
                        'expected_final_value': float(np.mean(final_values)),
                        'expected_return': float(np.mean(returns)),
                        'volatility': float(np.std(returns)),
                        'probability_loss': float(np.mean(returns < 0)),
                        'max_loss_5pct': float(np.percentile(returns, 5)),
                        'max_gain_95pct': float(np.percentile(returns, 95))
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio scenario simulation: {str(e)}")
            raise e
