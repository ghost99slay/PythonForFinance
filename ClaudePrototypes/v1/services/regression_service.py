"""
Regression Analysis Service
Implements simple and multivariate regression analysis for financial data.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from .data_service import DataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegressionService:
    """Service for regression analysis in financial contexts."""
    
    def __init__(self):
        self.data_service = DataService()
    
    def analyze_regression(self, file_path: Optional[str], dependent_var: str,
                          independent_vars: List[str], use_sample_data: bool = True) -> Dict[str, Any]:
        """
        Perform regression analysis on provided or sample data.
        
        Args:
            file_path: Path to CSV file (optional)
            dependent_var: Name of dependent variable
            independent_vars: List of independent variable names
            use_sample_data: Whether to use sample housing data if no file provided
            
        Returns:
            Dictionary containing regression analysis results
        """
        try:
            # Load data
            if file_path and file_path.strip():
                data = pd.read_csv(file_path)
            elif use_sample_data:
                data = self._create_sample_housing_data()
            else:
                raise ValueError("No data source provided")
            
            # Validate variables exist in data
            all_vars = [dependent_var] + independent_vars
            missing_vars = [var for var in all_vars if var not in data.columns]
            if missing_vars:
                # Try to match with available columns (case-insensitive)
                available_cols = data.columns.tolist()
                suggestions = self._suggest_column_matches(missing_vars, available_cols)
                raise ValueError(f"Variables not found: {missing_vars}. Available columns: {available_cols}. Suggestions: {suggestions}")
            
            # Prepare data
            X = data[independent_vars]
            y = data[dependent_var]
            
            # Remove any rows with NaN values
            combined_data = pd.concat([X, y], axis=1).dropna()
            X_clean = combined_data[independent_vars]
            y_clean = combined_data[dependent_var]
            
            if len(X_clean) == 0:
                raise ValueError("No valid data remaining after removing NaN values")
            
            # Perform simple regression if only one independent variable
            if len(independent_vars) == 1:
                regression_results = self._simple_regression(X_clean.iloc[:, 0], y_clean, independent_vars[0])
            else:
                regression_results = self._multivariate_regression(X_clean, y_clean, independent_vars)
            
            # Add data insights
            data_insights = self._generate_data_insights(X_clean, y_clean, dependent_var, independent_vars)
            
            # Create visualizations data
            chart_data = self._create_regression_chart_data(X_clean, y_clean, regression_results, independent_vars)
            
            return {
                'analysis_type': 'simple' if len(independent_vars) == 1 else 'multivariate',
                'dependent_variable': dependent_var,
                'independent_variables': independent_vars,
                'data_summary': {
                    'total_observations': len(combined_data),
                    'valid_observations': len(X_clean),
                    'missing_observations': len(data) - len(X_clean)
                },
                'regression_results': regression_results,
                'data_insights': data_insights,
                'chart_data': chart_data,
                'sample_predictions': self._generate_sample_predictions(X_clean, regression_results)
            }
            
        except Exception as e:
            logger.error(f"Error in regression analysis: {str(e)}")
            raise e
    
    def _simple_regression(self, X: pd.Series, y: pd.Series, var_name: str) -> Dict[str, Any]:
        """Perform simple linear regression."""
        try:
            # Using scipy.stats for detailed statistics
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            
            # Using statsmodels for additional statistics
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            # Calculate predictions and residuals
            y_pred = slope * X + intercept
            residuals = y - y_pred
            
            # Additional statistics
            n = len(X)
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            
            # Confidence intervals for slope
            t_val = stats.t.ppf(0.975, n-2)  # 95% confidence interval
            slope_ci_lower = slope - t_val * std_err
            slope_ci_upper = slope + t_val * std_err
            
            return {
                'model_type': 'Simple Linear Regression',
                'coefficients': {
                    'intercept': float(intercept),
                    var_name: float(slope)
                },
                'statistics': {
                    'r_squared': float(r_value**2),
                    'adjusted_r_squared': float(1 - (1 - r_value**2) * (n - 1) / (n - 2)),
                    'p_value': float(p_value),
                    'standard_error': float(std_err),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'f_statistic': float(model.fvalue) if hasattr(model, 'fvalue') else None,
                    'f_pvalue': float(model.f_pvalue) if hasattr(model, 'f_pvalue') else None
                },
                'confidence_intervals': {
                    'slope_lower': float(slope_ci_lower),
                    'slope_upper': float(slope_ci_upper)
                },
                'predictions': y_pred.tolist(),
                'residuals': residuals.tolist(),
                'equation': f"{dependent_var} = {intercept:.3f} + {slope:.3f} * {var_name}",
                'interpretation': self._interpret_simple_regression(slope, r_value**2, p_value, var_name)
            }
            
        except Exception as e:
            logger.error(f"Error in simple regression: {str(e)}")
            raise e
    
    def _multivariate_regression(self, X: pd.DataFrame, y: pd.Series, var_names: List[str]) -> Dict[str, Any]:
        """Perform multivariate linear regression."""
        try:
            # Using statsmodels for comprehensive statistics
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            # Using scikit-learn for additional metrics
            sklearn_model = LinearRegression()
            sklearn_model.fit(X, y)
            y_pred = sklearn_model.predict(X)
            
            # Extract coefficients and statistics
            coefficients = {'intercept': float(model.params['const'])}
            p_values = {'intercept': float(model.pvalues['const'])}
            conf_intervals = {'intercept': [float(model.conf_int().loc['const', 0]), float(model.conf_int().loc['const', 1])]}
            
            for i, var in enumerate(var_names):
                coefficients[var] = float(model.params[var])
                p_values[var] = float(model.pvalues[var])
                conf_intervals[var] = [float(model.conf_int().loc[var, 0]), float(model.conf_int().loc[var, 1])]
            
            # Calculate additional metrics
            residuals = y - y_pred
            n = len(X)
            k = len(var_names)
            
            # Adjusted R-squared
            adj_r_squared = 1 - (1 - model.rsquared) * (n - 1) / (n - k - 1)
            
            # Create equation string
            equation_parts = [f"{model.params['const']:.3f}"]
            for var in var_names:
                coef = model.params[var]
                sign = "+" if coef >= 0 else "-"
                equation_parts.append(f" {sign} {abs(coef):.3f} * {var}")
            equation = f"y = {''.join(equation_parts)}"
            
            return {
                'model_type': 'Multivariate Linear Regression',
                'coefficients': coefficients,
                'p_values': p_values,
                'confidence_intervals': conf_intervals,
                'statistics': {
                    'r_squared': float(model.rsquared),
                    'adjusted_r_squared': float(adj_r_squared),
                    'f_statistic': float(model.fvalue),
                    'f_pvalue': float(model.f_pvalue),
                    'mse': float(model.mse_resid),
                    'rmse': float(np.sqrt(model.mse_resid)),
                    'aic': float(model.aic),
                    'bic': float(model.bic)
                },
                'predictions': y_pred.tolist(),
                'residuals': residuals.tolist(),
                'equation': equation,
                'variable_significance': self._assess_variable_significance(p_values, var_names),
                'model_summary': str(model.summary()),
                'interpretation': self._interpret_multivariate_regression(model, var_names)
            }
            
        except Exception as e:
            logger.error(f"Error in multivariate regression: {str(e)}")
            raise e
    
    def _create_sample_housing_data(self) -> pd.DataFrame:
        """Create sample housing data similar to the course materials."""
        try:
            np.random.seed(42)  # For reproducible results
            n = 100
            
            # Generate synthetic housing data
            house_size = np.random.normal(1500, 500, n)
            house_size = np.clip(house_size, 500, 3000)  # Realistic range
            
            num_rooms = np.random.poisson(4, n) + 2  # 2-8 rooms typically
            num_rooms = np.clip(num_rooms, 2, 8)
            
            year_built = np.random.randint(1950, 2024, n)
            
            # House price based on realistic relationships
            base_price = 50000
            price_per_sqft = 150 + np.random.normal(0, 20, n)
            room_premium = 5000 + np.random.normal(0, 1000, n)
            year_premium = (year_built - 1950) * 500 + np.random.normal(0, 10000, n)
            
            house_price = (base_price + 
                          house_size * price_per_sqft + 
                          num_rooms * room_premium + 
                          year_premium +
                          np.random.normal(0, 25000, n))  # Random noise
            
            # Ensure positive prices
            house_price = np.maximum(house_price, 50000)
            
            return pd.DataFrame({
                'House Price': house_price,
                'House Size (sq.ft.)': house_size,
                'Number of Rooms': num_rooms,
                'Year of Construction': year_built
            })
            
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            raise e
    
    def _suggest_column_matches(self, missing_vars: List[str], available_cols: List[str]) -> Dict[str, List[str]]:
        """Suggest column matches for missing variables."""
        suggestions = {}
        
        for missing_var in missing_vars:
            missing_lower = missing_var.lower()
            matches = []
            
            for col in available_cols:
                col_lower = col.lower()
                # Simple matching based on substring
                if missing_lower in col_lower or col_lower in missing_lower:
                    matches.append(col)
            
            suggestions[missing_var] = matches[:3]  # Top 3 suggestions
        
        return suggestions
    
    def _interpret_simple_regression(self, slope: float, r_squared: float, 
                                   p_value: float, var_name: str) -> List[str]:
        """Generate interpretation for simple regression results."""
        interpretations = []
        
        # Slope interpretation
        if slope > 0:
            interpretations.append(f"For each unit increase in {var_name}, the dependent variable increases by {slope:.3f} units on average")
        else:
            interpretations.append(f"For each unit increase in {var_name}, the dependent variable decreases by {abs(slope):.3f} units on average")
        
        # R-squared interpretation
        if r_squared > 0.7:
            interpretations.append(f"Strong relationship: {r_squared:.1%} of variance is explained by {var_name}")
        elif r_squared > 0.3:
            interpretations.append(f"Moderate relationship: {r_squared:.1%} of variance is explained by {var_name}")
        else:
            interpretations.append(f"Weak relationship: Only {r_squared:.1%} of variance is explained by {var_name}")
        
        # Statistical significance
        if p_value < 0.001:
            interpretations.append(f"Highly statistically significant relationship (p < 0.001)")
        elif p_value < 0.01:
            interpretations.append(f"Statistically significant relationship (p < 0.01)")
        elif p_value < 0.05:
            interpretations.append(f"Statistically significant relationship (p < 0.05)")
        else:
            interpretations.append(f"No statistically significant relationship (p = {p_value:.3f})")
        
        return interpretations
    
    def _interpret_multivariate_regression(self, model, var_names: List[str]) -> List[str]:
        """Generate interpretation for multivariate regression results."""
        interpretations = []
        
        # Overall model fit
        r_squared = model.rsquared
        if r_squared > 0.7:
            interpretations.append(f"Strong model fit: {r_squared:.1%} of variance explained")
        elif r_squared > 0.3:
            interpretations.append(f"Moderate model fit: {r_squared:.1%} of variance explained")
        else:
            interpretations.append(f"Weak model fit: Only {r_squared:.1%} of variance explained")
        
        # F-statistic
        if model.f_pvalue < 0.05:
            interpretations.append(f"Model is statistically significant (F-statistic p-value: {model.f_pvalue:.3f})")
        else:
            interpretations.append(f"Model is not statistically significant (F-statistic p-value: {model.f_pvalue:.3f})")
        
        # Individual variables
        significant_vars = []
        insignificant_vars = []
        
        for var in var_names:
            if model.pvalues[var] < 0.05:
                significant_vars.append(var)
            else:
                insignificant_vars.append(var)
        
        if significant_vars:
            interpretations.append(f"Significant predictors: {', '.join(significant_vars)}")
        
        if insignificant_vars:
            interpretations.append(f"Non-significant predictors: {', '.join(insignificant_vars)}")
        
        return interpretations
    
    def _assess_variable_significance(self, p_values: Dict[str, float], var_names: List[str]) -> Dict[str, str]:
        """Assess the significance of each variable."""
        significance = {}
        
        for var in var_names:
            p_val = p_values[var]
            if p_val < 0.001:
                significance[var] = "Highly Significant (***)"
            elif p_val < 0.01:
                significance[var] = "Significant (**)"
            elif p_val < 0.05:
                significance[var] = "Marginally Significant (*)"
            else:
                significance[var] = "Not Significant"
        
        return significance
    
    def _generate_data_insights(self, X: pd.DataFrame, y: pd.Series, 
                              dependent_var: str, independent_vars: List[str]) -> List[str]:
        """Generate insights about the data."""
        insights = []
        
        try:
            # Data size
            insights.append(f"Analysis based on {len(X)} observations")
            
            # Dependent variable statistics
            insights.append(f"{dependent_var}: Mean = {y.mean():.2f}, Std = {y.std():.2f}")
            
            # Independent variable correlations with dependent variable
            correlations = []
            for var in independent_vars:
                corr = X[var].corr(y)
                correlations.append((var, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if correlations:
                strongest = correlations[0]
                insights.append(f"Strongest correlation with {dependent_var}: {strongest[0]} ({strongest[1]:.3f})")
            
            # Multicollinearity check for multivariate regression
            if len(independent_vars) > 1:
                corr_matrix = X.corr()
                high_corr_pairs = []
                
                for i in range(len(independent_vars)):
                    for j in range(i+1, len(independent_vars)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append((independent_vars[i], independent_vars[j], corr_val))
                
                if high_corr_pairs:
                    insights.append(f"High correlation between predictors detected - potential multicollinearity")
                else:
                    insights.append("No significant multicollinearity detected between predictors")
            
        except Exception as e:
            logger.error(f"Error generating data insights: {str(e)}")
            insights.append("Unable to generate detailed data insights")
        
        return insights
    
    def _create_regression_chart_data(self, X: pd.DataFrame, y: pd.Series,
                                    regression_results: Dict[str, Any], 
                                    var_names: List[str]) -> Dict[str, Any]:
        """Create chart data for regression visualization."""
        try:
            chart_data = {}
            
            # Scatter plot data for simple regression
            if len(var_names) == 1:
                chart_data['scatter_plot'] = {
                    'x': X.iloc[:, 0].tolist(),
                    'y': y.tolist(),
                    'x_label': var_names[0],
                    'y_label': y.name,
                    'regression_line': regression_results['predictions']
                }
            
            # Residual plots
            chart_data['residual_plot'] = {
                'fitted_values': regression_results['predictions'],
                'residuals': regression_results['residuals']
            }
            
            # Actual vs predicted
            chart_data['actual_vs_predicted'] = {
                'actual': y.tolist(),
                'predicted': regression_results['predictions']
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error creating regression chart data: {str(e)}")
            return {}
    
    def _generate_sample_predictions(self, X: pd.DataFrame, 
                                   regression_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sample predictions for demonstration."""
        try:
            sample_predictions = []
            
            # Take first 5 observations as examples
            for i in range(min(5, len(X))):
                prediction_data = {
                    'observation': i + 1,
                    'inputs': X.iloc[i].to_dict(),
                    'predicted_value': float(regression_results['predictions'][i])
                }
                sample_predictions.append(prediction_data)
            
            return sample_predictions
            
        except Exception as e:
            logger.error(f"Error generating sample predictions: {str(e)}")
            return []
    
    def compare_regression_models(self, data: pd.DataFrame, dependent_var: str,
                                model_specifications: List[List[str]]) -> Dict[str, Any]:
        """Compare multiple regression models with different variable specifications."""
        try:
            model_comparisons = {}
            
            for i, independent_vars in enumerate(model_specifications):
                model_name = f"Model_{i+1}"
                
                try:
                    # Prepare data
                    X = data[independent_vars]
                    y = data[dependent_var]
                    combined_data = pd.concat([X, y], axis=1).dropna()
                    X_clean = combined_data[independent_vars]
                    y_clean = combined_data[dependent_var]
                    
                    # Run regression
                    if len(independent_vars) == 1:
                        results = self._simple_regression(X_clean.iloc[:, 0], y_clean, independent_vars[0])
                    else:
                        results = self._multivariate_regression(X_clean, y_clean, independent_vars)
                    
                    model_comparisons[model_name] = {
                        'variables': independent_vars,
                        'r_squared': results['statistics']['r_squared'],
                        'adjusted_r_squared': results['statistics'].get('adjusted_r_squared'),
                        'aic': results['statistics'].get('aic'),
                        'bic': results['statistics'].get('bic'),
                        'rmse': results['statistics']['rmse']
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to fit {model_name}: {str(e)}")
                    continue
            
            # Determine best model
            if model_comparisons:
                best_model = max(model_comparisons.items(), 
                               key=lambda x: x[1].get('adjusted_r_squared', x[1]['r_squared']))
                
                return {
                    'model_comparisons': model_comparisons,
                    'best_model': best_model[0],
                    'comparison_criteria': 'Highest Adjusted R-squared'
                }
            else:
                return {'error': 'No models could be successfully fitted'}
            
        except Exception as e:
            logger.error(f"Error in model comparison: {str(e)}")
            raise e
