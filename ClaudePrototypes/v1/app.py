"""
Python for Finance Web Application
A comprehensive Flask application incorporating financial analysis concepts
including Portfolio Theory, CAPM, Monte Carlo Simulations, and more.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SelectField, IntegerField, SelectMultipleField, DateField
from wtforms.validators import DataRequired, NumberRange, Length
import os
from datetime import datetime, date
import json

# Import our custom services
from services.portfolio_service import PortfolioService
from services.capm_service import CAPMService
from services.risk_service import RiskService
from services.monte_carlo_service import MonteCarloService
from services.regression_service import RegressionService
from services.data_service import DataService

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize services
portfolio_service = PortfolioService()
capm_service = CAPMService()
risk_service = RiskService()
monte_carlo_service = MonteCarloService()
regression_service = RegressionService()
data_service = DataService()

# Forms for user input
class PortfolioForm(FlaskForm):
    tickers = StringField('Stock Tickers (comma-separated)', 
                         validators=[DataRequired(), Length(min=1, max=200)],
                         default='AAPL,MSFT,GOOGL,TSLA')
    start_date = DateField('Start Date', validators=[DataRequired()], 
                          default=date(2020, 1, 1))
    end_date = DateField('End Date', validators=[DataRequired()], 
                        default=date.today())

class CAPMForm(FlaskForm):
    ticker = StringField('Stock Ticker', validators=[DataRequired(), Length(min=1, max=10)],
                        default='AAPL')
    market_ticker = StringField('Market Index Ticker', validators=[DataRequired()],
                               default='^GSPC')
    start_date = DateField('Start Date', validators=[DataRequired()],
                          default=date(2019, 1, 1))
    end_date = DateField('End Date', validators=[DataRequired()],
                        default=date.today())
    risk_free_rate = FloatField('Risk-Free Rate (%)', validators=[DataRequired()],
                               default=2.5)

class MonteCarloForm(FlaskForm):
    ticker = StringField('Stock Ticker', validators=[DataRequired()],
                        default='AAPL')
    start_date = DateField('Start Date', validators=[DataRequired()],
                          default=date(2020, 1, 1))
    end_date = DateField('End Date', validators=[DataRequired()],
                        default=date.today())
    time_horizon = IntegerField('Time Horizon (days)', validators=[DataRequired()],
                               default=252)
    simulations = IntegerField('Number of Simulations', validators=[DataRequired()],
                              default=1000)

class RegressionForm(FlaskForm):
    file_upload = StringField('CSV File Path (optional)', default='')
    dependent_var = StringField('Dependent Variable', default='House Price')
    independent_vars = StringField('Independent Variables (comma-separated)',
                                  default='House Size (sq.ft.)')

@app.route('/')
def index():
    """Main dashboard showing overview of all financial tools."""
    return render_template('index.html')

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio_analysis():
    """Portfolio Theory and Markowitz Efficient Frontier."""
    form = PortfolioForm()
    results = None
    
    if form.validate_on_submit():
        try:
            tickers = [t.strip().upper() for t in form.tickers.data.split(',')]
            results = portfolio_service.analyze_portfolio(
                tickers, form.start_date.data, form.end_date.data
            )
            flash('Portfolio analysis completed successfully!', 'success')
        except Exception as e:
            flash(f'Error in portfolio analysis: {str(e)}', 'error')
    
    return render_template('portfolio.html', form=form, results=results)

@app.route('/capm', methods=['GET', 'POST'])
def capm_analysis():
    """Capital Asset Pricing Model analysis."""
    form = CAPMForm()
    results = None
    
    if form.validate_on_submit():
        try:
            results = capm_service.analyze_capm(
                form.ticker.data.upper(),
                form.market_ticker.data.upper(),
                form.start_date.data,
                form.end_date.data,
                form.risk_free_rate.data / 100
            )
            flash('CAPM analysis completed successfully!', 'success')
        except Exception as e:
            flash(f'Error in CAPM analysis: {str(e)}', 'error')
    
    return render_template('capm.html', form=form, results=results)

@app.route('/risk', methods=['GET', 'POST'])
def risk_analysis():
    """Investment Risk Analysis including correlation and diversification."""
    form = PortfolioForm()  # Reuse the same form
    results = None
    
    if form.validate_on_submit():
        try:
            tickers = [t.strip().upper() for t in form.tickers.data.split(',')]
            results = risk_service.analyze_risk(
                tickers, form.start_date.data, form.end_date.data
            )
            flash('Risk analysis completed successfully!', 'success')
        except Exception as e:
            flash(f'Error in risk analysis: {str(e)}', 'error')
    
    return render_template('risk.html', form=form, results=results)

@app.route('/monte-carlo', methods=['GET', 'POST'])
def monte_carlo_simulation():
    """Monte Carlo simulations for stock price forecasting."""
    form = MonteCarloForm()
    results = None
    
    if form.validate_on_submit():
        try:
            results = monte_carlo_service.run_simulation(
                form.ticker.data.upper(),
                form.start_date.data,
                form.end_date.data,
                form.time_horizon.data,
                form.simulations.data
            )
            flash('Monte Carlo simulation completed successfully!', 'success')
        except Exception as e:
            flash(f'Error in Monte Carlo simulation: {str(e)}', 'error')
    
    return render_template('monte_carlo.html', form=form, results=results)

@app.route('/regression', methods=['GET', 'POST'])
def regression_analysis():
    """Simple and Multivariate Regression Analysis."""
    form = RegressionForm()
    results = None
    
    if form.validate_on_submit():
        try:
            results = regression_service.analyze_regression(
                form.file_upload.data,
                form.dependent_var.data,
                [v.strip() for v in form.independent_vars.data.split(',')]
            )
            flash('Regression analysis completed successfully!', 'success')
        except Exception as e:
            flash(f'Error in regression analysis: {str(e)}', 'error')
    
    return render_template('regression.html', form=form, results=results)

@app.route('/api/stock-data/<ticker>')
def get_stock_data(ticker):
    """API endpoint to get stock data for AJAX requests."""
    try:
        data = data_service.get_stock_data(ticker.upper())
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/education')
def education():
    """Educational resources explaining financial concepts."""
    return render_template('education.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
