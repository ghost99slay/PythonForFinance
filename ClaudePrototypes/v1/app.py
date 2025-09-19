"""
Python for Finance Web Application
A comprehensive Flask application incorporating financial analysis concepts
including Portfolio Theory, CAPM, Monte Carlo Simulations, and more.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SelectField, IntegerField, SelectMultipleField, DateField, TextAreaField
from wtforms.validators import DataRequired, NumberRange, Length
import os
from datetime import datetime, date
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom services
from services.portfolio_service import PortfolioService
from services.capm_service import CAPMService
from services.risk_service import RiskService
from services.monte_carlo_service import MonteCarloService
from services.regression_service import RegressionService
from services.data_service import DataService
from services.index_service import IndexService

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize services
portfolio_service = PortfolioService()
capm_service = CAPMService()
risk_service = RiskService()
monte_carlo_service = MonteCarloService()
regression_service = RegressionService()
data_service = DataService()
index_service = IndexService()

# Forms for user input
class PortfolioForm(FlaskForm):
    index_selection = SelectField('Stock Index (Optional)',
                                  choices=[('', 'Select an index...')] +
                                         [('SP500', 'S&P 500 Index (~500 stocks)'),
                                          ('NASDAQ100', 'NASDAQ-100 Index (~100 stocks)'),
                                          ('DOW30', 'Dow Jones Industrial Average (30 stocks)'),
                                          ('RUSSELL2000', 'Russell 2000 Index (~2000 stocks)'),
                                          ('RUSSELL1000', 'Russell 1000 Index (~1000 stocks)')],
                                  default='')
    tickers = TextAreaField('Stock Tickers (comma-separated)', 
                           validators=[DataRequired(), Length(min=1, max=50000)],  # Allow for large indices
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
    try:
        logger.info("Index route accessed")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return f"Index error: {str(e)}", 500

@app.route('/test')
def test_route():
    """Simple test route to check if Flask is working."""
    return "Flask is working! ✅"

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio_analysis():
    """Portfolio Theory and Markowitz Efficient Frontier."""
    try:
        logger.info(f"Portfolio route accessed - Method: {request.method}")
        form = PortfolioForm()
        results = None
        
        if request.method == 'POST':
            logger.info(f"Portfolio form submitted. Form valid: {form.validate_on_submit()}")
            if form.errors:
                logger.error(f"Form validation errors: {form.errors}")
        
        if form.validate_on_submit():
            try:
                tickers = [t.strip().upper() for t in form.tickers.data.split(',') if t.strip()]
                logger.info(f"Starting portfolio analysis for {len(tickers)} tickers: {tickers[:10]}...")
                
                # Provide user feedback for large portfolios but don't limit
                if len(tickers) > 300:
                    flash(f'Analyzing full index with {len(tickers)} stocks. This is a comprehensive analysis and may take 2-5 minutes.', 'info')
                elif len(tickers) > 100:
                    flash(f'Analyzing large portfolio with {len(tickers)} stocks. This may take 1-3 minutes.', 'info')
                
                results = portfolio_service.analyze_portfolio(
                    tickers, form.start_date.data, form.end_date.data
                )
                logger.info("Portfolio analysis completed successfully")
                
                # Show user how many tickers were successfully analyzed
                if 'valid_tickers_analyzed' in results and 'original_tickers_requested' in results:
                    analyzed = results['valid_tickers_analyzed']
                    requested = results['original_tickers_requested']
                    if analyzed == requested:
                        flash(f'Portfolio analysis completed successfully! Analyzed all {analyzed} stocks.', 'success')
                    else:
                        excluded = requested - analyzed
                        flash(f'Portfolio analysis completed! Analyzed {analyzed}/{requested} stocks. {excluded} stocks excluded due to missing data.', 'info')
                else:
                    flash('Portfolio analysis completed successfully!', 'success')
            except Exception as e:
                logger.error(f"Portfolio analysis failed: {str(e)}")
                flash(f'Error in portfolio analysis: {str(e)}', 'error')
        
        logger.info("Rendering portfolio template")
        return render_template('portfolio.html', form=form, results=results)
        
    except Exception as e:
        logger.error(f"Error in portfolio route: {str(e)}")
        flash(f'Application error: {str(e)}', 'error')
        # Try to create a basic form to prevent complete failure
        try:
            form = PortfolioForm()
            return render_template('portfolio.html', form=form, results=None)
        except Exception as e2:
            logger.error(f"Critical error: {str(e2)}")
            return f"Critical application error: {str(e2)}", 500

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

@app.route('/api/index-constituents/<index_symbol>')
def get_index_constituents(index_symbol):
    """API endpoint to get index constituents for AJAX requests."""
    try:
        max_stocks = int(request.args.get('max_stocks', 1000))  # Allow up to 1000 stocks
        skip_validation = request.args.get('skip_validation', 'false').lower() == 'true'
        
        # Get date range from request if provided
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        start_date = None
        end_date = None
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                pass
        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            except ValueError:
                pass
        
        tickers, metadata = index_service.get_index_constituents(
            index_symbol.upper(), max_stocks, skip_validation, start_date, end_date
        )
        
        return jsonify({
            'success': True,
            'tickers': tickers,
            'metadata': metadata,
            'tickers_string': ','.join(tickers)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'tickers': [],
            'metadata': {},
            'tickers_string': ''
        }), 400

@app.route('/api/available-indices')
def get_available_indices():
    """API endpoint to get list of available indices."""
    try:
        indices = index_service.get_available_indices()
        return jsonify({
            'success': True,
            'indices': indices
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'indices': {}
        }), 400

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
