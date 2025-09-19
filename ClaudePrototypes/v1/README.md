# Python for Finance Web Application

A comprehensive financial analysis platform implementing modern financial theory concepts including Portfolio Theory, CAPM, Risk Analysis, Monte Carlo Simulations, and Regression Analysis.

## Features

### 🏦 Portfolio Analysis

- **Markowitz Modern Portfolio Theory** implementation
- **Efficient Frontier** calculation and visualization
- **Optimal portfolio allocation** (Maximum Sharpe ratio, Minimum volatility)
- **Stock index analysis** with real-time constituent fetching:
  - **S&P 500** (~500 largest US companies)
  - **NASDAQ-100** (~100 largest non-financial NASDAQ companies)
  - **Dow Jones Industrial Average** (30 large US companies)
  - **Russell 2000/1000** (Small/large-cap companies)
- **Web scraping** of current index constituents from Wikipedia
- **Date range validation** excluding stocks without data in analysis period
- **Automatic ticker validation** and data quality checks
- **Correlation analysis** and diversification metrics
- **Risk-return optimization**

### 📈 CAPM Analysis

- **Beta calculation** with confidence intervals
- **Alpha analysis** and performance attribution
- **Expected return** calculation using CAPM
- **Risk-adjusted performance metrics** (Sharpe ratio, Treynor ratio)
- **Security characteristic line** visualization

### ⚡ Risk Analysis

- **Comprehensive risk metrics** (volatility, skewness, kurtosis)
- **Value at Risk (VaR)** and Conditional VaR calculation
- **Correlation matrices** and diversification analysis
- **Systematic vs. idiosyncratic risk** decomposition
- **Maximum drawdown** and downside deviation

### 🎲 Monte Carlo Simulations

- **Stock price forecasting** using geometric Brownian motion
- **Options pricing** with Black-Scholes comparison
- **Scenario analysis** and probability distributions
- **Business revenue simulation** for planning
- **Risk probability calculations**

### 📊 Regression Analysis

- **Simple and multivariate regression** analysis
- **Statistical significance testing** and diagnostics
- **Model comparison** and selection
- **Residual analysis** and assumption validation
- **Sample housing data** for demonstration

### 📚 Educational Resources

- **Interactive learning modules** for each concept
- **Mathematical foundations** and formulas
- **Practical applications** and examples
- **Comprehensive explanations** of financial theory

## Technology Stack

- **Backend:** Flask (Python web framework)
- **Data Analysis:** pandas, NumPy, SciPy, statsmodels, scikit-learn
- **Financial Data:** yfinance for real-time market data
- **Visualization:** Plotly.js for interactive charts
- **Frontend:** Bootstrap 5 for responsive UI
- **Forms:** Flask-WTF for form handling

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd ClaudePrototypes/v1
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables (optional):**

   ```bash
   export SECRET_KEY='your-secret-key-here'
   ```

5. **Run the application:**

   ```bash
   python app.py
   ```

6. **Open in browser:**
   Navigate to `http://localhost:5000`

## Usage Guide

### Portfolio Analysis

1. Enter comma-separated stock tickers (e.g., `AAPL,MSFT,GOOGL,TSLA`)
2. Select analysis date range
3. View efficient frontier, optimal allocations, and correlation analysis

### CAPM Analysis

1. Enter a single stock ticker
2. Choose market benchmark (default: S&P 500)
3. Set risk-free rate and date range
4. Analyze beta, alpha, and expected returns

### Risk Analysis

1. Input multiple stock tickers for portfolio risk assessment
2. Review individual and portfolio risk metrics
3. Examine correlation matrices and diversification benefits

### Monte Carlo Simulation

1. Select stock ticker and historical data period
2. Configure simulation parameters (time horizon, number of simulations)
3. Analyze price forecasts, probabilities, and options pricing

### Regression Analysis

1. Define dependent and independent variables
2. Upload CSV data or use sample housing dataset
3. Review regression results, coefficients, and model diagnostics

## API Endpoints

- `/` - Main dashboard
- `/portfolio` - Portfolio analysis tool with index fund selection
- `/capm` - CAPM analysis tool
- `/risk` - Risk analysis tool
- `/monte-carlo` - Monte Carlo simulation tool
- `/regression` - Regression analysis tool
- `/education` - Educational resources
- `/api/stock-data/<ticker>` - JSON API for stock data
- `/api/available-indices` - JSON API for available index funds
- `/api/index-constituents/<index>` - JSON API for index constituents with validation

## File Structure

```
v1/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── services/             # Business logic services
│   ├── __init__.py
│   ├── data_service.py   # Financial data retrieval
│   ├── portfolio_service.py  # Portfolio analysis
│   ├── capm_service.py   # CAPM calculations
│   ├── risk_service.py   # Risk analysis
│   ├── monte_carlo_service.py  # Monte Carlo simulations
│   ├── regression_service.py   # Regression analysis
│   └── index_service.py  # Index fund constituents and validation
├── app/
│   ├── templates/        # HTML templates
│   │   ├── base.html     # Base template
│   │   ├── index.html    # Dashboard
│   │   ├── portfolio.html # Portfolio analysis
│   │   ├── capm.html     # CAPM analysis
│   │   ├── risk.html     # Risk analysis
│   │   ├── monte_carlo.html # Monte Carlo
│   │   ├── regression.html  # Regression
│   │   ├── education.html   # Educational content
│   │   └── errors/       # Error pages
│   └── static/           # Static assets (CSS, JS, images)
├── models/               # Data models (if needed)
├── utils/                # Utility functions
└── tests/                # Unit tests
```

## Features in Detail

### Real-time Financial Data

- Fetches live market data using Yahoo Finance API
- Caches data to improve performance
- Handles missing data and data validation
- Supports multiple asset classes and market indices

### Advanced Analytics

- Implements academic financial models with proper mathematical foundations
- Provides comprehensive statistical analysis and diagnostics
- Generates actionable insights and recommendations
- Supports both historical analysis and forward-looking projections

### Interactive Visualizations

- Dynamic charts using Plotly.js for better data interpretation
- Responsive design that works on desktop and mobile devices
- Interactive elements for exploring data relationships
- Export capabilities for charts and analysis results

### Educational Integration

- Each tool includes links to relevant educational content
- Mathematical formulas and theoretical explanations
- Practical examples and use cases
- Progressive learning from basic to advanced concepts

## Contributing

This is a prototype application demonstrating financial analysis concepts. Potential improvements:

1. **Database Integration:** Store analysis results and user portfolios
2. **User Authentication:** Personal portfolios and saved analyses
3. **More Asset Classes:** Bonds, commodities, cryptocurrencies
4. **Advanced Models:** Fama-French factors, Black-Litterman
5. **Real-time Updates:** WebSocket integration for live data
6. **API Expansion:** RESTful API for programmatic access

## Disclaimer

This application is for educational purposes only. It should not be used as the sole basis for investment decisions. Always consult with qualified financial professionals before making investment choices.

## License

This project is created for educational purposes as part of a Python for Finance learning initiative.

## Contact

For questions or suggestions about this implementation, please refer to the educational materials or consult financial literature for theoretical foundations.
