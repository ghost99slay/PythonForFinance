#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality
of the Python for Finance Web Application.
"""

import sys
import importlib
from datetime import datetime, date

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    required_modules = [
        'flask', 'pandas', 'numpy', 'scipy', 'yfinance',
        'statsmodels', 'sklearn', 'matplotlib', 'plotly'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - NOT FOUND")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required modules are available!")
        return True

def test_services():
    """Test that all service modules can be imported and initialized."""
    print("\n🔍 Testing services...")
    
    try:
        from services.data_service import DataService
        from services.portfolio_service import PortfolioService
        from services.capm_service import CAPMService
        from services.risk_service import RiskService
        from services.monte_carlo_service import MonteCarloService
        from services.regression_service import RegressionService
        
        # Test service initialization
        data_service = DataService()
        portfolio_service = PortfolioService()
        capm_service = CAPMService()
        risk_service = RiskService()
        monte_carlo_service = MonteCarloService()
        regression_service = RegressionService()
        
        print("   ✅ DataService")
        print("   ✅ PortfolioService")
        print("   ✅ CAPMService")
        print("   ✅ RiskService")
        print("   ✅ MonteCarloService")
        print("   ✅ RegressionService")
        
        print("✅ All services initialized successfully!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Service initialization error: {e}")
        return False

def test_flask_app():
    """Test that the Flask app can be created."""
    print("\n🔍 Testing Flask application...")
    
    try:
        from app import app
        
        # Test that the app was created
        if app is None:
            print("   ❌ Flask app is None")
            return False
        
        # Test that routes are registered
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/portfolio', '/capm', '/risk', '/monte-carlo', '/regression', '/education']
        
        missing_routes = []
        for route in expected_routes:
            if route not in routes:
                missing_routes.append(route)
        
        if missing_routes:
            print(f"   ❌ Missing routes: {missing_routes}")
            return False
        
        print("   ✅ Flask app created successfully")
        print("   ✅ All expected routes registered")
        print("✅ Flask application test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Flask app error: {e}")
        return False

def test_sample_data():
    """Test sample data generation."""
    print("\n🔍 Testing sample data generation...")
    
    try:
        from services.regression_service import RegressionService
        
        regression_service = RegressionService()
        sample_data = regression_service._create_sample_housing_data()
        
        if sample_data is None or len(sample_data) == 0:
            print("   ❌ Sample data generation failed")
            return False
        
        expected_columns = ['House Price', 'House Size (sq.ft.)', 'Number of Rooms', 'Year of Construction']
        missing_columns = [col for col in expected_columns if col not in sample_data.columns]
        
        if missing_columns:
            print(f"   ❌ Missing columns in sample data: {missing_columns}")
            return False
        
        print(f"   ✅ Sample data generated with {len(sample_data)} rows")
        print(f"   ✅ All expected columns present: {list(sample_data.columns)}")
        print("✅ Sample data generation test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Sample data error: {e}")
        return False

def test_index_service():
    """Test the new index service functionality."""
    print("\n🔍 Testing index service...")
    
    try:
        from services.index_service import IndexService
        
        index_service = IndexService()
        
        # Test getting available indices
        indices = index_service.get_available_indices()
        if not indices or 'SPY' not in indices:
            print("   ❌ Available indices not working")
            return False
        print("   ✅ Available indices retrieved")
        
        # Test getting index info
        spy_info = index_service.get_index_info('SPY')
        if not spy_info or 'name' not in spy_info:
            print("   ❌ Index info not working")
            return False
        print("   ✅ Index info retrieved")
        
        # Test predefined constituents structure
        for symbol, data in index_service.predefined_constituents.items():
            if not all(key in data for key in ['name', 'description', 'tickers']):
                print(f"   ❌ Invalid structure for {symbol}")
                return False
            if not data['tickers']:
                print(f"   ❌ No tickers for {symbol}")
                return False
        print("   ✅ Predefined constituents structure valid")
        
        print("✅ Index service test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Index service error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints with the Flask test client."""
    print("\n🔍 Testing API endpoints...")
    
    try:
        from app import app
        
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Test available indices API
        response = client.get('/api/available-indices')
        if response.status_code != 200:
            print(f"   ❌ Available indices API failed: {response.status_code}")
            return False
        print("   ✅ Available indices API working")
        
        # Test portfolio page loads
        response = client.get('/portfolio')
        if response.status_code != 200:
            print(f"   ❌ Portfolio page failed: {response.status_code}")
            return False
        
        response_data = response.get_data(as_text=True)
        if 'Index Fund Selection' not in response_data:
            print("   ❌ Portfolio page missing index dropdown")
            return False
        print("   ✅ Portfolio page with index dropdown working")
        
        print("✅ API endpoints test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ API endpoints error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Python for Finance - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_services,
        test_flask_app,
        test_sample_data,
        test_index_service,
        test_api_endpoints
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ Your installation is ready!")
        print("🚀 Start the application with: python run.py")
        print("🌐 Then open: http://localhost:5000")
        return True
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("\n🔧 Please fix the issues above before running the application.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
