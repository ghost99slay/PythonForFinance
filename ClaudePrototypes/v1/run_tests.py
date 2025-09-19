#!/usr/bin/env python3
"""
Comprehensive test runner for the Python for Finance application.
Includes unit tests, integration tests, and API tests.
"""

import unittest
import sys
import os
from datetime import datetime

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_unit_tests():
    """Run unit tests for individual services."""
    print("🧪 Running Unit Tests")
    print("=" * 50)
    
    try:
        # Import and run index service tests
        from tests.test_index_service import TestIndexService
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestIndexService))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ Could not import unit tests: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running unit tests: {e}")
        return False

def run_integration_tests():
    """Run integration tests for Flask app."""
    print("\n🔗 Running Integration Tests")
    print("=" * 50)
    
    try:
        # Import and run Flask integration tests
        from tests.test_app_integration import TestAppIntegration, TestFormIntegration
        
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestAppIntegration))
        suite.addTest(unittest.makeSuite(TestFormIntegration))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ Could not import integration tests: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running integration tests: {e}")
        return False

def run_api_tests():
    """Run API endpoint tests."""
    print("\n🌐 Running API Tests")
    print("=" * 50)
    
    try:
        from app import app
        import json
        
        app.config['TESTING'] = True
        client = app.test_client()
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Available indices API
        total_tests += 1
        print("Testing /api/available-indices...")
        response = client.get('/api/available-indices')
        if response.status_code == 200:
            data = json.loads(response.get_data(as_text=True))
            if data.get('success') and 'SPY' in data.get('indices', {}):
                print("  ✅ Available indices API working")
                tests_passed += 1
            else:
                print("  ❌ Available indices API returned invalid data")
        else:
            print(f"  ❌ Available indices API failed: {response.status_code}")
        
        # Test 2: Index constituents API
        total_tests += 1
        print("Testing /api/index-constituents/SPY...")
        response = client.get('/api/index-constituents/SPY?max_stocks=5')
        if response.status_code == 200:
            data = json.loads(response.get_data(as_text=True))
            if data.get('success') and data.get('tickers'):
                print(f"  ✅ Index constituents API working (got {len(data['tickers'])} tickers)")
                tests_passed += 1
            else:
                print("  ❌ Index constituents API returned no tickers")
        else:
            print(f"  ❌ Index constituents API failed: {response.status_code}")
        
        # Test 3: Portfolio page with new dropdown
        total_tests += 1
        print("Testing /portfolio page...")
        response = client.get('/portfolio')
        if response.status_code == 200:
            response_data = response.get_data(as_text=True)
            if 'Index Fund Selection' in response_data and 'indexSelect' in response_data:
                print("  ✅ Portfolio page has index dropdown")
                tests_passed += 1
            else:
                print("  ❌ Portfolio page missing index dropdown")
        else:
            print(f"  ❌ Portfolio page failed: {response.status_code}")
        
        # Test 4: Error handling
        total_tests += 1
        print("Testing error handling...")
        response = client.get('/api/index-constituents/INVALID')
        if response.status_code == 400:
            data = json.loads(response.get_data(as_text=True))
            if not data.get('success') and 'error' in data:
                print("  ✅ Error handling working correctly")
                tests_passed += 1
            else:
                print("  ❌ Error response format incorrect")
        else:
            print(f"  ❌ Expected 400 error, got: {response.status_code}")
        
        success_rate = tests_passed / total_tests if total_tests > 0 else 0
        print(f"\nAPI Tests: {tests_passed}/{total_tests} passed ({success_rate:.1%})")
        
        return tests_passed == total_tests
        
    except Exception as e:
        print(f"❌ Error running API tests: {e}")
        return False

def run_performance_tests():
    """Run basic performance tests."""
    print("\n⚡ Running Performance Tests")
    print("=" * 50)
    
    try:
        from services.index_service import IndexService
        import time
        
        index_service = IndexService()
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Index loading performance
        total_tests += 1
        print("Testing index loading performance...")
        start_time = time.time()
        
        indices = index_service.get_available_indices()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if duration < 1.0 and len(indices) >= 5:  # Should be fast and return multiple indices
            print(f"  ✅ Index loading fast ({duration:.3f}s) with {len(indices)} indices")
            tests_passed += 1
        else:
            print(f"  ❌ Index loading slow ({duration:.3f}s) or insufficient indices")
        
        # Test 2: Caching performance
        total_tests += 1
        print("Testing caching performance...")
        
        # First call (should populate cache)
        start_time = time.time()
        tickers1, _ = index_service.get_index_constituents('SPY', max_stocks=10)
        first_call_time = time.time() - start_time
        
        # Second call (should use cache)
        start_time = time.time()
        tickers2, _ = index_service.get_index_constituents('SPY', max_stocks=10)
        second_call_time = time.time() - start_time
        
        if second_call_time < first_call_time * 0.1 and tickers1 == tickers2:  # Cache should be 10x faster
            print(f"  ✅ Caching working (first: {first_call_time:.3f}s, cached: {second_call_time:.3f}s)")
            tests_passed += 1
        else:
            print(f"  ❌ Caching not effective (first: {first_call_time:.3f}s, second: {second_call_time:.3f}s)")
        
        success_rate = tests_passed / total_tests if total_tests > 0 else 0
        print(f"\nPerformance Tests: {tests_passed}/{total_tests} passed ({success_rate:.1%})")
        
        return tests_passed == total_tests
        
    except Exception as e:
        print(f"❌ Error running performance tests: {e}")
        return False

def run_data_validation_tests():
    """Run data validation and quality tests."""
    print("\n🔍 Running Data Validation Tests")
    print("=" * 50)
    
    try:
        from services.index_service import IndexService
        
        index_service = IndexService()
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Predefined constituents quality
        total_tests += 1
        print("Testing predefined constituents quality...")
        
        all_valid = True
        for index_symbol, index_data in index_service.predefined_constituents.items():
            # Check required fields
            if not all(key in index_data for key in ['name', 'description', 'tickers']):
                print(f"  ❌ {index_symbol} missing required fields")
                all_valid = False
                continue
            
            # Check ticker format
            for ticker in index_data['tickers']:
                if not isinstance(ticker, str) or len(ticker) < 1 or len(ticker) > 5:
                    print(f"  ❌ {index_symbol} has invalid ticker: {ticker}")
                    all_valid = False
                
                if ticker != ticker.upper():
                    print(f"  ❌ {index_symbol} has non-uppercase ticker: {ticker}")
                    all_valid = False
        
        if all_valid:
            print("  ✅ All predefined constituents have valid structure")
            tests_passed += 1
        
        # Test 2: Index info completeness
        total_tests += 1
        print("Testing index info completeness...")
        
        info_complete = True
        for index_symbol in index_service.predefined_constituents.keys():
            try:
                info = index_service.get_index_info(index_symbol)
                required_fields = ['symbol', 'name', 'description', 'total_constituents']
                
                if not all(field in info for field in required_fields):
                    print(f"  ❌ {index_symbol} info missing required fields")
                    info_complete = False
                
                if info['total_constituents'] <= 0:
                    print(f"  ❌ {index_symbol} has no constituents")
                    info_complete = False
                    
            except Exception as e:
                print(f"  ❌ Error getting info for {index_symbol}: {e}")
                info_complete = False
        
        if info_complete:
            print("  ✅ All index info complete")
            tests_passed += 1
        
        success_rate = tests_passed / total_tests if total_tests > 0 else 0
        print(f"\nData Validation Tests: {tests_passed}/{total_tests} passed ({success_rate:.1%})")
        
        return tests_passed == total_tests
        
    except Exception as e:
        print(f"❌ Error running data validation tests: {e}")
        return False

def main():
    """Run all test suites."""
    print("🚀 Python for Finance - Comprehensive Test Suite")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run all test suites
    test_suites = [
        ("Unit Tests", run_unit_tests),
        ("Integration Tests", run_integration_tests),
        ("API Tests", run_api_tests),
        ("Performance Tests", run_performance_tests),
        ("Data Validation Tests", run_data_validation_tests)
    ]
    
    results = []
    for suite_name, test_function in test_suites:
        try:
            result = test_function()
            results.append((suite_name, result))
        except Exception as e:
            print(f"❌ {suite_name} failed with exception: {e}")
            results.append((suite_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed_suites = 0
    total_suites = len(results)
    
    for suite_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{suite_name:.<30} {status}")
        if passed:
            passed_suites += 1
    
    print("=" * 70)
    
    success_rate = passed_suites / total_suites if total_suites > 0 else 0
    
    if passed_suites == total_suites:
        print(f"🎉 ALL TESTS PASSED! ({passed_suites}/{total_suites})")
        print("\n✅ Your index functionality is ready!")
        print("🌟 New features:")
        print("   • Index fund dropdown on portfolio page")
        print("   • Automatic constituent loading with validation")
        print("   • Error handling for invalid stocks")
        print("   • Caching for improved performance")
        print("   • Comprehensive test coverage")
        print("\n🚀 Start the application with: python run.py")
        return True
    else:
        print(f"❌ SOME TESTS FAILED ({passed_suites}/{total_suites}) - {success_rate:.1%} success rate")
        print("\n🔧 Please review the failed tests above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
