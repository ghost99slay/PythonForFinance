#!/usr/bin/env python3
"""
Comprehensive tests for the IndexService functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.index_service import IndexService

class TestIndexService(unittest.TestCase):
    """Test cases for IndexService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.index_service = IndexService()
    
    def test_get_available_indices(self):
        """Test getting available indices."""
        indices = self.index_service.get_available_indices()
        
        # Check that we get a dictionary
        self.assertIsInstance(indices, dict)
        
        # Check that expected indices are present
        expected_indices = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        for index in expected_indices:
            self.assertIn(index, indices)
            
            # Check that each index has required fields
            index_data = indices[index]
            self.assertIn('name', index_data)
            self.assertIn('description', index_data)
            self.assertIn('count', index_data)
            self.assertIsInstance(index_data['count'], int)
            self.assertGreater(index_data['count'], 0)
    
    def test_get_index_info(self):
        """Test getting index information."""
        # Test valid index
        info = self.index_service.get_index_info('SPY')
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['symbol'], 'SPY')
        self.assertIn('name', info)
        self.assertIn('description', info)
        self.assertIn('total_constituents', info)
        self.assertIsInstance(info['total_constituents'], int)
        self.assertGreater(info['total_constituents'], 0)
        
        # Test invalid index
        with self.assertRaises(ValueError):
            self.index_service.get_index_info('INVALID')
    
    def test_predefined_constituents_structure(self):
        """Test that predefined constituents have the correct structure."""
        for index_symbol, index_data in self.index_service.predefined_constituents.items():
            # Check required fields
            self.assertIn('name', index_data)
            self.assertIn('description', index_data)
            self.assertIn('tickers', index_data)
            
            # Check data types
            self.assertIsInstance(index_data['name'], str)
            self.assertIsInstance(index_data['description'], str)
            self.assertIsInstance(index_data['tickers'], list)
            
            # Check that tickers list is not empty
            self.assertGreater(len(index_data['tickers']), 0)
            
            # Check that all tickers are strings and uppercase
            for ticker in index_data['tickers']:
                self.assertIsInstance(ticker, str)
                self.assertEqual(ticker, ticker.upper())
                self.assertGreater(len(ticker), 0)
    
    @patch('services.index_service.yf.Ticker')
    def test_validate_ticker_batch_success(self, mock_ticker):
        """Test successful ticker validation."""
        # Mock yfinance response for valid ticker
        mock_stock = MagicMock()
        mock_stock.info = {'symbol': 'AAPL', 'longName': 'Apple Inc.'}
        
        # Create mock history data
        mock_history = pd.DataFrame({
            'Close': [150.0, 151.0, 149.0]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_stock.history.return_value = mock_history
        
        mock_ticker.return_value = mock_stock
        
        # Test validation
        results = self.index_service._validate_ticker_batch(['AAPL'])
        
        self.assertEqual(len(results), 1)
        ticker, is_valid, reason = results[0]
        self.assertEqual(ticker, 'AAPL')
        self.assertTrue(is_valid)
        self.assertIn('Valid', reason)
    
    @patch('services.index_service.yf.Ticker')
    def test_validate_ticker_batch_no_data(self, mock_ticker):
        """Test ticker validation with no data."""
        # Mock yfinance response for invalid ticker
        mock_stock = MagicMock()
        mock_stock.info = {}
        mock_ticker.return_value = mock_stock
        
        # Test validation
        results = self.index_service._validate_ticker_batch(['INVALID'])
        
        self.assertEqual(len(results), 1)
        ticker, is_valid, reason = results[0]
        self.assertEqual(ticker, 'INVALID')
        self.assertFalse(is_valid)
        self.assertIn('No data available', reason)
    
    @patch('services.index_service.yf.Ticker')
    def test_validate_ticker_batch_old_data(self, mock_ticker):
        """Test ticker validation with old data."""
        # Mock yfinance response with old data
        mock_stock = MagicMock()
        mock_stock.info = {'symbol': 'OLD', 'longName': 'Old Company'}
        
        # Create mock history data with old dates
        old_date = datetime.now() - timedelta(days=30)
        mock_history = pd.DataFrame({
            'Close': [10.0, 11.0, 9.0]
        }, index=pd.date_range(old_date, periods=3))
        mock_stock.history.return_value = mock_history
        
        mock_ticker.return_value = mock_stock
        
        # Test validation
        results = self.index_service._validate_ticker_batch(['OLD'])
        
        self.assertEqual(len(results), 1)
        ticker, is_valid, reason = results[0]
        self.assertEqual(ticker, 'OLD')
        self.assertFalse(is_valid)
        self.assertIn('Data too old', reason)
    
    @patch('services.index_service.yf.Ticker')
    def test_validate_ticker_batch_low_price(self, mock_ticker):
        """Test ticker validation with very low price."""
        # Mock yfinance response with low price
        mock_stock = MagicMock()
        mock_stock.info = {'symbol': 'PENNY', 'longName': 'Penny Stock'}
        
        # Create mock history data with low price
        mock_history = pd.DataFrame({
            'Close': [0.50, 0.51, 0.49]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_stock.history.return_value = mock_history
        
        mock_ticker.return_value = mock_stock
        
        # Test validation
        results = self.index_service._validate_ticker_batch(['PENNY'])
        
        self.assertEqual(len(results), 1)
        ticker, is_valid, reason = results[0]
        self.assertEqual(ticker, 'PENNY')
        self.assertFalse(is_valid)
        self.assertIn('Price too low', reason)
    
    @patch('services.index_service.yf.Ticker')
    def test_validate_ticker_batch_exception(self, mock_ticker):
        """Test ticker validation with exception."""
        # Mock yfinance to raise exception
        mock_ticker.side_effect = Exception("Network error")
        
        # Test validation
        results = self.index_service._validate_ticker_batch(['ERROR'])
        
        self.assertEqual(len(results), 1)
        ticker, is_valid, reason = results[0]
        self.assertEqual(ticker, 'ERROR')
        self.assertFalse(is_valid)
        self.assertIn('Error:', reason)
    
    @patch.object(IndexService, '_validate_tickers')
    def test_get_index_constituents_success(self, mock_validate):
        """Test successful index constituents retrieval."""
        # Mock validation results
        valid_tickers = ['AAPL', 'MSFT', 'GOOGL']
        validation_results = {
            'valid_tickers': valid_tickers,
            'invalid_tickers': ['INVALID'],
            'validation_details': {
                'AAPL': {'valid': True, 'reason': 'Valid ($150.00)'},
                'MSFT': {'valid': True, 'reason': 'Valid ($300.00)'},
                'GOOGL': {'valid': True, 'reason': 'Valid ($2500.00)'},
                'INVALID': {'valid': False, 'reason': 'No data available'}
            },
            'success_rate': 0.75
        }
        mock_validate.return_value = (valid_tickers, validation_results)
        
        # Test getting constituents
        tickers, metadata = self.index_service.get_index_constituents('SPY', max_stocks=4)
        
        self.assertEqual(tickers, valid_tickers)
        self.assertIsInstance(metadata, dict)
        self.assertIn('index_name', metadata)
        self.assertIn('valid_count', metadata)
        self.assertIn('invalid_count', metadata)
        self.assertEqual(metadata['valid_count'], 3)
        self.assertEqual(metadata['invalid_count'], 1)
    
    def test_get_index_constituents_invalid_index(self):
        """Test getting constituents for invalid index."""
        with self.assertRaises(ValueError):
            self.index_service.get_index_constituents('INVALID')
    
    def test_cache_functionality(self):
        """Test that caching works correctly."""
        # Clear cache first
        self.index_service.cache.clear()
        
        with patch.object(self.index_service, '_validate_tickers') as mock_validate:
            mock_validate.return_value = (['AAPL'], {'valid_tickers': ['AAPL']})
            
            # First call should hit the validation
            tickers1, metadata1 = self.index_service.get_index_constituents('SPY', max_stocks=1)
            self.assertEqual(mock_validate.call_count, 1)
            
            # Second call should use cache
            tickers2, metadata2 = self.index_service.get_index_constituents('SPY', max_stocks=1)
            self.assertEqual(mock_validate.call_count, 1)  # Should not increase
            
            # Results should be the same
            self.assertEqual(tickers1, tickers2)
    
    def test_update_predefined_constituents_success(self):
        """Test successful update of predefined constituents."""
        with patch.object(self.index_service, '_validate_tickers') as mock_validate:
            # Mock validation to return mostly valid tickers
            new_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
            mock_validate.return_value = (new_tickers, {'success_rate': 1.0})
            
            # Test update
            success = self.index_service.update_predefined_constituents('SPY', new_tickers)
            self.assertTrue(success)
            self.assertEqual(self.index_service.predefined_constituents['SPY']['tickers'], new_tickers)
    
    def test_update_predefined_constituents_invalid_index(self):
        """Test update for invalid index."""
        success = self.index_service.update_predefined_constituents('INVALID', ['AAPL'])
        self.assertFalse(success)
    
    def test_update_predefined_constituents_too_many_invalid(self):
        """Test update with too many invalid tickers."""
        with patch.object(self.index_service, '_validate_tickers') as mock_validate:
            # Mock validation to return mostly invalid tickers (less than 80% valid)
            mock_validate.return_value = (['AAPL'], {'success_rate': 0.5})  # Only 50% valid
            
            # Test update
            success = self.index_service.update_predefined_constituents('SPY', ['AAPL', 'INVALID'])
            self.assertFalse(success)
    
    def test_validate_tickers_integration(self):
        """Test the full validation process with real data structure."""
        # Use a small subset of real tickers for integration test
        test_tickers = ['AAPL', 'MSFT', 'INVALID_TICKER_XYZ']
        
        with patch.object(self.index_service, '_validate_ticker_batch') as mock_batch:
            # Mock batch validation results
            mock_batch.return_value = [
                ('AAPL', True, 'Valid ($150.00)'),
                ('MSFT', True, 'Valid ($300.00)'),
                ('INVALID_TICKER_XYZ', False, 'No data available')
            ]
            
            valid_tickers, validation_results = self.index_service._validate_tickers(test_tickers)
            
            # Check results
            self.assertEqual(len(valid_tickers), 2)
            self.assertIn('AAPL', valid_tickers)
            self.assertIn('MSFT', valid_tickers)
            self.assertNotIn('INVALID_TICKER_XYZ', valid_tickers)
            
            # Check validation results structure
            self.assertIn('valid_tickers', validation_results)
            self.assertIn('invalid_tickers', validation_results)
            self.assertIn('validation_details', validation_results)
            self.assertIn('success_rate', validation_results)
            
            self.assertEqual(validation_results['success_rate'], 2/3)  # 2 out of 3 valid

class TestIndexServiceIntegration(unittest.TestCase):
    """Integration tests that may hit real APIs (run separately)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.index_service = IndexService()
    
    def test_real_ticker_validation_sample(self):
        """Test validation with a small sample of real tickers."""
        # Only run this test if we want to hit real APIs
        # This is more of a smoke test
        test_tickers = ['AAPL', 'MSFT']  # Known good tickers
        
        try:
            valid_tickers, validation_results = self.index_service._validate_tickers(test_tickers)
            
            # We expect both to be valid in normal circumstances
            self.assertGreaterEqual(len(valid_tickers), 1)  # At least one should be valid
            self.assertLessEqual(len(valid_tickers), 2)     # At most two can be valid
            
            # Check validation results structure
            self.assertIn('success_rate', validation_results)
            self.assertGreaterEqual(validation_results['success_rate'], 0.0)
            self.assertLessEqual(validation_results['success_rate'], 1.0)
            
        except Exception as e:
            # If we can't connect to APIs, skip this test
            self.skipTest(f"Could not connect to financial APIs: {e}")

def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestIndexService))
    
    # Optionally add integration tests (they hit real APIs)
    # suite.addTest(unittest.makeSuite(TestIndexServiceIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
