#!/usr/bin/env python3
"""
Integration tests for the Flask application with index functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from services.index_service import IndexService

class TestAppIntegration(unittest.TestCase):
    """Test Flask app integration with index functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
        self.client = self.app.test_client()
        
        # Create a test application context
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Clean up after tests."""
        self.app_context.pop()
    
    def test_portfolio_page_loads(self):
        """Test that the portfolio page loads with the new dropdown."""
        response = self.client.get('/portfolio')
        self.assertEqual(response.status_code, 200)
        
        # Check that the response contains the new index dropdown
        response_data = response.get_data(as_text=True)
        self.assertIn('Index Fund Selection', response_data)
        self.assertIn('indexSelect', response_data)
        self.assertIn('SPY', response_data)
        self.assertIn('QQQ', response_data)
    
    def test_available_indices_api(self):
        """Test the available indices API endpoint."""
        response = self.client.get('/api/available-indices')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.get_data(as_text=True))
        self.assertTrue(data['success'])
        self.assertIn('indices', data)
        
        indices = data['indices']
        self.assertIn('SPY', indices)
        self.assertIn('QQQ', indices)
        
        # Check structure of index data
        spy_data = indices['SPY']
        self.assertIn('name', spy_data)
        self.assertIn('description', spy_data)
        self.assertIn('count', spy_data)
        self.assertIsInstance(spy_data['count'], int)
    
    @patch('services.index_service.IndexService.get_index_constituents')
    def test_index_constituents_api_success(self, mock_get_constituents):
        """Test the index constituents API endpoint with successful response."""
        # Mock the service response
        mock_tickers = ['AAPL', 'MSFT', 'GOOGL']
        mock_metadata = {
            'index_name': 'SPDR S&P 500 ETF',
            'valid_count': 3,
            'invalid_count': 0,
            'requested_count': 3
        }
        mock_get_constituents.return_value = (mock_tickers, mock_metadata)
        
        # Make API request
        response = self.client.get('/api/index-constituents/SPY?max_stocks=3')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.get_data(as_text=True))
        self.assertTrue(data['success'])
        self.assertEqual(data['tickers'], mock_tickers)
        self.assertEqual(data['metadata'], mock_metadata)
        self.assertEqual(data['tickers_string'], 'AAPL,MSFT,GOOGL')
        
        # Verify the service was called correctly
        mock_get_constituents.assert_called_once_with('SPY', 3)
    
    @patch('services.index_service.IndexService.get_index_constituents')
    def test_index_constituents_api_error(self, mock_get_constituents):
        """Test the index constituents API endpoint with error response."""
        # Mock the service to raise an exception
        mock_get_constituents.side_effect = ValueError("Index not supported")
        
        # Make API request
        response = self.client.get('/api/index-constituents/INVALID')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.get_data(as_text=True))
        self.assertFalse(data['success'])
        self.assertIn('error', data)
        self.assertEqual(data['tickers'], [])
        self.assertEqual(data['metadata'], {})
        self.assertEqual(data['tickers_string'], '')
    
    def test_index_constituents_api_default_max_stocks(self):
        """Test that the API uses default max_stocks when not specified."""
        with patch('services.index_service.IndexService.get_index_constituents') as mock_get_constituents:
            mock_get_constituents.return_value = ([], {})
            
            # Make API request without max_stocks parameter
            response = self.client.get('/api/index-constituents/SPY')
            
            # Verify the service was called with default value
            mock_get_constituents.assert_called_once_with('SPY', 50)
    
    def test_index_constituents_api_custom_max_stocks(self):
        """Test that the API respects custom max_stocks parameter."""
        with patch('services.index_service.IndexService.get_index_constituents') as mock_get_constituents:
            mock_get_constituents.return_value = ([], {})
            
            # Make API request with custom max_stocks
            response = self.client.get('/api/index-constituents/QQQ?max_stocks=25')
            
            # Verify the service was called with custom value
            mock_get_constituents.assert_called_once_with('QQQ', 25)
    
    @patch('services.portfolio_service.PortfolioService.analyze_portfolio')
    def test_portfolio_form_submission_with_index(self, mock_analyze):
        """Test portfolio form submission with index selection."""
        # Mock the portfolio analysis
        mock_results = {
            'tickers': ['AAPL', 'MSFT'],
            'analysis_period': {'start_date': '2020-01-01', 'end_date': '2023-01-01'},
            'expected_returns': {'AAPL': 0.15, 'MSFT': 0.12},
            'optimal_portfolios': {
                'maximum_sharpe': {'sharpe_ratio': 1.5, 'volatility': 0.2, 'expected_return': 0.13},
                'minimum_volatility': {'volatility': 0.15, 'expected_return': 0.11}
            },
            'insights': ['Test insight']
        }
        mock_analyze.return_value = mock_results
        
        # Submit form data
        form_data = {
            'index_selection': 'SPY',
            'tickers': 'AAPL,MSFT',
            'start_date': '2020-01-01',
            'end_date': '2023-01-01'
        }
        
        response = self.client.post('/portfolio', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # Check that the analysis was called
        mock_analyze.assert_called_once()
    
    def test_portfolio_form_validation(self):
        """Test portfolio form validation."""
        # Submit form with missing required data
        form_data = {
            'tickers': '',  # Empty tickers should fail validation
            'start_date': '2020-01-01',
            'end_date': '2023-01-01'
        }
        
        response = self.client.post('/portfolio', data=form_data)
        self.assertEqual(response.status_code, 200)
        
        # Check that validation error is shown
        response_data = response.get_data(as_text=True)
        # The form should be re-rendered with validation errors
        self.assertIn('form', response_data)
    
    def test_api_endpoints_case_insensitive(self):
        """Test that API endpoints handle case-insensitive ticker symbols."""
        with patch('services.index_service.IndexService.get_index_constituents') as mock_get_constituents:
            mock_get_constituents.return_value = (['AAPL'], {'valid_count': 1})
            
            # Test lowercase
            response = self.client.get('/api/index-constituents/spy')
            self.assertEqual(response.status_code, 200)
            mock_get_constituents.assert_called_with('SPY', 50)
            
            # Reset mock
            mock_get_constituents.reset_mock()
            
            # Test mixed case
            response = self.client.get('/api/index-constituents/SpY')
            self.assertEqual(response.status_code, 200)
            mock_get_constituents.assert_called_with('SPY', 50)
    
    def test_error_handling_in_views(self):
        """Test error handling in view functions."""
        # Test with an endpoint that might fail
        with patch('services.index_service.IndexService.get_available_indices') as mock_get_indices:
            mock_get_indices.side_effect = Exception("Database error")
            
            response = self.client.get('/api/available-indices')
            self.assertEqual(response.status_code, 400)
            
            data = json.loads(response.get_data(as_text=True))
            self.assertFalse(data['success'])
            self.assertIn('error', data)

class TestFormIntegration(unittest.TestCase):
    """Test form integration with the new index functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def tearDown(self):
        """Clean up after tests."""
        self.app_context.pop()
    
    def test_portfolio_form_has_index_field(self):
        """Test that PortfolioForm has the new index selection field."""
        from app import PortfolioForm
        
        form = PortfolioForm()
        
        # Check that the form has the new field
        self.assertTrue(hasattr(form, 'index_selection'))
        
        # Check that the field has the expected choices
        choices = form.index_selection.choices
        choice_values = [choice[0] for choice in choices]
        
        self.assertIn('', choice_values)  # Empty option
        self.assertIn('SPY', choice_values)
        self.assertIn('QQQ', choice_values)
        self.assertIn('DIA', choice_values)
        self.assertIn('IWM', choice_values)
        self.assertIn('VTI', choice_values)
    
    def test_form_validation_with_index_selection(self):
        """Test form validation when index is selected."""
        from app import PortfolioForm
        
        # Test with index selected but empty tickers (should still be valid if tickers get populated)
        form_data = {
            'index_selection': 'SPY',
            'tickers': 'AAPL,MSFT',  # Some tickers are required
            'start_date': '2020-01-01',
            'end_date': '2023-01-01'
        }
        
        with self.app.test_request_context(method='POST', data=form_data):
            form = PortfolioForm()
            # The form should be valid
            self.assertTrue(form.validate())
    
    def test_form_validation_without_index_selection(self):
        """Test form validation when no index is selected."""
        from app import PortfolioForm
        
        # Test with manual tickers
        form_data = {
            'index_selection': '',
            'tickers': 'AAPL,MSFT,GOOGL',
            'start_date': '2020-01-01',
            'end_date': '2023-01-01'
        }
        
        with self.app.test_request_context(method='POST', data=form_data):
            form = PortfolioForm()
            self.assertTrue(form.validate())

def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAppIntegration))
    suite.addTest(unittest.makeSuite(TestFormIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
