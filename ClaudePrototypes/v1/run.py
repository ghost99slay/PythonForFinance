#!/usr/bin/env python3
"""
Startup script for Python for Finance Web Application
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Set debug mode based on environment
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Set host and port
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print("=" * 60)
    print("🏦 Python for Finance Web Application")
    print("=" * 60)
    print(f"🚀 Starting server on http://{host}:{port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print("=" * 60)
    print("📊 Available Tools:")
    print("   • Portfolio Theory & Efficient Frontier")
    print("   • CAPM Analysis & Beta Calculation") 
    print("   • Risk Analysis & Correlation")
    print("   • Monte Carlo Simulations")
    print("   • Regression Analysis")
    print("   • Educational Resources")
    print("=" * 60)
    print("💡 Tips:")
    print("   • Use sample tickers: AAPL,MSFT,GOOGL,TSLA")
    print("   • Try different date ranges for analysis")
    print("   • Check the Education section for theory")
    print("=" * 60)
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)
