"""
Market Structure Analysis Main Script

This script demonstrates the use of the market structure analysis tools
on a diversified set of financial assets.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import analysis pipeline
from analysis_pipeline import run_market_structure_analysis


def main():
    """
    Main function to execute the full market structure analysis
    """
    print("Market Structure Analysis Using Computational Geometry")
    print("======================================================")
    
    # Define parameters
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    # Define assets to analyze
    # Using a selection of ETFs representing different asset classes and sectors
    tickers = [
        # US Indices
        'SPY',    # S&P 500
        'QQQ',    # Nasdaq 100
        'IWM',    # Russell 2000
        'DIA',    # Dow Jones Industrial Average
        
        # US Sectors
        'XLK',    # Technology
        'XLF',    # Financials
        'XLE',    # Energy
        'XLV',    # Healthcare
        'XLP',    # Consumer Staples
        'XLY',    # Consumer Discretionary
        
        # International
        'EFA',    # Developed Markets
        'EEM',    # Emerging Markets
        'EWJ',    # Japan
        'EWG',    # Germany
        'INDA',   # India
        
        # Fixed Income
        'TLT',    # Long-Term Treasury
        'IEF',    # Intermediate Treasury
        'LQD',    # Corporate Bonds
        'HYG',    # High Yield Bonds
        'MUB',    # Municipal Bonds
        
        # Commodities
        'GLD',    # Gold
        'SLV',    # Silver
        'USO',    # Oil
        'DBC',    # Commodities Basket
        
        # Other
        'VNQ',    # Real Estate
        'XLRE',   # Real Estate
        'XLU',    # Utilities
    ]
    
    # Create output directory
    output_dir = 'market_analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the analysis
    results = run_market_structure_analysis(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    print(f"\nResults saved to {output_dir}/")
    
    # Return results for further analysis if needed
    return results


if __name__ == "__main__":
    main()