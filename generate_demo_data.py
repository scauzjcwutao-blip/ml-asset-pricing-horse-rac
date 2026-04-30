"""
generate_demo_data.py
Download a small stock panel dataset for demonstration of the ML Horse-Race Pipeline.
No WRDS access required. Suitable for quick testing and presentation.

Author: Tao Wu
Date: April 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_panel():
    """
    Generate synthetic stock panel data when yfinance is rate-limited.
    Creates realistic-looking data with 10 stocks over 10 years.
    """
    print("⚠️  yfinance rate-limited. Generating synthetic demo data...")

    np.random.seed(42)
    tickers = ['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5',
               'STOCK6', 'STOCK7', 'STOCK8', 'STOCK9', 'STOCK10']

    # Create date range (10 years of daily data)
    dates = pd.date_range(start='2015-01-01', end='2025-04-01', freq='B')  # Business days

    data_list = []
    for ticker in tickers:
        # Generate random walk price series
        n = len(dates)
        returns = np.random.normal(0.0005, 0.02, n)  # Daily returns with drift
        prices = 100 * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'Open': prices * (1 + np.random.normal(0, 0.005, n)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n)
        })
        data_list.append(df)

    df = pd.concat(data_list, ignore_index=True)

    # Sort and calculate target: next-day return
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    df['return'] = df.groupby('ticker')['Adj Close'].pct_change().shift(-1)

    # Add simple lagged features
    for lag in [1, 5, 10]:
        df[f'return_lag{lag}'] = df.groupby('ticker')['return'].shift(lag)

    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)

    return df


def download_demo_panel():
    """
    Download historical daily data for selected stocks and create a simple panel
    with lagged returns as features and next-day return as the target.
    Falls back to synthetic data if yfinance is rate-limited.
    """
    print("Downloading demo stock panel data using yfinance...")

    # Selected major stocks (small but representative panel)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']

    try:
        # Download data (auto_adjust=False to keep Adj Close column)
        data = yf.download(tickers, start="2015-01-01", end="2025-04-01",
                            group_by='ticker', auto_adjust=False, progress=False)

        # Reshape to long panel format
        df = data.stack(level=0, future_stack=True).reset_index()
        df.columns = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        # Sort and calculate target: next-day return
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        df['return'] = df.groupby('ticker')['Adj Close'].pct_change().shift(-1)

        # Add simple lagged features
        for lag in [1, 5, 10]:
            df[f'return_lag{lag}'] = df.groupby('ticker')['return'].shift(lag)

        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)

        # Check if we got any data
        if len(df) == 0:
            raise ValueError("No data downloaded from yfinance")

    except Exception as e:
        print(f"⚠️  yfinance download failed: {e}")
        df = generate_synthetic_panel()

    # Save to data folder
    output_path = Path('data/demo_stock_panel.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Demo panel data successfully saved!")
    print(f"   File path : {output_path}")
    print(f"   Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Tickers   : {df['ticker'].nunique()} ({', '.join(sorted(df['ticker'].unique()))})")

    return df


if __name__ == "__main__":
    download_demo_panel()
