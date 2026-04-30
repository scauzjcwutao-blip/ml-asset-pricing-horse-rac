"""
Fama-French 5 Factors Data Downloader
Automatically downloads the latest Fama-French 5 Factors (daily) from Kenneth French's data library.
"""

import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import os


def download_ff5_factors():
    """
    Download Fama-French 5 Factors (2x3) daily data from Kenneth French's website.
    
    Returns:
        pandas.DataFrame: Daily Fama-French 5 Factors (in decimal form)
    """
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    print("Downloading Fama-French 5 Factors (daily)...")
    
    start = dt.datetime(1990, 1, 1)
    end = dt.datetime.today()
    
    # ==================== 仅网络请求部分放入 try ====================
    try:
        ff5 = pdr.DataReader(
            'F-F_Research_Data_5_Factors_2x3_daily',
            'famafrench',
            start=start,
            end=end
        )[0] / 100   # Convert from percent to decimal
        
        # Critical fix: Convert PeriodIndex → TimestampIndex
        ff5.index = ff5.index.to_timestamp()
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   Please check your internet connection or try again later.")
        raise
    
    # ==================== 数据校验（与网络无关）===================
    if ff5.empty:
        raise ValueError("Downloaded Fama-French 5 Factors DataFrame is empty.")
    
    # Save to CSV
    output_path = 'data/ff5_factors.csv'
    ff5.to_csv(output_path)
    
    # Success message
    print(f"✅ Fama-French 5 Factors download completed!")
    print(f"   File saved : {output_path}")
    print(f"   Shape      : {ff5.shape} (rows × columns)")
    print(f"   Date range : {ff5.index[0].date()} to {ff5.index[-1].date()}")
    print(f"   Last date  : {ff5.index[-1].date()}")
    
    return ff5


if __name__ == "__main__":
    download_ff5_factors()
