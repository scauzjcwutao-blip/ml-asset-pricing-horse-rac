"""
Unobserved Performance of Hedge Funds (Weigert et al., Journal of Finance 2024)
Python Replication Script
Author: Tao Wu
Date: April 2026

This script replicates the key steps:
1. Construct buy-and-hold portfolio using 13F holdings
2. Calculate Unobserved Performance (UP)
3. Form UP 5-1 long-short portfolios
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Replicating Weigert et al. (JF 2024) - Unobserved Performance of Hedge Funds")
print("=" * 80)

# ========================== 1. Load Demo Data ==========================
def load_demo_data():
    """For demonstration only. Replace with real TASS/HFR + 13F data in production."""
    print("Loading demonstration data...")

    # Simulate monthly fund returns (1994-2019)
    dates = pd.date_range(start='1994-01-01', end='2019-12-31', freq='M')
    fund_returns = pd.Series(
        np.random.normal(0.008, 0.04, len(dates)),
        index=dates,
        name='fund_ret'
    )

    # Simulate quarterly 13F holdings (multiple stocks per quarter)
    quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='Q')
    holdings_list = []
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']

    for q in quarters:
        n_stocks = np.random.randint(8, 20)  # 每季度持有 8-20 只股票
        selected_stocks = np.random.choice(stocks, n_stocks, replace=False)
        weights = np.random.dirichlet(np.ones(n_stocks))
        for stock, weight in zip(selected_stocks, weights):
            holdings_list.append({
                'report_date': q,
                'stock': stock,
                'weight': weight
            })

    holdings = pd.DataFrame(holdings_list)
    print(f"✅ Fund returns: {fund_returns.shape}")
    print(f"✅ 13F holdings: {holdings.shape} ({len(quarters)} quarters)")
    return fund_returns, holdings


# ========================== 2. Construct Buy-and-Hold Portfolio ==========================
def construct_buy_and_hold_portfolio(holdings: pd.DataFrame, quarter_end: pd.Timestamp):
    """Build virtual buy-and-hold portfolio from previous quarter's 13F."""
    # Use previous quarter's report
    prev_quarter = holdings[holdings['report_date'] == quarter_end]
    if prev_quarter.empty:
        return pd.Series(dtype=float)

    prev_quarter = prev_quarter.copy()
    total_weight = prev_quarter['weight'].sum()
    prev_quarter['weight'] = prev_quarter['weight'] / total_weight

    return prev_quarter.set_index('stock')['weight']


# ========================== 3. Calculate Unobserved Performance (UP) ==========================
def calculate_unobserved_performance(fund_returns: pd.Series, holdings: pd.DataFrame):
    """UP_t = Fund Reported Return_t - Virtual Buy-and-Hold Return_t"""
    up_list = []
    quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='Q')

    for q in quarters:
        # Get reported fund return (nearest monthly return)
        reported_ret = fund_returns.get(q, np.nan)
        if pd.isna(reported_ret):
            reported_ret = fund_returns.asof(q)

        bh_weights = construct_buy_and_hold_portfolio(holdings, q)

        if not bh_weights.empty:
            # In real version: multiply by actual stock returns of that quarter
            virtual_ret = np.random.normal(0.006, 0.03)   # placeholder
            up = reported_ret - virtual_ret
        else:
            up = np.nan

        up_list.append({'quarter': q, 'UP': up})

    up_df = pd.DataFrame(up_list).set_index('quarter')
    print(f"✅ Calculated Unobserved Performance for {len(up_df)} quarters.")
    return up_df


# ========================== 4. Form UP 5-1 Long-Short Portfolio ==========================
def form_up_long_short(up_df: pd.DataFrame):
    """Sort into UP quintiles and form 5-1 long-short portfolio each quarter."""
    up_df = up_df.dropna().copy()
    up_df['quintile'] = up_df.groupby(up_df.index)['UP'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
    )

    # Average UP per quintile per quarter
    portfolio_returns = up_df.groupby(['quarter', 'quintile'])['UP'].mean()

    # 5-1 Long-Short
    long = portfolio_returns.xs(5, level='quintile')
    short = portfolio_returns.xs(1, level='quintile')
    long_short = long - short

    print(f"\n📊 UP 5-1 Long-Short Portfolio:")
    print(f"   Mean monthly return : {long_short.mean():.4f}")
    print(f"   Annualized          : {long_short.mean()*12*100:.2f}%")
    print(f"   Number of quarters  : {len(long_short)}")
    print(f"   t-stat (simple)     : {long_short.mean() / (long_short.std() / np.sqrt(len(long_short))):.2f}")

    return long_short


# ========================== Main Execution ==========================
if __name__ == "__main__":
    fund_returns, holdings = load_demo_data()
    
    up_df = calculate_unobserved_performance(fund_returns, holdings)
    
    long_short_returns = form_up_long_short(up_df)

    # Save results
    Path("data/results").mkdir(parents=True, exist_ok=True)
    long_short_returns.to_csv("data/results/up_5_1_long_short.csv")
    up_df.to_csv("data/results/unobserved_performance.csv")

    print("\n" + "="*80)
    print("🎉 Replication completed successfully!")
    print("💡 Replace demo data with real 13F holdings + fund returns for full replication.")
    print("📁 Results saved to data/results/")
    print("="*80)
