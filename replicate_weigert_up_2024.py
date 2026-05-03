"""
Unobserved Performance of Hedge Funds (Weigert et al., Journal of Finance 2024)
Python Replication Script - Real Data Ready Version (Multi-Fund Support)
Author: Tao Wu
Date: May 2026

This script supports real WRDS data + interactive CSV loading with multiple funds.
"""

# =============================================================================
# CSV FILE FORMAT REQUIREMENTS (for real WRDS data)
# =============================================================================
"""
1. fund_returns.csv          ← NOW SUPPORTS MULTIPLE FUNDS
   - Columns: 'date' (index or column), 'fund_id', 'fund_ret' (decimal)
   Example:
       date       | fund_id     | fund_ret
       1994-01-31 | FUND001     | 0.0123
       1994-01-31 | FUND002     | -0.0056

2. holdings.csv
   - Columns: 'report_date', 'stock', 'weight'

3. stock_returns.csv
   - Index: date (quarter-end)
   - Columns: stock tickers with quarterly returns

4. factors.csv
   - Index: date (quarter-end)
   - Columns: 'MKT', 'SMB', 'HML', 'MOM'
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Replicating Weigert et al. (JF 2024) - Unobserved Performance of Hedge Funds")
print("Multi-Fund Support Enabled")
print("=" * 80)


# ========================== 1. Load Data (Interactive + Real CSV Support) ==========================
def load_data():
    """Interactive data loader with multi-fund support."""
    print("\n🔄 Data Loading Mode")
    print("   Press Enter to use demo data, or enter the full path to your CSV files (WRDS format).")

    # Fund returns (multi-fund support)
    fund_path = input("   Enter path to fund_returns.csv (or press Enter for demo): ").strip()
    if fund_path:
        fund_df = pd.read_csv(fund_path, parse_dates=['date'])
        fund_df = fund_df.set_index('date')
        available_funds = fund_df['fund_id'].unique()
        print(f"✅ Loaded real fund_returns for {len(available_funds)} funds: {available_funds.tolist()}")
        
        # Let user select fund
        print(f"   Available Fund IDs: {available_funds.tolist()}")
        selected_fund = input("   Enter Fund ID (or press Enter for the first one): ").strip()
        if not selected_fund:
            selected_fund = available_funds[0]
        fund_returns = fund_df[fund_df['fund_id'] == selected_fund]['fund_ret']
        print(f"✅ Selected Fund ID: {selected_fund} ({fund_returns.shape[0]} observations)")
    else:
        # Demo data (single fund)
        dates = pd.date_range(start='1994-01-01', end='2019-12-31', freq='ME')
        fund_returns = pd.Series(np.random.normal(0.008, 0.04, len(dates)), index=dates, name='fund_ret')
        print("✅ Using demo fund_returns (single fund)")

    # 13F Holdings
    holdings_path = input("   Enter path to holdings.csv (or press Enter for demo): ").strip()
    if holdings_path:
        holdings = pd.read_csv(holdings_path, parse_dates=['report_date'])
        print(f"✅ Loaded real holdings: {holdings.shape}")
    else:
        # Demo holdings (unchanged)
        quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='QE')
        holdings_list = []
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        for q in quarters:
            n_stocks = np.random.randint(8, 20)
            selected_stocks = np.random.choice(stocks, n_stocks, replace=False)
            weights = np.random.dirichlet(np.ones(n_stocks))
            for stock, weight in zip(selected_stocks, weights):
                holdings_list.append({'report_date': q, 'stock': stock, 'weight': weight})
        holdings = pd.DataFrame(holdings_list)
        print("✅ Using demo holdings")

    # Stock returns & Factors (unchanged)
    stock_path = input("   Enter path to stock_returns.csv (or press Enter for demo): ").strip()
    if stock_path:
        stock_returns = pd.read_csv(stock_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded real stock_returns: {stock_returns.shape}")
    else:
        quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='QE')
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
        stock_returns = pd.DataFrame(
            np.random.normal(0.006, 0.03, size=(len(quarters), len(stocks))),
            index=quarters,
            columns=stocks
        )
        print("✅ Using demo stock_returns")

    factor_path = input("   Enter path to factors.csv (or press Enter for demo): ").strip()
    if factor_path:
        factors = pd.read_csv(factor_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded real factors: {factors.shape}")
    else:
        quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='QE')
        factors = pd.DataFrame({
            'MKT': np.random.normal(0.007, 0.04, len(quarters)),
            'SMB': np.random.normal(0.003, 0.03, len(quarters)),
            'HML': np.random.normal(0.002, 0.03, len(quarters)),
            'MOM': np.random.normal(0.004, 0.03, len(quarters))
        }, index=quarters)
        print("✅ Using demo factors")

    return fund_returns, holdings, stock_returns, factors
    
# ========================== 2. Construct Lagged Buy-and-Hold Portfolio ==========================
def construct_buy_and_hold_portfolio(holdings: pd.DataFrame, quarter_end: pd.Timestamp):
    """Build virtual buy-and-hold portfolio using PREVIOUS quarter's 13F holdings."""
    prev_reports = holdings[holdings['report_date'] < quarter_end]
    if prev_reports.empty:
        return pd.Series(dtype=float)

    prev_quarter = prev_reports['report_date'].max()
    prev_holdings = holdings[holdings['report_date'] == prev_quarter].copy()

    total_weight = prev_holdings['weight'].sum()
    prev_holdings['weight'] = prev_holdings['weight'] / total_weight

    return prev_holdings.set_index('stock')['weight']
# ========================== 3. Calculate Unobserved Performance (UP) ==========================
def calculate_unobserved_performance(fund_returns: pd.Series, 
                                    holdings: pd.DataFrame, 
                                    stock_returns: pd.DataFrame):
    """UP_t = Fund Reported Return_t - Virtual Buy-and-Hold Return_t"""
    up_list = []
    quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='QE')

    for q in quarters:
        reported_ret = fund_returns.asof(q)
        bh_weights = construct_buy_and_hold_portfolio(holdings, q)

        if not bh_weights.empty:
            stock_ret_q = stock_returns.loc[stock_returns.index == q]
            if not stock_ret_q.empty:
                common_stocks = bh_weights.index.intersection(stock_ret_q.columns)
                if len(common_stocks) > 0:
                    virtual_ret = (bh_weights.loc[common_stocks] * stock_ret_q[common_stocks].iloc[0]).sum()
                else:
                    virtual_ret = np.nan
            else:
                virtual_ret = np.nan
        else:
            virtual_ret = np.nan

        up = reported_ret - virtual_ret if not pd.isna(virtual_ret) else np.nan
        up_list.append({'quarter': q, 'UP': up})

    up_df = pd.DataFrame(up_list).set_index('quarter')
    print(f"✅ Calculated Unobserved Performance for {len(up_df.dropna())} valid quarters.")
    return up_df
 # ========================== 4. Form UP 5-1 Long-Short Portfolio ==========================
def form_up_long_short(up_df: pd.DataFrame):
    """Sort into UP quintiles and form 5-1 long-short portfolio each quarter."""
    up_df = up_df.dropna().copy()
    up_df['quintile'] = up_df.groupby(up_df.index)['UP'].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') + 1
    )

    portfolio_returns = up_df.groupby(['quarter', 'quintile'])['UP'].mean()

    try:
        long = portfolio_returns.xs(5, level='quintile')
        short = portfolio_returns.xs(1, level='quintile')
        long_short = long - short
    except KeyError:
        long_short = pd.Series(dtype=float)

    print(f"\n📊 UP 5-1 Long-Short Portfolio (Raw):")
    print(f"   Mean quarterly return : {long_short.mean():.4f}")
    print(f"   Annualized            : {long_short.mean() * 4 * 100:.2f}%")
    print(f"   Number of quarters    : {len(long_short)}")

    return long_short
# ========================== 5. Risk-Adjusted Alpha (Carhart 4-Factor) ==========================
def compute_risk_adjusted_alpha(ls_returns: pd.Series, factor_returns: pd.DataFrame):
    """Compute Carhart 4-factor alpha + t-stat for the long-short portfolio."""
    df = pd.concat([ls_returns.rename('LS'), factor_returns], axis=1).dropna()
    
    if len(df) < 10:
        return {"alpha": np.nan, "t_stat": np.nan, "n_obs": len(df)}

    X = sm.add_constant(df[['MKT', 'SMB', 'HML', 'MOM']])
    y = df['LS']
    model = sm.OLS(y, X).fit()
    
    alpha = model.params['const']
    t_stat = model.tvalues['const']
    
    print(f"\n📊 Risk-Adjusted Alpha (Carhart 4-Factor):")
    print(f"   Alpha (quarterly)     : {alpha:.4f}")
    print(f"   t-stat                : {t_stat:.2f}")
    print(f"   Observations          : {len(df)}")
    
    return {"alpha": alpha, "t_stat": t_stat, "n_obs": len(df)}                                       


# ========================== Main Execution ==========================
if __name__ == "__main__":
    fund_returns, holdings, stock_returns, factors = load_data()
    
    up_df = calculate_unobserved_performance(fund_returns, holdings, stock_returns)
    
    long_short_returns = form_up_long_short(up_df)
    
    compute_risk_adjusted_alpha(long_short_returns, factors)

    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    long_short_returns.to_csv(output_dir / "up_5_1_long_short.csv")
    up_df.to_csv(output_dir / "unobserved_performance.csv")

    print("\n" + "="*80)
    print("🎉 Replication completed successfully!")
    print("📁 Results saved to data/results/")
    print("🔄 You can now replace the demo data with real WRDS data for the full replication.")
    print("="*80)
