"""
Unobserved Performance of Hedge Funds (Weigert et al., Journal of Finance 2024)
Python Replication Script - Corrected Full Version
Author: Tao Wu
Date: May 2026

Key Features:
  • Multi-fund panel data processing
  • Monthly → Quarterly compounding
  • Cross-sectional quintile sorting each quarter
  • Predictive sort: t-period UP → t+1 period fund returns
  • Newey-West (HAC) standard errors
  • Demo + Real WRDS data support
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

np.random.seed(42)

print("=" * 80)
print("Replicating Weigert et al. (JF 2024) - Unobserved Performance of Hedge Funds")
print("Corrected Full Version: Multi-Fund Panel + Cross-Sectional Quintile Sort")
print("=" * 80)


# =============================================================================
# CSV FILE FORMAT REQUIREMENTS
# =============================================================================
"""
1. fund_returns.csv
   - Columns: 'date', 'fund_id', 'fund_ret' (monthly decimal return)

2. holdings.csv
   - Columns: 'report_date', 'fund_id', 'stock', 'weight'

3. stock_returns.csv
   - Columns: 'date', 'stock', 'ret' (monthly decimal return)

4. factors.csv
   - Columns: 'date', 'MKT', 'SMB', 'HML', 'MOM' (monthly)
"""


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def generate_demo_data(n_funds=100, n_stocks=50, start='1994-01-31', end='2019-12-31'):
    """Generate realistic demo data for testing."""
    print(f"\n📊 Generating demo data: {n_funds} funds, {n_stocks} stocks")
    months = pd.date_range(start=start, end=end, freq='ME')
    quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='QE')
    fund_ids = [f"FUND{i:04d}" for i in range(1, n_funds + 1)]
    stock_ids = [f"STK{i:03d}" for i in range(1, n_stocks + 1)]

    # Fund monthly returns
    fund_records = []
    for fid in fund_ids:
        alpha = np.random.normal(0.005, 0.003)
        vol = np.random.uniform(0.02, 0.06)
        rets = np.random.normal(alpha, vol, len(months))
        for date, ret in zip(months, rets):
            fund_records.append({'date': date, 'fund_id': fid, 'fund_ret': ret})
    fund_returns_df = pd.DataFrame(fund_records)

    # Stock monthly returns
    stock_records = []
    for sid in stock_ids:
        mu = np.random.normal(0.008, 0.005)
        vol = np.random.uniform(0.04, 0.12)
        rets = np.random.normal(mu, vol, len(months))
        for date, ret in zip(months, rets):
            stock_records.append({'date': date, 'stock': sid, 'ret': ret})
    stock_returns_df = pd.DataFrame(stock_records)

    # Quarterly holdings per fund
    holdings_records = []
    for fid in fund_ids:
        n_hold = np.random.randint(5, 20)
        current_stocks = np.random.choice(stock_ids, n_hold, replace=False)
        for q in quarters:
            if np.random.random() < 0.3:
                n_hold = np.random.randint(5, 20)
                current_stocks = np.random.choice(stock_ids, n_hold, replace=False)
            weights = np.random.dirichlet(np.ones(len(current_stocks)))
            for stock, weight in zip(current_stocks, weights):
                holdings_records.append({
                    'report_date': q, 'fund_id': fid, 'stock': stock, 'weight': weight
                })
    holdings_df = pd.DataFrame(holdings_records)

    # Monthly factors
    factors_df = pd.DataFrame({
        'date': months,
        'MKT': np.random.normal(0.006, 0.04, len(months)),
        'SMB': np.random.normal(0.002, 0.03, len(months)),
        'HML': np.random.normal(0.003, 0.03, len(months)),
        'MOM': np.random.normal(0.005, 0.04, len(months)),
    })

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


def load_real_data():
    """Load real WRDS CSV data."""
    print("\n📂 Loading real data from CSV files...")

    fund_path = input("   Path to fund_returns.csv: ").strip()
    fund_returns_df = pd.read_csv(fund_path, parse_dates=['date'])

    holdings_path = input("   Path to holdings.csv: ").strip()
    holdings_df = pd.read_csv(holdings_path, parse_dates=['report_date'])

    stock_path = input("   Path to stock_returns.csv: ").strip()
    stock_returns_df = pd.read_csv(stock_path, parse_dates=['date'])

    factor_path = input("   Path to factors.csv: ").strip()
    factors_df = pd.read_csv(factor_path, parse_dates=['date'])

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


def load_data():
    """Main data loading dispatcher."""
    print("\n🔄 Data Loading Mode")
    choice = input("   Use [D]emo data or [R]eal CSV data? (D/R, default=D): ").strip().upper()
    if choice == 'R':
        return load_real_data()
    else:
        return generate_demo_data(n_funds=100, n_stocks=50)


# =============================================================================
# 2. COMPOUND MONTHLY RETURNS TO QUARTERLY (Per Fund)
# =============================================================================

def compound_monthly_to_quarterly(fund_returns_df):
    """
    Compound monthly fund returns into quarterly returns.
    Input:  DataFrame with columns ['date', 'fund_id', 'fund_ret']
    Output: DataFrame with columns ['quarter', 'fund_id', 'fund_ret_q']
    """
    print("\n🔄 Compounding monthly returns to quarterly...")

    df = fund_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    # Compound: (1+r1)*(1+r2)*(1+r3) - 1
    quarterly = df.groupby(['fund_id', 'quarter']).apply(
        lambda x: (1 + x['fund_ret']).prod() - 1, include_groups=False
    ).reset_index()
    quarterly.columns = ['fund_id', 'quarter', 'fund_ret_q']

    # Filter: require at least 2 months of data in a quarter
    month_count = df.groupby(['fund_id', 'quarter']).size().reset_index(name='n_months')
    quarterly = quarterly.merge(month_count, on=['fund_id', 'quarter'])
    quarterly = quarterly[quarterly['n_months'] >= 2].drop(columns='n_months')

    n_funds = quarterly['fund_id'].nunique()
    n_quarters = quarterly['quarter'].nunique()
    print(f"   ✅ Quarterly returns: {len(quarterly)} obs ({n_funds} funds × up to {n_quarters} quarters)")

    return quarterly


# =============================================================================
# 3. COMPOUND STOCK RETURNS TO QUARTERLY
# =============================================================================

def compound_stock_returns_quarterly(stock_returns_df):
    """
    Compound monthly stock returns to quarterly.
    Input:  Long-format DataFrame with ['date', 'stock', 'ret']
    Output: Wide-format DataFrame, index=quarter, columns=stocks, values=quarterly return
    """
    print("\n🔄 Compounding stock returns to quarterly...")

    df = stock_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    quarterly = df.groupby(['stock', 'quarter']).apply(
        lambda x: (1 + x['ret']).prod() - 1, include_groups=False
    ).reset_index()
    quarterly.columns = ['stock', 'quarter', 'ret_q']

    # Pivot to wide format
    stock_ret_wide = quarterly.pivot(index='quarter', columns='stock', values='ret_q')
    print(f"   ✅ Quarterly stock returns: {stock_ret_wide.shape[0]} quarters × {stock_ret_wide.shape[1]} stocks")

    return stock_ret_wide


# =============================================================================
# 4. COMPOUND FACTORS TO QUARTERLY
# =============================================================================

def compound_factors_quarterly(factors_df):
    """Compound monthly factor returns to quarterly."""
    print("\n🔄 Compounding factors to quarterly...")

    df = factors_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    factor_cols = ['MKT', 'SMB', 'HML', 'MOM']
    quarterly = df.groupby('quarter')[factor_cols].apply(
        lambda x: (1 + x).prod() - 1
    )

    print(f"   ✅ Quarterly factors: {len(quarterly)} quarters")
    return quarterly


# =============================================================================
# 5. CALCULATE UNOBSERVED PERFORMANCE (UP) FOR ALL FUNDS
# =============================================================================

# =============================================================================
# Optimized Version: Pre-grouped holdings_index (Significant Performance Improvement)
# =============================================================================

def build_holdings_index(holdings_df: pd.DataFrame) -> dict:
    """
    Pre-group holdings by fund_id to create an index dictionary.
    This avoids repeatedly filtering the entire DataFrame in every loop iteration.
    """
    print("   🔧 Building holdings index by fund_id (performance optimization)...")
    return {fid: group for fid, group in holdings_df.groupby('fund_id')}


def construct_buy_and_hold_return(fund_id, quarter, holdings_by_fund: dict, stock_ret_wide):
    """
    Optimized version: Uses the pre-grouped holdings_by_fund dictionary 
    for O(1) access to each fund's holdings.
    """
    if fund_id not in holdings_by_fund:
        return np.nan

    fund_holdings = holdings_by_fund[fund_id]

    prev_reports = fund_holdings[fund_holdings['report_date'] < quarter]
    if prev_reports.empty:
        return np.nan

    prev_quarter = prev_reports['report_date'].max()
    prev_holdings = prev_reports[prev_reports['report_date'] == prev_quarter].copy()

    total_weight = prev_holdings['weight'].sum()
    if total_weight <= 0:
        return np.nan

    prev_holdings['weight'] = prev_holdings['weight'] / total_weight

    if quarter not in stock_ret_wide.index:
        return np.nan

    stock_rets = stock_ret_wide.loc[quarter]
    common_stocks = [s for s in prev_holdings['stock'].values 
                     if s in stock_rets.index and not pd.isna(stock_rets[s])]

    if len(common_stocks) == 0:
        return np.nan

    weights = prev_holdings.set_index('stock').loc[common_stocks, 'weight']
    weights = weights / weights.sum()          # Re-normalize after dropping missing stocks
    returns = stock_rets[common_stocks]

    return (weights * returns).sum()


def calculate_up_panel(fund_quarterly, holdings_df, stock_ret_wide):
    """
    Calculate Unobserved Performance (UP) panel for all funds 
    (with pre-grouped holdings optimization).
    """
    print("\n🔄 Calculating Unobserved Performance (UP) for all funds...")

    # === Key Optimization: Pre-build holdings index ===
    holdings_by_fund = build_holdings_index(holdings_df)

    quarters = sorted(fund_quarterly['quarter'].unique())
    fund_ids = fund_quarterly['fund_id'].unique()

    results = []
    total = len(fund_ids)
    progress_step = max(1, total // 10)

    for idx, fund_id in enumerate(fund_ids):
        if (idx + 1) % progress_step == 0:
            print(f"   Processing fund {idx + 1}/{total}...")

        fund_data = fund_quarterly[fund_quarterly['fund_id'] == fund_id]

        for _, row in fund_data.iterrows():
            q = row['quarter']
            reported_ret = row['fund_ret_q']

            # Use the optimized function
            bh_ret = construct_buy_and_hold_return(
                fund_id, q, holdings_by_fund, stock_ret_wide
            )

            if pd.isna(bh_ret):
                continue

            up = reported_ret - bh_ret
            results.append({
                'quarter': q,
                'fund_id': fund_id,
                'UP': up,
                'fund_ret_q': reported_ret
            })

    up_panel = pd.DataFrame(results)

    n_obs = len(up_panel)
    n_funds = up_panel['fund_id'].nunique()
    n_quarters = up_panel['quarter'].nunique()
    print(f"   ✅ UP Panel: {n_obs} observations ({n_funds} funds, {n_quarters} quarters)")
    print(f"   UP mean: {up_panel['UP'].mean():.4f}, std: {up_panel['UP'].std():.4f}")

    return up_panel


# =============================================================================
# 6. CROSS-SECTIONAL QUINTILE SORT + NEXT-QUARTER PORTFOLIO RETURNS
# =============================================================================

def form_long_short_portfolio(up_panel, min_funds_per_quarter=20):
    """
    Each quarter t:
      1. Sort all funds by UP(t) into quintiles
      2. Compute equal-weighted average NEXT-QUARTER return for each quintile
      3. Long-Short = Q5 - Q1 (high UP minus low UP)

    This is the standard predictive portfolio sort methodology.
    """
    print("\n🔄 Forming Long-Short Portfolio (Predictive Sort)...")

    quarters = sorted(up_panel['quarter'].unique())
    ls_results = []

    skipped = 0
    for i in range(len(quarters) - 1):
        current_q = quarters[i]
        next_q = quarters[i + 1]

        # Funds with valid UP in current quarter
        current_data = up_panel[up_panel['quarter'] == current_q].copy()

        if len(current_data) < min_funds_per_quarter:
            skipped += 1
            continue

        # Assign quintiles based on current-quarter UP
        try:
            current_data['quintile'] = pd.qcut(
                current_data['UP'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
            ).astype(int)
        except ValueError:
            # Not enough unique values for 5 quantiles
            skipped += 1
            continue

        # Get next-quarter returns for these funds
        next_data = up_panel[up_panel['quarter'] == next_q][['fund_id', 'fund_ret_q']]

        # Merge: assign quintile from t, get return from t+1
        merged = current_data[['fund_id', 'quintile']].merge(next_data, on='fund_id', how='inner')

        if len(merged) < min_funds_per_quarter:
            skipped += 1
            continue

        # Equal-weighted portfolio returns by quintile
        port_returns = merged.groupby('quintile')['fund_ret_q'].mean()

        if 5 in port_returns.index and 1 in port_returns.index:
            ls_ret = port_returns[5] - port_returns[1]
            ls_results.append({
                'quarter': next_q,
                'LS_return': ls_ret,
                'Q5_return': port_returns[5],
                'Q1_return': port_returns[1],
                'n_funds': len(merged),
                'n_Q5': len(merged[merged['quintile'] == 5]),
                'n_Q1': len(merged[merged['quintile'] == 1]),
            })

    ls_df = pd.DataFrame(ls_results).set_index('quarter')

    print(f"   ✅ Long-Short portfolio: {len(ls_df)} quarters (skipped {skipped} quarters)")
    print(f"\n   📊 Summary Statistics:")
    print(f"   {'Mean quarterly LS return:':<30} {ls_df['LS_return'].mean():.4f}")
    print(f"   {'Std quarterly LS return:':<30} {ls_df['LS_return'].std():.4f}")
    print(f"   {'Annualized LS return:':<30} {ls_df['LS_return'].mean() * 4:.4f} ({ls_df['LS_return'].mean() * 4 * 100:.2f}%)")
    print(f"   {'Mean Q5 (High UP) return:':<30} {ls_df['Q5_return'].mean():.4f}")
    print(f"   {'Mean Q1 (Low UP) return:':<30} {ls_df['Q1_return'].mean():.4f}")
    print(f"   {'Avg funds per quarter:':<30} {ls_df['n_funds'].mean():.0f}")
    print(f"   {'t-stat (simple):':<30} {ls_df['LS_return'].mean() / (ls_df['LS_return'].std() / np.sqrt(len(ls_df))):.2f}")

    return ls_df


# =============================================================================
# 7. RISK-ADJUSTED ALPHA (Carhart 4-Factor, Newey-West)
# =============================================================================

def compute_risk_adjusted_alpha(ls_df, factors_quarterly, max_lags=4):
    """
    Regress Long-Short returns on Carhart 4-factor model:
      LS_t = alpha + b1*MKT_t + b2*SMB_t + b3*HML_t + b4*MOM_t + epsilon_t

    Uses Newey-West (HAC) standard errors with specified max lags.
    """
    print("\n🔄 Computing Risk-Adjusted Alpha (Carhart 4-Factor, Newey-West)...")

    # Align time indices
    combined = ls_df[['LS_return']].join(factors_quarterly, how='inner')
    combined = combined.dropna()

    if len(combined) < 12:
        print("   ⚠️  Insufficient observations for regression (need ≥12).")
        return None

    y = combined['LS_return']
    X = sm.add_constant(combined[['MKT', 'SMB', 'HML', 'MOM']])

    # OLS with Newey-West (HAC) standard errors
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})

    print(f"\n   {'='*60}")
    print(f"   Carhart 4-Factor Regression (Newey-West, {max_lags} lags)")
    print(f"   {'='*60}")
    print(f"   {'Variable':<12} {'Coef':>10} {'Std Err':>10} {'t-stat':>10} {'p-value':>10}")
    print(f"   {'-'*52}")
    for var in model.params.index:
        label = 'Alpha' if var == 'const' else var
        print(f"   {label:<12} {model.params[var]:>10.4f} {model.bse[var]:>10.4f} "
              f"{model.tvalues[var]:>10.2f} {model.pvalues[var]:>10.4f}")
    print(f"   {'-'*52}")
    print(f"   {'R-squared:':<20} {model.rsquared:.4f}")
    print(f"   {'Observations:':<20} {int(model.nobs)}")
    print(f"   {'='*60}")

    alpha_quarterly = model.params['const']
    alpha_annual = alpha_quarterly * 4
    t_stat = model.tvalues['const']

    print(f"\n   📊 Key Result:")
    print(f"   Alpha (quarterly) = {alpha_quarterly:.4f} (t = {t_stat:.2f})")
    print(f"   Alpha (annualized) = {alpha_annual:.4f} ({alpha_annual * 100:.2f}%)")

    significance = ""
    if abs(t_stat) > 2.576:
        significance = "*** (1% level)"
    elif abs(t_stat) > 1.96:
        significance = "** (5% level)"
    elif abs(t_stat) > 1.645:
        significance = "* (10% level)"
    else:
        significance = "(not significant)"
    print(f"   Statistical significance: {significance}")

    return {
        'model': model,
        'alpha_quarterly': alpha_quarterly,
        'alpha_annual': alpha_annual,
        't_stat': t_stat,
        'p_value': model.pvalues['const'],
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
    }


# =============================================================================
# 8. ADDITIONAL ANALYSIS: Quintile Monotonicity
# =============================================================================

def analyze_quintile_monotonicity(up_panel, factors_quarterly, min_funds=20):
    """
    Check whether average next-quarter returns increase monotonically
    from Q1 (low UP) to Q5 (high UP) — a key prediction of the paper.
    """
    print("\n🔄 Analyzing Quintile Monotonicity (Table 3 replication)...")

    quarters = sorted(up_panel['quarter'].unique())
    port_returns = {q: [] for q in range(1, 6)}

    for i in range(len(quarters) - 1):
        current_q = quarters[i]
        next_q = quarters[i + 1]

        current_data = up_panel[up_panel['quarter'] == current_q].copy()
        if len(current_data) < min_funds:
            continue

        try:
            current_data['quintile'] = pd.qcut(
                current_data['UP'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
            ).astype(int)
        except ValueError:
            continue

        next_data = up_panel[up_panel['quarter'] == next_q][['fund_id', 'fund_ret_q']]
        merged = current_data[['fund_id', 'quintile']].merge(next_data, on='fund_id', how='inner')

        for q_bin in range(1, 6):
            q_ret = merged[merged['quintile'] == q_bin]['fund_ret_q'].mean()
            if not pd.isna(q_ret):
                port_returns[q_bin].append({'quarter': next_q, 'ret': q_ret})

    print(f"\n   {'='*70}")
    print(f"   {'Quintile':<12} {'Mean Ret (Q)':>14} {'Annualized':>12} {'Std':>10} {'Sharpe (ann)':>14}")
    print(f"   {'-'*70}")

    for q_bin in range(1, 6):
        if port_returns[q_bin]:
            rets = pd.Series([r['ret'] for r in port_returns[q_bin]])
            mean_q = rets.mean()
            std_q = rets.std()
            ann_ret = mean_q * 4
            sharpe = (mean_q * 4) / (std_q * 2) if std_q > 0 else np.nan
            print(f"   Q{q_bin} {'(Low UP)' if q_bin == 1 else '(High UP)' if q_bin == 5 else '':<8}"
                  f" {mean_q:>14.4f} {ann_ret*100:>10.2f}% {std_q:>10.4f} {sharpe:>14.2f}")

    print(f"   {'-'*70}")

     # Compute alpha for each quintile
    print(f"\n   Risk-Adjusted Alphas by Quintile:")
    print(f"   {'Quintile':<12} {'Alpha (Q)':>12} {'Alpha (Ann)':>14} {'t-stat':>10}")
    print(f"   {'-'*50}")

    for q_bin in range(1, 6):
        if port_returns[q_bin]:
            ret_series = pd.DataFrame(port_returns[q_bin]).set_index('quarter')['ret']
            combined = ret_series.to_frame('ret').join(factors_quarterly, how='inner').dropna()
            if len(combined) >= 12:
                y = combined['ret']
                X = sm.add_constant(combined[['MKT', 'SMB', 'HML', 'MOM']])
                model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
                alpha = model.params['const']
                t = model.tvalues['const']
                label = '(Low UP)' if q_bin == 1 else '(High UP)' if q_bin == 5 else ''
                print(f"   Q{q_bin} {label:<8} {alpha:>12.4f} {alpha * 400:>12.2f}% {t:>10.2f}")

    print(f"   {'='*70}")


# =============================================================================
# 9. PERSISTENCE ANALYSIS
# =============================================================================

def analyze_up_persistence(up_panel):
    """
    Analyze whether UP is persistent over time (autocorrelation).
    Papers shows UP has significant positive autocorrelation → skill is persistent.
    """
    print("\n🔄 Analyzing UP Persistence (Autocorrelation)...")

    # Create lagged UP for each fund
    panel = up_panel[['quarter', 'fund_id', 'UP']].copy()
    panel = panel.sort_values(['fund_id', 'quarter'])

    # Shift within each fund
    panel['UP_lag1'] = panel.groupby('fund_id')['UP'].shift(1)
    panel['UP_lag2'] = panel.groupby('fund_id')['UP'].shift(2)
    panel['UP_lag4'] = panel.groupby('fund_id')['UP'].shift(4)

    valid = panel.dropna(subset=['UP', 'UP_lag1'])

    if len(valid) < 20:
        print("   ⚠️  Insufficient data for persistence analysis.")
        return

    # Fama-MacBeth style: cross-sectional regression each quarter, then average
    quarters = sorted(valid['quarter'].unique())
    coeffs = []
    for q in quarters:
        q_data = valid[valid['quarter'] == q]
        if len(q_data) < 10:
            continue
        X = sm.add_constant(q_data['UP_lag1'])
        y = q_data['UP']
        try:
            model = sm.OLS(y, X).fit()
            coeffs.append(model.params['UP_lag1'])
        except Exception:
            continue

    if coeffs:
        mean_coeff = np.mean(coeffs)
        t_stat = mean_coeff / (np.std(coeffs) / np.sqrt(len(coeffs)))
        print(f"   Fama-MacBeth AR(1) coefficient: {mean_coeff:.4f} (t = {t_stat:.2f})")
        print(f"   → UP is {'persistent' if t_stat > 1.96 else 'NOT significantly persistent'}")
    else:
        print("   ⚠️  Could not compute persistence.")


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Step 1: Load Data
    fund_returns_df, holdings_df, stock_returns_df, factors_df = load_data()

    # Step 2: Compound to Quarterly
    fund_quarterly = compound_monthly_to_quarterly(fund_returns_df)
    stock_ret_wide = compound_stock_returns_quarterly(stock_returns_df)
    factors_quarterly = compound_factors_quarterly(factors_df)

    # Step 3: Calculate UP Panel
    up_panel = calculate_up_panel(fund_quarterly, holdings_df, stock_ret_wide)

    if up_panel.empty:
        print("\n❌ No valid UP observations. Check data alignment.")
        exit()

    # Step 4: Form Long-Short Portfolio (Predictive Sort)
    ls_df = form_long_short_portfolio(up_panel, min_funds_per_quarter=20)

    if ls_df.empty:
        print("\n❌ Could not form long-short portfolio.")
        exit()

    # Step 5: Risk-Adjusted Alpha
    alpha_results = compute_risk_adjusted_alpha(ls_df, factors_quarterly, max_lags=4)

    # Step 6~7: Additional Analysis
    analyze_quintile_monotonicity(up_panel, factors_quarterly)
    analyze_up_persistence(up_panel)

    # Step 8: Save Results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    up_panel.to_csv(output_dir / "up_panel.csv", index=False)
    ls_df.to_csv(output_dir / "long_short_returns.csv")

    print("\n" + "=" * 80)
    print("🎉 Replication Completed Successfully!")
    print("=" * 80)
    print(f"📁 Results saved to: {output_dir.resolve()}")
    print("   - up_panel.csv")
    print("   - long_short_returns.csv")
    print("\nReady for real WRDS data!")
    print("=" * 80)
