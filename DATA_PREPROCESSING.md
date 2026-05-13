***

# Data Preprocessing Pipeline for Empirical Asset Pricing

This document describes the complete data preprocessing workflow used to build the firm‑month panel for our empirical asset pricing and machine learning experiments.  
The goal is to start from raw WRDS data (CRSP and Compustat), construct clean characteristics, and produce an analysis‑ready panel that can be fed into the model factory and rolling horse‑race evaluation.

## Overview

We follow four main stages:

1. **Data extraction from WRDS** (CRSP, Compustat, CCM link table).  
2. **Sample cleaning and filters** (security type, listing, prices, returns).  
3. **Feature engineering and alignment** (accounting lag, frequency alignment, winsorization, missing values).  
4. **Panel construction and export** (firm‑month panel + time index for rolling evaluation).

All processing is done in Python (either locally or in WRDS Cloud) using `wrds`, `pandas`, `numpy`, and related libraries.

***

## 1. Data Extraction from WRDS

### 1.1 WRDS connection

We connect to WRDS via the official `wrds` Python package, which provides a thin wrapper around the underlying PostgreSQL database.

```python
import wrds

# Establish a WRDS connection (prompts for username/password the first time)
conn = wrds.Connection()
```

### 1.2 CRSP monthly returns

We extract monthly stock returns and basic price/volume information from the CRSP monthly stock file (`crsp.msf`), together with share codes and exchange codes from the monthly stock events file (`crsp.mse`).

```python
crsp_msf = conn.raw_sql(
    """
    SELECT
        msf.permno,
        msf.permco,
        msf.date,
        msf.ret,
        msf.retx,
        msf.prc,
        msf.shrout,
        mse.shrcd,
        mse.exchcd
    FROM crsp.msf AS msf
    LEFT JOIN crsp.mse AS mse
      ON msf.permno = mse.permno
     AND msf.date   = mse.date
    WHERE msf.date >= '1970-01-01'
    """,
    date_cols=['date'],
)
```

We convert the CRSP date to a `YYYYMM` style period indicator that will later serve as a time index for the panel and rolling evaluation.

```python
import pandas as pd

crsp_msf['yyyymm'] = crsp_msf['date'].dt.to_period('M').astype(int)
```

### 1.3 Compustat fundamentals

We pull annual accounting information from Compustat North America (`comp.funda`) by GVKEY, focusing on a set of core balance sheet and income statement variables used to construct classic characteristics (e.g., book‑to‑market, profitability, investment).

```python
funda = conn.raw_sql(
    """
    SELECT
        gvkey,
        datadate,
        fyear,
        at,      -- total assets
        ceq,     -- common equity
        sale,    -- sales
        cogs,    -- cost of goods sold
        xint,    -- interest expense
        xsga,    -- SG&A
        ib,      -- income before extraordinary items
        capx     -- capital expenditures
    FROM comp.funda
    WHERE indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc = 'D'
      AND consol = 'C'
      AND fyear >= 1960
    """,
    date_cols=['datadate'],
)
```

We also extract the CRSP–Compustat link table (`crsp.ccmxpf_linktable`) to map GVKEYs to CRSP PERMNOs.

```python
ccm = conn.raw_sql(
    """
    SELECT
        gvkey,
        lpermno AS permno,
        linktype,
        linkprim,
        linkdt,
        linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE lpermno IS NOT NULL
    """,
    date_cols=['linkdt', 'linkenddt'],
)
```

***

## 2. Sample Cleaning and Filters

### 2.1 CRSP stock universe filters

We restrict the CRSP universe to common stocks listed on major US exchanges, and exclude micro‑priced penny stocks. A typical set of filters is:

- **Security type**: common shares only (`shrcd` in {10, 11}).  
- **Exchange**: NYSE, AMEX, NASDAQ (`exchcd` in {1, 2, 3}).  
- **Price filter**: absolute price ≥ 5 USD to avoid extreme penny stocks.  

```python
crsp_clean = crsp_msf.copy()

# Common shares
crsp_clean = crsp_clean[crsp_clean['shrcd'].isin([10, 11])]

# Major exchanges
crsp_clean = crsp_clean[crsp_clean['exchcd'].isin([1, 2, 3])]

# Use absolute price; CRSP prices are negative for bid/ask
crsp_clean['prc_abs'] = crsp_clean['prc'].abs()
crsp_clean = crsp_clean[crsp_clean['prc_abs'] >= 5.0]
```

We typically also treat CRSP missing returns (`ret` set to “B”, “C”, or missing) carefully. For a baseline, we may drop observations with missing returns, or substitute delisting returns when available.

```python
crsp_clean = crsp_clean[crsp_clean['ret'].notna()]
crsp_clean['ret'] = crsp_clean['ret'].astype(float)
```

### 2.2 Basic market equity

We compute market equity (ME) as `price × shares outstanding`, scaled into millions for numerical stability.

```python
crsp_clean['me'] = crsp_clean['prc_abs'] * crsp_clean['shrout'] / 1000.0  # in million USD
```

***

## 3. Linking CRSP and Compustat

We follow the standard CRSP–Compustat linking procedure via the CCM link table, restricting to high‑quality links (`linktype` and `linkprim` filters) and respecting the effective link dates.

### 3.1 CCM link cleaning

```python
# Keep standard link types and primary links
ccm = ccm[
    ccm['linktype'].isin(['LU', 'LC'])  # standard CRSP-Compustat links
    & ccm['linkprim'].isin(['P', 'C'])  # primary links
]

# Treat open-ended linkenddt as far future
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
```

### 3.2 Map fundamentals to CRSP

We assign each Compustat observation to a CRSP PERMNO via date‑based merging:

```python
# Convert datadate to a calendar date if needed
funda['datadate'] = pd.to_datetime(funda['datadate'])
```

We then merge on GVKEY and require `datadate` to fall within the CCM link interval:

```python
# Merge Compustat with CCM on gvkey
funda_ccm = funda.merge(ccm[['gvkey', 'permno', 'linkdt', 'linkenddt']],
                        on='gvkey', how='inner')

# Keep observations where datadate lies within the link span
mask = (funda_ccm['datadate'] >= funda_ccm['linkdt']) & \
       (funda_ccm['datadate'] <= funda_ccm['linkenddt'])
funda_ccm = funda_ccm[mask]
```

***

## 4. Accounting Lag and Frequency Alignment

To avoid look‑ahead bias, we lag accounting information so that it becomes available to investors with a delay (e.g., 6 months after the fiscal year end). A common convention is to align a given fiscal year’s fundamentals with returns from July of year *t+1* to June of *t+2*; here we use a simpler but conservative lag at the monthly level.

### 4.1 Construct a reporting year‑month

```python
funda_ccm['fyearm'] = funda_ccm['datadate'].dt.to_period('M').astype(int)
```

We then create a “usable from” month index by adding a lag (e.g., 6 months):

```python
# Example: fundamentals become investable 6 months after datadate
funda_ccm['yyyymm_effective'] = (
    (funda_ccm['datadate'] + pd.offsets.MonthEnd(6))
    .dt.to_period('M').astype(int)
)
```

### 4.2 Expand fundamentals to a monthly panel

We want a firm‑month panel, so we carry forward the latest available fundamentals for each PERMNO until new accounting information arrives.

A simple approach:

```python
# Keep the key variables for characteristics
acct_vars = [
    'at', 'ceq', 'sale', 'cogs', 'xint', 'xsga', 'ib', 'capx'
]

funda_ccm = funda_ccm[['permno', 'yyyymm_effective'] + acct_vars].copy()
funda_ccm = funda_ccm.sort_values(['permno', 'yyyymm_effective'])

# For each permno, forward-fill accounting variables over time
funda_ccm = funda_ccm.set_index(['permno', 'yyyymm_effective'])
funda_ccm = funda_ccm.groupby(level=0).ffill().reset_index()
```

We finally merge the monthly fundamentals into the CRSP monthly file:

```python
# Merge on permno and yyyymm (CRSP) with the effective accounting date
panel = crsp_clean.merge(
    funda_ccm,
    left_on=['permno', 'yyyymm'],
    right_on=['permno', 'yyyymm_effective'],
    how='left'
)
panel.drop(columns=['yyyymm_effective'], inplace=True)
```

***

## 5. Feature Engineering

With CRSP returns and lagged Compustat variables in a unified monthly panel, we construct standard asset pricing characteristics.

### 5.1 Book‑to‑market (B/M)

```python
# Book equity
panel['be'] = panel['ceq']  # can be refined with preferred stock and other adjustments

# Book-to-market
panel['bm'] = panel['be'] / panel['me']
```

### 5.2 Profitability and investment

```python
# Operating profitability (simplified)
panel['op'] = (panel['sale'] - panel['cogs'] - panel['xsga']) / panel['be']

# Investment rate (change in total assets)
panel = panel.sort_values(['permno', 'date'])
panel['at_lag'] = panel.groupby('permno')['at'].shift(12)  # 12-month lag
panel['inv'] = (panel['at'] - panel['at_lag']) / panel['at_lag']
```

### 5.3 Momentum

We construct a 12–2 momentum variable based on past returns, skipping the most recent month.

```python
# Use raw returns retx (excluding distributions) for momentum
panel = panel.sort_values(['permno', 'date'])

# Rolling cumulative log return from t-12 to t-2
panel['log_retx'] = np.log1p(panel['retx'].fillna(0.0))

panel['mom12_2'] = (
    panel.groupby('permno')['log_retx']
         .rolling(window=11, min_periods=8)
         .sum()
         .reset_index(level=0, drop=True)
)

# Drop the most recent month contribution (t-1) by shifting
panel['mom12_2'] = panel.groupby('permno')['mom12_2'].shift(1)

# Convert back to simple return
panel['mom12_2'] = np.expm1(panel['mom12_2'])
```

***

## 6. Winsorization and Missing Values

Characteristics used as model inputs are winsorized cross‑sectionally to reduce the impact of extreme outliers. We perform winsorization at the monthly cross‑section level, typically at the 1st and 99th percentiles, followed by median imputation where necessary.

### 6.1 Cross‑sectional winsorization

```python
char_cols = ['bm', 'op', 'inv', 'mom12_2']

def winsorize_series(x, p=0.01):
    lower = x.quantile(p)
    upper = x.quantile(1 - p)
    return x.clip(lower, upper)

panel = panel.sort_values(['yyyymm', 'permno'])

for col in char_cols:
    panel[col] = (
        panel.groupby('yyyymm')[col]
              .transform(lambda x: winsorize_series(x, p=0.01))
    )
```

### 6.2 Missing values

Some characteristics will still be missing (especially at the beginning of each firm’s history). We keep those observations but will rely on the model pipelines to handle missing values systematically:

- Linear and neural network models use a `SimpleImputer(strategy='median')` as the first step in their pipeline.  
- Tree‑based models in sklearn are preceded by imputation only.  
- XGBoost and LightGBM handle `NaN` natively.

You can optionally drop rows with too many missing characteristics or define a minimum history requirement per firm.

***

## 7. Final Panel and Export

### 7.1 Construct model inputs

We define:

- `y`: next‑month excess return (or raw return, depending on the specification).  
- `X`: matrix of standardized characteristics (standardization is handled inside the model pipelines).  
- `time_index`: monthly time index used for rolling evaluation.  

```python
# Target variable: next-month return
panel = panel.sort_values(['permno', 'date'])
panel['ret_fwd'] = panel.groupby('permno')['ret'].shift(-1)

# Drop the last month for each firm where ret_fwd is missing
panel = panel[panel['ret_fwd'].notna()]

feature_cols = char_cols  # list of constructed characteristics
X = panel[feature_cols].values
y = panel['ret_fwd'].values
time_index = panel['yyyymm'].values
```

### 7.2 Save for downstream modeling

We store both the full panel and the numpy arrays for modeling in a compact format (e.g., Parquet for the panel, NumPy `.npz` for the modeling arrays).

```python
# Save full panel
panel.to_parquet("data/firm_month_panel.parquet", index=False)

# Save modeling arrays
np.savez_compressed(
    "data/model_inputs.npz",
    X=X,
    y=y,
    time_index=time_index,
    feature_names=np.array(feature_cols),
)
```

***

## 8. Integration with the Modeling and Evaluation Code

The resulting `firm_month_panel.parquet` and `model_inputs.npz` can be directly consumed by the modeling code:

- Use the `AssetPricingModels` factory to obtain a specific estimator (e.g., LASSO, Random Forest, XGBoost).  
- Use the `rolling_horse_race` function to run time‑series aware rolling or expanding window evaluation based on `time_index`.

A minimal example:

```python
from models import AssetPricingModels, rolling_horse_race  # your model module

data = np.load("data/model_inputs.npz")
X = data['X']
y = data['y']
time_index = data['time_index']

factory = AssetPricingModels(random_state=42, n_jobs=-1)
model = factory.get_model('lasso', alpha=0.01)

result = rolling_horse_race(
    model=model,
    X=X,
    y=y,
    time_index=time_index,
    train_window=120,   # e.g. 120 months (= 10 years) rolling window
    test_window=1,
)
```

This pipeline ensures that:

- No future information leaks into the training set.  
- All characteristics are properly lagged and cleaned.  
- The modeling layer receives a stable, well‑defined input space.

***

If you tell me your actual variable list (which characteristics you plan to use), I can tailor the feature engineering section to match exactly what you want to document in the GitHub repo.
