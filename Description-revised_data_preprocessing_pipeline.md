# Data Preprocessing Pipeline for Empirical Asset Pricing

This document describes a robust data preprocessing workflow for empirical asset pricing research using WRDS data, with a focus on CRSP monthly returns, Compustat annual fundamentals, and time-series-safe feature construction. It is designed for firm-month panel prediction tasks and explicitly addresses common sources of look-ahead bias, duplicate records, and survivorship distortions.

## Scope

The workflow assumes the following data sources on WRDS:

- **CRSP monthly stock file** for returns and prices.
- **CRSP name history table** for share and exchange codes.
- **CRSP delisting table** for delisting returns.
- **Compustat annual fundamentals** for accounting characteristics.
- **CRSP-Compustat Merged (CCM) link table** for mapping `gvkey` to `permno`.
- **Risk-free rate data** for excess return construction, if the target is defined as excess return rather than raw return.

The final output is a clean firm-month panel with one observation per `permno-month`, lagged accounting variables, market variables, diagnostic checks, and a time index suitable for rolling or expanding window evaluation.

## Design Principles

The pipeline follows six principles:

1. **Point-in-time integrity**: every feature used to predict month `t+1` must be observable by the end of month `t`.
2. **No global leakage**: transformations that learn from data distributions, such as scaling, PCA, and model-based imputation, must be fit only on the training window, not on the full sample. [datasciencecentral](https://www.datasciencecentral.com/avoiding-look-ahead-bias-in-time-series-modelling-1/)
3. **One row per asset-month**: after all merges, the panel must contain at most one observation for each `permno` and month.
4. **Accounting data are mapped, not exact-matched**: annual Compustat observations must be carried forward only after an economically valid reporting lag.
5. **Delisting events matter**: final-month returns must incorporate delisting returns when available to reduce survivorship bias.
6. **Diagnostics are mandatory**: every major merge and transformation should be followed by sanity checks.

## Data Sources and Core Tables

### CRSP market data

The main monthly return source is `crsp.msf`. This table contains monthly returns, ex-dividend returns, prices, shares outstanding, and identifiers such as `permno` and `permco`.

A direct join from `crsp.msf` to `crsp.mse` on `permno + date` should be avoided. The `mse` file is an event file, so one security-date pair can have multiple rows if multiple events occur on the same day. A direct join can therefore duplicate return observations and inflate the panel.

For share code (`shrcd`) and exchange code (`exchcd`), the safer approach is to use the CRSP name history table, typically `crsp.msenames`, and merge by identifier plus date interval:

```sql
names.namedt <= msf.date <= names.nameendt
```

This preserves a one-to-many history structure without creating false duplicates at the monthly return level.

### CRSP delisting data

Delisting returns should be extracted from `crsp.msedelist`. If a stock delists, the last observable monthly return in `crsp.msf` may not capture the full economic loss or gain. The standard adjustment is:

\[
ret^{adj} = (1 + ret) (1 + dlret) - 1
\]

This step is particularly important for distressed firms, where omitting delisting returns mechanically biases returns upward through survivorship.

### Compustat annual fundamentals

Accounting variables are drawn from `comp.funda`, usually restricted to standard industrial-format consolidated statements, for example:

- `indfmt = 'INDL'`
- `datafmt = 'STD'`
- `popsrc = 'D'`
- `consol = 'C'`

These filters isolate the standard annual accounting statements used in most empirical asset pricing studies.

### CCM link table

Linking Compustat to CRSP should be done through the CRSP-Compustat merged link table, usually `crsp.ccmxpf_linktable`, using valid link types, primary links, and link date ranges.

### Risk-free rate data

If the prediction target is next-month excess return rather than raw return, the risk-free rate must be merged at the monthly frequency before final target construction. This can come from a Fama-French monthly risk-free series or another approved Treasury-based WRDS source.

## Step 1: Extract and Clean CRSP Monthly Returns

### 1.1 Pull monthly returns

Start from `crsp.msf` and retrieve at least the following columns:

- `permno`
- `permco`
- `date`
- `ret`
- `retx`
- `prc`
- `shrout`

Convert the date into an explicit calendar month identifier. If downstream code expects `YYYYMM`, do not use `to_period('M').astype(int)`, because that produces a period ordinal rather than a human-readable year-month code. Use:

```python
crsp['yyyymm'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month
```

### 1.2 Merge name-history attributes correctly

To obtain `shrcd` and `exchcd`, merge CRSP monthly returns to `crsp.msenames` using `permno` and the active-name interval.

In pseudocode:

```python
# keep rows where namedt <= date <= nameendt
```

This is safer than merging to `crsp.mse` by exact date and avoids event-driven duplication.

### 1.3 Restrict the investable universe

Typical baseline filters include:

- Common shares only: `shrcd in {10, 11}`.
- Major exchanges only: `exchcd in {1, 2, 3}`.
- Positive or sufficiently large price, often `abs(prc) >= 5` for robustness checks.

Depending on the project, additional filters may exclude ADRs, REITs, closed-end funds, or other special security types.

### 1.4 Merge delisting returns

Merge `crsp.msedelist` onto the monthly return panel and adjust the final observed return whenever `dlret` is available:

```python
ret_adjusted = (1 + ret) * (1 + dlret) - 1
```

If `dlret` is missing for a delisted firm, some studies apply fallback rules by delisting code. Whether such rules are used should be stated explicitly in the replication notes.

### 1.5 Aggregate market equity at the PERMCO level

For firms with multiple share classes, market equity should be aggregated across all PERMNOs that belong to the same PERMCO in a given month. Using `abs(prc) * shrout` at the individual PERMNO level can overstate ratios such as book-to-market for multi-class firms.

A standard approach is:

1. Compute security-level market equity.
2. Sum market equity across all PERMNOs within the same `permco-month`.
3. Assign the aggregated market equity back to the security that is kept in the final panel, often the largest-PERMNO security in that month.

This step should be documented clearly because it affects any characteristic that uses market equity in the denominator.

## Step 2: Extract and Clean Compustat Fundamentals

### 2.1 Pull annual accounting variables

From `comp.funda`, retrieve the annual variables needed for the characteristic set, for example:

- `gvkey`
- `datadate`
- `fyear`
- `at`
- `ceq`
- `sale`
- `cogs`
- `xsga`
- `xint`
- `ib`
- `capx`
- `sic`

If the study excludes financial firms, keep `sic` or another industry code at this stage.

### 2.2 Exclude financial firms when appropriate

Many empirical asset pricing studies exclude financial firms because their balance sheet structure differs materially from industrial firms. A common filter is to remove SIC codes 6000-6999.

If the study keeps financials, the documentation should explain why. If the study removes them, that filter should be applied explicitly and consistently.

### 2.3 Compute annual characteristics at the accounting frequency

Variables that are naturally defined using annual accounting statements should be computed at the annual Compustat level before any monthly expansion.

For example, investment is better defined as:

\[
inv_t = \frac{at_t - at_{t-1}}{at_{t-1}}
\]

computed on annual Compustat observations by `gvkey`, not by applying `shift(12)` after merging to a monthly panel. Computing `inv` at the monthly panel level assumes a complete monthly sequence and can fail when months are missing.

Other annual characteristics, such as profitability or book equity, should likewise be formed before monthly mapping whenever possible.

## Step 3: Link Compustat to CRSP

### 3.1 Use valid CCM links

Merge `comp.funda` to `crsp.ccmxpf_linktable` by `gvkey`, and keep only valid link types and primary links. A common choice is:

- `linktype in {'LU', 'LC'}`
- `linkprim in {'P', 'C'}`

Open-ended `linkenddt` values should be filled with a far-future date before applying the date filter.

### 3.2 Enforce link-date validity

Only retain accounting observations whose `datadate` falls inside the active CCM link window:

```python
linkdt <= datadate <= linkenddt
```

This prevents invalid historical mappings and helps avoid duplicated firm identities.

### 3.3 Deduplicate linked records

After CCM linking, it is possible to obtain multiple accounting records for the same `permno-month`, especially when link intervals overlap or when accounting observations are mapped too aggressively.

The final linked accounting dataset should be deduplicated with an explicit rule. A common rule is:

- for a given `permno` and target month, keep the most recent valid `datadate` that is already public by that month.

The deduplication rule should be documented because it directly affects panel uniqueness.

## Step 4: Map Annual Accounting Data to Monthly Returns

This is the most important step for avoiding hidden logic errors.

### 4.1 Do not rely on annual forward-fill alone

Forward-filling Compustat observations within the annual accounting table does not generate missing months. If the accounting file has one row per firm-year, `groupby('permno').ffill()` only fills across existing annual rows and does not create intermediate monthly observations.

As a result, an exact merge on:

```python
yyyymm == yyyymm_effective
```

will fail for most firm-months and produce mostly missing accounting variables.

### 4.2 Recommended approach: point-in-time monthly mapping

There are two valid implementations.

#### Option A: `merge_asof`

Sort both datasets by `permno` and date, then match each CRSP month to the most recent accounting observation whose effective date is less than or equal to the CRSP month. This is typically the cleanest approach in pandas when the logic is strictly point-in-time.

#### Option B: build a monthly grid first

Create a complete firm-month grid for each `permno`, place annual accounting values at the month when they become investable, then forward-fill within each `permno`, and finally merge the completed accounting panel to CRSP.

Both methods are valid, but one must be implemented explicitly. A plain forward-fill over annual rows is not sufficient.

### 4.3 Apply a reporting lag

To avoid look-ahead bias, accounting data must become usable only after a delay. A common conservative implementation is to make an annual observation available six months after `datadate`, though some projects instead use actual filing dates where available.

The monthly mapping therefore uses an effective date, for example:

```python
effective_date = datadate + MonthEnd(6)
```

Then each month is matched to the latest accounting record with:

```python
effective_date <= month_end
```

This ensures that month `t` only sees accounting information that was already public by the end of month `t`. [analyzingalpha](https://analyzingalpha.com/look-ahead-bias)

## Step 5: Construct Characteristics

### 5.1 Market-based variables

Construct market-based variables such as:

- market equity
- lagged return
- momentum
- turnover or liquidity proxies
- volatility measures

All of these should be built using only information available up to the end of the formation month.

### 5.2 Accounting-based variables

Construct annual characteristics such as:

- book-to-market
- operating profitability
- investment
- leverage
- asset growth

These should be computed at the accounting frequency first, then mapped to the monthly panel using the effective-date logic above.

### 5.3 Momentum construction

Momentum signals should be based on past returns only, with the usual skip-month convention if required by the design. For example, a 12-2 momentum signal for month `t` uses returns from months `t-12` through `t-2`, not `t-1` or `t+1`.

### 5.4 Excess return target

If the prediction target is next-month excess return, merge the monthly risk-free rate and define:

\[
rx_{t+1} = r_{t+1} - rf_{t+1}
\]

The documentation should state clearly whether the target is raw return, excess return, or excess return net of delisting adjustment.

## Step 6: Handle Outliers, Missing Values, and Standardization

### 6.1 Winsorization

Outlier treatment should be done cross-sectionally within each month, not over the full sample. For example, each characteristic can be clipped at the 1st and 99th percentiles using only the firms available in that month.

Using full-sample quantiles introduces future information into earlier months and therefore creates leakage. [towardsdatascience](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)

### 6.2 Missing values

Missing values can be handled in one of two ways:

- **Preprocessing-stage imputation**: for example, cross-sectional median imputation by month.
- **Model-pipeline imputation**: for example, `SimpleImputer(strategy='median')` fit only within the training window.

The choice should be stated explicitly. If the final modeling code already applies imputation inside a scikit-learn pipeline, the data-processing document should say that raw missingness is preserved until model fitting.

### 6.3 Cross-sectional standardization

If characteristics are standardized before modeling, the documentation should explain whether this occurs:

- in the preprocessing stage, month by month, using cross-sectional z-scores or rank transforms; or
- inside the model pipeline, where the transformer is fit only on the training window.

For empirical asset pricing, both approaches can be valid, but the implementation must be point-in-time consistent and not depend on full-sample moments. [codesignal](https://codesignal.com/learn/courses/preparing-financial-data-for-machine-learning/lessons/addressing-data-leakage-in-time-series)

## Step 7: Construct the Final Firm-Month Panel

### 7.1 Enforce uniqueness

After all merges, verify that each `permno-yyyymm` pair appears at most once. If duplicates remain, trace them back to the relevant join and resolve them before modeling.

A useful invariant is:

```python
panel.groupby(['permno', 'yyyymm']).size().max() == 1
```

### 7.2 Define sample start and end dates carefully

Even if CRSP extraction begins in 1970 and Compustat extraction begins earlier, the usable modeling sample will usually start later because:

- accounting variables require a reporting lag,
- some characteristics require prior history,
- momentum or volatility signals need rolling windows,
- delisting-adjusted return targets may require aligned auxiliary files.

For a monthly panel using annual accounting variables with a six-month lag and momentum-style predictors, the practical sample start may be around 1972-1973 rather than the raw earliest CRSP date.

The final documentation should therefore distinguish between:

- raw extraction period,
- post-merge coverage period,
- final modeling sample period.

### 7.3 Export clean artifacts

Store at least two outputs:

- a full panel file, such as `firm_month_panel.parquet`, with identifiers, dates, target, and raw characteristics;
- a modeling-ready artifact, such as `model_inputs.npz` or another panel subset used in the machine learning experiments.

## Step 8: Sanity Checks and Diagnostics

A complete preprocessing pipeline should include a short diagnostics section.

Recommended checks:

- Number of observations before and after each major merge.
- Number of unique firms per month through time.
- Duplicate counts for `permno-yyyymm` after each merge.
- Share of missing values by variable and by month.
- Distributional summaries of key variables before and after winsorization.
- Coverage comparison for CRSP-only versus CRSP+Compustat merged samples.

These diagnostics help identify silent problems such as duplicate inflation, broken month mapping, excessive data loss, or systematic missingness in certain periods.

## Step 9: Time-Series-Safe Modeling Interface

The preprocessing layer should hand off data to the modeling layer in a form that preserves time ordering.

At minimum, provide:

- `X`: feature matrix.
- `y`: target vector.
- `time_index`: month identifier.
- optionally `permno` for portfolio sorting and firm-level diagnostics.

All model fitting, scaling, dimensionality reduction, and hyperparameter tuning should occur inside rolling or expanding windows, never on the full sample. [linkedin](https://www.linkedin.com/posts/soledad-galli_data-leakage-in-time-series-forecasting-activity-7297570555526877184-GlsC)

If the project supports both rolling and expanding windows, the documentation should state the rationale for each:

- **Rolling window**: keeps the training sample length fixed and adapts more quickly to structural change.
- **Expanding window**: uses all available history and may stabilize parameter estimates when signals are weak.

## Step 10: Versioning and Reproducibility Notes

The documentation should record the exact data environment used for extraction and preprocessing, including:

- WRDS platform used, such as WRDS Cloud.
- CRSP product version, especially if using a post-migration WRDS environment where table names or field names may differ.
- Extraction date or snapshot period.
- Python version and key package versions.

This is especially important if the project may later be rerun under a different WRDS schema or CRSP release.

## Suggested Minimal Output Schema

A practical final panel might contain the following columns:

| Column | Description |
|---|---|
| `permno` | CRSP security identifier |
| `permco` | CRSP company identifier |
| `gvkey` | Compustat firm identifier |
| `date` | Month-end date |
| `yyyymm` | Calendar month in `YYYYMM` format |
| `ret` | Raw monthly return |
| `retx` | Monthly return excluding distributions |
| `dlret` | Delisting return, if available |
| `ret_adj` | Delisting-adjusted return |
| `rf` | Monthly risk-free rate |
| `rx_fwd` | Next-month excess return target |
| `me_company` | Company-level market equity |
| `bm` | Book-to-market |
| `op` | Operating profitability |
| `inv` | Investment / asset growth |
| `mom12_2` | Twelve-to-two momentum |
| `sic` | Industry code |

## Recommended Processing Order

1. Extract CRSP monthly returns.
2. Merge CRSP name-history data using date intervals.
3. Apply universe filters.
4. Merge delisting returns and build adjusted returns.
5. Compute security-level and company-level market equity.
6. Extract Compustat fundamentals.
7. Exclude financial firms if required.
8. Compute annual accounting characteristics.
9. Link Compustat to CRSP through CCM with valid date ranges.
10. Map annual accounting data to monthly observations using `merge_asof` or a monthly grid plus forward-fill.
11. Merge risk-free rate if excess returns are used.
12. Build market-based and accounting-based characteristics.
13. Apply cross-sectional outlier treatment.
14. Run diagnostics and deduplication checks.
15. Export the final panel and modeling inputs.

## Summary

A credible asset-pricing preprocessing pipeline is not just a sequence of joins. It must preserve point-in-time information, avoid duplicate inflation, incorporate delisting outcomes, treat multi-class firms correctly, and map annual accounting information into the monthly panel using an explicit time-valid procedure. The most common silent failure is not in model estimation but in preprocessing logic, especially around date alignment, duplicate records, and leakage from future information. [corporatefinanceinstitute](https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/look-ahead-bias/)
