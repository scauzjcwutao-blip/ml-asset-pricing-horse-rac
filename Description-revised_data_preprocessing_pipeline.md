Data Preprocessing Pipeline for Empirical Asset Pricing
This document describes a robust data preprocessing workflow for empirical asset pricing research using WRDS data, with a focus on CRSP monthly returns, Compustat annual fundamentals, and time-series-safe feature construction. It is designed for firm-month panel prediction tasks and explicitly addresses common sources of look-ahead bias, duplicate records, and survivorship distortions.

Scope
The workflow assumes the following data sources on WRDS:

CRSP monthly stock file for returns and prices.

CRSP name history table for share and exchange codes.

CRSP delisting table for delisting returns.

Compustat annual fundamentals for accounting characteristics.

CRSP-Compustat Merged (CCM) link table for mapping gvkey to permno.

Risk-free rate data for excess return construction, if the target is defined as excess return rather than raw return.

The final output is a clean firm-month panel with one observation per permno-month, lagged accounting variables, market variables, diagnostic checks, and a time index suitable for rolling or expanding window evaluation.

Design Principles
The pipeline follows six principles:

Point-in-time integrity: every feature used to predict month t+1 must be observable by the end of month t.

No global leakage: transformations that learn from data distributions, such as scaling, PCA, and model-based imputation, must be fit only on the training window, not on the full sample.

One row per asset-month: after all merges, the panel must contain at most one observation for each permno and month.

Accounting data are mapped, not exact-matched: annual Compustat observations must be carried forward only after an economically valid reporting lag.

Delisting events matter: final-month returns must incorporate delisting returns when available to reduce survivorship bias.

Diagnostics are mandatory: every major merge and transformation should be followed by sanity checks.

Data Sources and Core Tables
CRSP market data
The main monthly return source is crsp.msf. This table contains monthly returns, ex-dividend returns, prices, shares outstanding, and identifiers such as permno and permco.

A direct join from crsp.msf to crsp.mse on permno + date should be avoided. The mse file is an event file, so one security-date pair can have multiple rows if multiple events occur on the same day. A direct join can therefore duplicate return observations and inflate the panel.

For share code (shrcd) and exchange code (exchcd), the safer approach is to use the CRSP name history table, typically crsp.msenames, and merge by identifier plus date interval:

sql
names.namedt <= msf.date <= names.nameendt
This preserves a one-to-many history structure without creating false duplicates at the monthly return level.

CRSP delisting data
Delisting returns should be extracted from crsp.msedelist. If a stock delists, the last observable monthly return in crsp.msf may not capture the full economic loss or gain. The standard adjustment is:

r
e
t
a
d
j
=
(
1
+
r
e
t
)
(
1
+
d
l
r
e
t
)
−
1
ret 
adj
 =(1+ret)(1+dlret)−1
This step is particularly important for distressed firms, where omitting delisting returns mechanically biases returns upward through survivorship.

Compustat annual fundamentals
Accounting variables are drawn from comp.funda, usually restricted to standard industrial-format consolidated statements, for example:

indfmt = 'INDL'

datafmt = 'STD'

popsrc = 'D'

consol = 'C'

These filters isolate the standard annual accounting statements used in most empirical asset pricing studies.

CCM link table
Linking Compustat to CRSP should be done through the CRSP-Compustat merged link table, usually crsp.ccmxpf_linktable, using valid link types, primary links, and link date ranges.

Risk-free rate data
If the prediction target is next-month excess return rather than raw return, the risk-free rate must be merged at the monthly frequency before final target construction. This can come from a Fama-French monthly risk-free series or another approved Treasury-based WRDS source.

Step 1: Extract and Clean CRSP Monthly Returns
1.1 Pull monthly returns
Start from crsp.msf and retrieve at least the following columns:

permno

permco

date

ret

retx

prc

shrout

Convert the date into an explicit calendar month identifier. If downstream code expects YYYYMM, do not use to_period('M').astype(int), because that produces a period ordinal rather than a human-readable year-month code. Use:

python
crsp['yyyymm'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month
1.2 Merge name-history attributes correctly
To obtain shrcd and exchcd, merge CRSP monthly returns to crsp.msenames using permno and the active-name interval.

In pseudocode:

python
# keep rows where namedt <= date <= nameendt
This is safer than merging to crsp.mse by exact date and avoids event-driven duplication.

1.3 Restrict the investable universe
Typical baseline filters include:

Common shares only: shrcd in {10, 11}.

Major exchanges only: exchcd in {1, 2, 3}.

Positive or sufficiently large price, often abs(prc) >= 5 for robustness checks.

Depending on the project, additional filters may exclude ADRs, REITs, closed-end funds, or other special security types.

1.4 Merge delisting returns
Merge crsp.msedelist onto the monthly return panel and adjust the final observed return whenever dlret is available:

python
ret_adjusted = (1 + ret) * (1 + dlret) - 1
If dlret is missing for a delisted firm, some studies apply fallback rules by delisting code. Whether such rules are used should be stated explicitly in the replication notes.

1.5 Aggregate market equity at the PERMCO level
For firms with multiple share classes, market equity should be aggregated across all PERMNOs that belong to the same PERMCO in a given month. Using abs(prc) * shrout at the individual PERMNO level can overstate ratios such as book-to-market for multi-class firms.

A standard approach is:

Compute security-level market equity.

Sum market equity across all PERMNOs within the same permco-month.

Assign the aggregated market equity back to the security that is kept in the final panel, often the largest-PERMNO security in that month.

This step should be documented clearly because it affects any characteristic that uses market equity in the denominator.

Step 2: Extract and Clean Compustat Fundamentals
2.1 Pull annual accounting variables
From comp.funda, retrieve the annual variables needed for the characteristic set, for example:

gvkey

datadate

fyear

at

ceq

sale

cogs

xsga

xint

ib

capx

sic

If the study excludes financial firms, keep sic or another industry code at this stage.

2.2 Exclude financial firms when appropriate
Many empirical asset pricing studies exclude financial firms because their balance sheet structure differs materially from industrial firms. A common filter is to remove SIC codes 6000-6999.

If the study keeps financials, the documentation should explain why. If the study removes them, that filter should be applied explicitly and consistently.

2.3 Compute annual characteristics at the accounting frequency
Variables that are naturally defined using annual accounting statements should be computed at the annual Compustat level before any monthly expansion.

For example, investment is better defined as:

i
n
v
t
=
a
t
t
−
a
t
t
−
1
a
t
t
−
1
inv 
t
​
 = 
at 
t−1
​
 
at 
t
​
 −at 
t−1
​
 
​
 
computed on annual Compustat observations by gvkey, not by applying shift(12) after merging to a monthly panel. Computing inv at the monthly panel level assumes a complete monthly sequence and can fail when months are missing.

Other annual characteristics, such as profitability or book equity, should likewise be formed before monthly mapping whenever possible.

Step 3: Link Compustat to CRSP
3.1 Use valid CCM links
Merge comp.funda to crsp.ccmxpf_linktable by gvkey, and keep only valid link types and primary links. A common choice is:

linktype in {'LU', 'LC'}

linkprim in {'P', 'C'}

Open-ended linkenddt values should be filled with a far-future date before applying the date filter.

3.2 Enforce link-date validity
Only retain accounting observations whose datadate falls inside the active CCM link window:

python
linkdt <= datadate <= linkenddt
This prevents invalid historical mappings and helps avoid duplicated firm identities.

3.3 Deduplicate linked records
After CCM linking, it is possible to obtain multiple accounting records for the same permno-month, especially when link intervals overlap or when accounting observations are mapped too aggressively.

The final linked accounting dataset should be deduplicated with an explicit rule. A common rule is:

for a given permno and target month, keep the most recent valid datadate that is already public by that month.

The deduplication rule should be documented because it directly affects panel uniqueness.

Step 4: Map Annual Accounting Data to Monthly Returns
This is the most important step for avoiding hidden logic errors.

4.1 Do not rely on annual forward-fill alone
Forward-filling Compustat observations within the annual accounting table does not generate missing months. If the accounting file has one row per firm-year, groupby('permno').ffill() only fills across existing annual rows and does not create intermediate monthly observations.

As a result, an exact merge on:

python
yyyymm == yyyymm_effective
will fail for most firm-months and produce mostly missing accounting variables.

4.2 Recommended approach: point-in-time monthly mapping
There are two valid implementations.

Option A: merge_asof
Sort both datasets by permno and date, then match each CRSP month to the most recent accounting observation whose effective date is less than or equal to the CRSP month. This is typically the cleanest approach in pandas when the logic is strictly point-in-time.

Option B: build a monthly grid first
Create a complete firm-month grid for each permno, place annual accounting values at the month when they become investable, then forward-fill within each permno, and finally merge the completed accounting panel to CRSP.

Both methods are valid, but one must be implemented explicitly. A plain forward-fill over annual rows is not sufficient.

4.3 Apply a reporting lag
To avoid look-ahead bias, accounting data must become usable only after a delay. A common conservative implementation is to make an annual observation available six months after datadate, though some projects instead use actual filing dates where available.

The monthly mapping therefore uses an effective date, for example:

python
effective_date = datadate + MonthEnd(6)
Then each month is matched to the latest accounting record with:

python
effective_date <= month_end
This ensures that month t only sees accounting information that was already public by the end of month t.

Step 5: Construct Characteristics
5.1 Market-based variables
Construct market-based variables such as:

market equity

lagged return

momentum

turnover or liquidity proxies

volatility measures

All of these should be built using only information available up to the end of the formation month.

5.2 Accounting-based variables
Construct annual characteristics such as:

book-to-market

operating profitability

investment

leverage

asset growth

These should be computed at the accounting frequency first, then mapped to the monthly panel using the effective-date logic above.

5.3 Momentum construction
Momentum signals should be based on past returns only, with the usual skip-month convention if required by the design. For example, a 12-2 momentum signal for month t uses returns from months t-12 through t-2, not t-1 or t+1.

5.4 Excess return target
If the prediction target is next-month excess return, merge the monthly risk-free rate and define:

r
x
t
+
1
=
r
t
+
1
−
r
f
t
+
1
rx 
t+1
​
 =r 
t+1
​
 −rf 
t+1
​
 
The documentation should state clearly whether the target is raw return, excess return, or excess return net of delisting adjustment.

Step 6: Handle Outliers, Missing Values, and Standardization
6.1 Winsorization
Outlier treatment should be done cross-sectionally within each month, not over the full sample. For example, each characteristic can be clipped at the 1st and 99th percentiles using only the firms available in that month.

Using full-sample quantiles introduces future information into earlier months and therefore creates leakage.

6.2 Missing values
Missing values can be handled in one of two ways:

Preprocessing-stage imputation: for example, cross-sectional median imputation by month.

Model-pipeline imputation: for example, SimpleImputer(strategy='median') fit only within the training window.

The choice should be stated explicitly. If the final modeling code already applies imputation inside a scikit-learn pipeline, the data-processing document should say that raw missingness is preserved until model fitting.

6.3 Cross-sectional standardization
If characteristics are standardized before modeling, the documentation should explain whether this occurs:

in the preprocessing stage, month by month, using cross-sectional z-scores or rank transforms; or

inside the model pipeline, where the transformer is fit only on the training window.

For empirical asset pricing, both approaches can be valid, but the implementation must be point-in-time consistent and not depend on full-sample moments.

Step 7: Construct the Final Firm-Month Panel
7.1 Enforce uniqueness
After all merges, verify that each permno-yyyymm pair appears at most once. If duplicates remain, trace them back to the relevant join and resolve them before modeling.

A useful invariant is:

python
panel.groupby(['permno', 'yyyymm']).size().max() == 1
7.2 Define sample start and end dates carefully
Even if CRSP extraction begins in 1970 and Compustat extraction begins earlier, the usable modeling sample will usually start later because:

accounting variables require a reporting lag,

some characteristics require prior history,

momentum or volatility signals need rolling windows,

delisting-adjusted return targets may require aligned auxiliary files.

For a monthly panel using annual accounting variables with a six-month lag and momentum-style predictors, the practical sample start may be around 1972-1973 rather than the raw earliest CRSP date.

The final documentation should therefore distinguish between:

raw extraction period,

post-merge coverage period,

final modeling sample period.

7.3 Export clean artifacts
Store at least two outputs:

a full panel file, such as firm_month_panel.parquet, with identifiers, dates, target, and raw characteristics;

a modeling-ready artifact, such as model_inputs.npz or another panel subset used in the machine learning experiments.

Step 8: Sanity Checks and Diagnostics
A complete preprocessing pipeline should include a short diagnostics section.

Recommended checks:

Number of observations before and after each major merge.

Number of unique firms per month through time.

Duplicate counts for permno-yyyymm after each merge.

Share of missing values by variable and by month.

Distributional summaries of key variables before and after winsorization.

Coverage comparison for CRSP-only versus CRSP+Compustat merged samples.

These diagnostics help identify silent problems such as duplicate inflation, broken month mapping, excessive data loss, or systematic missingness in certain periods.

Step 9: Time-Series-Safe Modeling Interface
The preprocessing layer should hand off data to the modeling layer in a form that preserves time ordering.

At minimum, provide:

X: feature matrix.

y: target vector.

time_index: month identifier.

optionally permno for portfolio sorting and firm-level diagnostics.

All model fitting, scaling, dimensionality reduction, and hyperparameter tuning should occur inside rolling or expanding windows, never on the full sample.

If the project supports both rolling and expanding windows, the documentation should state the rationale for each:

Rolling window: keeps the training sample length fixed and adapts more quickly to structural change.

Expanding window: uses all available history and may stabilize parameter estimates when signals are weak.

Step 10: Versioning and Reproducibility Notes
The documentation should record the exact data environment used for extraction and preprocessing, including:

WRDS platform used, such as WRDS Cloud.

CRSP product version, especially if using a post-migration WRDS environment where table names or field names may differ.

Extraction date or snapshot period.

Python version and key package versions.

This is especially important if the project may later be rerun under a different WRDS schema or CRSP release.

Suggested Minimal Output Schema
A practical final panel might contain the following columns:

Column	Description
permno	CRSP security identifier
permco	CRSP company identifier
gvkey	Compustat firm identifier
date	Month-end date
yyyymm	Calendar month in YYYYMM format
ret	Raw monthly return
retx	Monthly return excluding distributions
dlret	Delisting return, if available
ret_adj	Delisting-adjusted return
rf	Monthly risk-free rate
rx_fwd	Next-month excess return target
me_company	Company-level market equity
bm	Book-to-market
op	Operating profitability
inv	Investment / asset growth
mom12_2	Twelve-to-two momentum
sic	Industry code
Recommended Processing Order
Extract CRSP monthly returns.

Merge CRSP name-history data using date intervals.

Apply universe filters.

Merge delisting returns and build adjusted returns.

Compute security-level and company-level market equity.

Extract Compustat fundamentals.

Exclude financial firms if required.

Compute annual accounting characteristics.

Link Compustat to CRSP through CCM with valid date ranges.

Map annual accounting data to monthly observations using merge_asof or a monthly grid plus forward-fill.

Merge risk-free rate if excess returns are used.

Build market-based and accounting-based characteristics.

Apply cross-sectional outlier treatment.

Run diagnostics and deduplication checks.

Export the final panel and modeling inputs.

Summary
A credible asset-pricing preprocessing pipeline is not just a sequence of joins. It must preserve point-in-time information, avoid duplicate inflation, incorporate delisting outcomes, treat multi-class firms correctly, and map annual accounting information into the monthly panel using an explicit time-valid procedure. The most common silent failure is not in model estimation but in preprocessing logic, especially around date alignment, duplicate records, and leakage from future information.

Further Considerations / Extensions
This section summarizes additional methodological choices and edge cases that are often important in empirical asset pricing projects, especially when the final panel is used both for machine learning prediction and for more traditional portfolio-sort analysis.

Book equity construction
The main document refers to book-to-market but does not yet specify the full construction of book equity. In practice, Compustat ceq (common equity) is not identical to the book equity concept used in Fama-French style research. A standard construction in the literature follows Davis, Fama, and French (2000): stockholders' equity is chosen using a priority order, deferred taxes and investment tax credits are added when appropriate, and preferred stock is subtracted using a redemption-value, liquidation-value, then par-value hierarchy.

If the project uses a simplified definition such as BE = ceq, this should be stated explicitly and treated as a design choice rather than as the canonical Fama-French implementation. A more precise version of the documentation should therefore include the exact formula, the field-selection priority, and any fallback logic used when the preferred fields are missing.

Negative book equity
Some firms have negative book equity. In such cases, book-to-market becomes negative and its economic interpretation is ambiguous in both portfolio sorts and predictive models. A common treatment is to set BM to missing for firms with negative BE while retaining those firms in the panel for other characteristics and return observations.

The document should make this rule explicit. If a different choice is made, such as retaining negative BM values for machine learning but excluding them from value-sorted portfolio tests, the distinction should be documented as part of the variable-definition section.

Fama-French convention versus monthly mapping
The pipeline described above uses a flexible monthly mapping approach in which annual accounting variables become investable after a lag and are then carried forward until new information arrives. This design is well suited to machine learning prediction tasks because it creates a point-in-time monthly feature panel and can incorporate asynchronous information arrival more naturally.

This differs from the traditional Fama-French convention, in which book-to-market is typically formed using June market equity and the latest available book equity, and that BM signal is then held fixed from July of year 
t
t through June of year 
t
+
1
t+1. The documentation should explicitly note that the monthly mapping is a methodological choice for prediction and that resulting portfolio-sort outputs will not be numerically identical to classical Fama-French sorts.

Annual versus quarterly Compustat
The current workflow is based on annual Compustat data (comp.funda). That is a valid design choice, but it should be described as such. Many recent empirical and machine-learning studies also use quarterly Compustat data (comp.fundq) to capture more timely fundamental changes, especially around earnings information and post-announcement dynamics.

If quarterly data are added in the future, the preprocessing logic must handle several additional complications: multiple observations within the same fiscal year, higher missingness, differing announcement lags, and potentially more complex point-in-time alignment rules. The document can therefore state that annual data are used in the baseline pipeline and treat quarterly integration as an extension.

Fixed six-month lag versus actual report dates
Using datadate + 6 months as the availability date for annual accounting information is conservative and widely defensible, but it is not the most precise point-in-time alignment possible. For quarterly data, Compustat provides rdq, and many studies also rely on IBES or filing-date sources when announcement timing is central to the design.

If the project prioritizes simplicity and conservative alignment, the fixed-lag rule is acceptable as a baseline. If the goal is tighter event-time precision, the documentation should explain whether actual report dates, filing dates, or point-in-time backtest packages are used instead of a fixed lag, and why that choice was made.

Treatment of extreme target returns
The current document explains outlier handling for characteristics but does not yet specify how extreme values of the target variable are treated. This matters because very large positive rebounds or near-total losses from distress and delisting can exert disproportionate influence on squared-loss models such as OLS and many neural-network objectives.

The documentation should therefore state whether next-month returns are kept in raw form, winsorized, truncated, transformed to ranks, or handled differently across model classes. If raw returns are retained for economic fidelity, that choice should be justified explicitly, especially when the pipeline already merges delisting returns into the target.

Winsorization and imputation order
The order of winsorization and imputation should be explicit. In most empirical asset-pricing pipelines, cross-sectional winsorization is performed first on the observed values within each month, while imputation is deferred to the model pipeline and fit only on the training window. This avoids leaking future information through full-sample imputers and keeps the raw missingness pattern intact until model estimation.

If the project instead performs imputation during preprocessing, the document should state whether imputation happens before or after clipping and whether the imputation is monthly, rolling, or full-sample. The key requirement is that the order and timing be fully transparent and point-in-time safe.

Explicit lag definitions for market variables
The document already defines the skip-month convention for 12-2 momentum, but other market-based variables should be documented with the same precision. Examples include last-month reversal, trailing volatility from daily returns, turnover over the past 
N
N months, beta estimation windows, and liquidity proxies.

For each such variable, the documentation should specify: the source frequency, the lookback window, whether the most recent month is included or skipped, and the exact month in which the signal becomes tradable. This makes the feature set easier to audit and reduces the risk of hidden look-ahead errors.

Nonlinear transformations such as logs
Many firm characteristics are highly skewed. Variables such as market equity, total assets, and occasionally book-to-market are often log-transformed before entering the model. If the pipeline uses log(ME), log(AT), or another monotonic nonlinear transformation, this should be described explicitly in the characteristic-construction section.

If no such transformations are used, that too is worth stating, because downstream model behavior can differ materially depending on whether the inputs are raw levels, ratios, ranks, or logs. In machine-learning settings, this choice can matter for both numerical stability and interpretability.

Macro and market-level variables
The current document focuses on firm-level predictors. If the final models also use macro or market-wide conditioning variables, such as market excess return, term spread, default spread, dividend yield, or volatility indices, those inputs should be documented separately.

For each macro variable, the documentation should state the data source, update frequency, release timing, and how it is aligned with the firm-month panel. Point-in-time alignment matters here as well, because many aggregate series are revised or released with lags.

CRSP versioning
The note on CRSP version differences can be made more concrete. WRDS has been migrating users toward newer CRSP schemas, and table names or field names may differ across legacy and newer environments. In some WRDS environments, monthly stock data may no longer be accessed under the same legacy table names used in older codebases.

The documentation should therefore record the exact CRSP schema used for development and extraction, and, if possible, include a short compatibility note for users working under a newer WRDS/CRSP environment. This is especially important for open-source replication repositories where others may not have the same database version.

Unbalanced panel structure
The final firm-month panel is naturally unbalanced because firms list, delist, merge, and disappear at different times. This is normal in empirical asset pricing, but it should be stated explicitly. Some downstream methods, such as fixed-effects panel regressions or sequence models, may implicitly require a minimum length of firm history.

If the project imposes any minimum-history rule, such as requiring at least 
N
N valid months per firm before entering the modeling sample, that threshold should be documented. If no such rule is used, the documentation can note that the panel remains fully unbalanced by design.

Portfolio-sort breakpoints
If the downstream analysis includes portfolio sorts, the preprocessing outputs may need additional columns or flags beyond the prediction-ready feature matrix. In particular, size or book-to-market portfolio formation often uses NYSE-only breakpoints rather than breakpoints based on the full universe.

The documentation should therefore indicate whether NYSE membership flags are retained in the final panel and whether breakpoint assignment is part of preprocessing or a later analysis step. This helps ensure that the panel supports both machine-learning prediction and classical long-short portfolio construction.

Summary of these extensions
These extensions do not overturn the main preprocessing pipeline; rather, they refine the variable definitions, clarify methodological choices, and document edge-case treatments more fully. The most important additions concern precise book-equity construction, explicit choices about timing conventions, treatment of extreme returns, and clear support for both predictive modeling and portfolio-sort analysis.

Additional Implementation Details
The following implementation notes focus on edge cases that do not change the overall logic of the preprocessing pipeline but materially improve auditability, reproducibility, and error detection in actual code.

CRSP missing return encodings
CRSP return fields can contain non-standard missing representations depending on the extraction environment, data vintage, or export path. The preprocessing code should therefore document how raw return values are normalized immediately after import and whether non-numeric or special missing codes are mapped to NaN before feature construction.

Those normalized missing returns should then be handled consistently in rolling features such as momentum, volatility, and turnover. In particular, they should not be silently converted to zero, because doing so changes both the effective signal and the sample-count diagnostics.

Why CRSP prices can be negative
The document already uses abs(prc) when computing market equity, but it is useful to explain why. In CRSP, a negative price does not indicate a negative economic price; rather, it indicates that the reported value is based on a bid-ask average rather than an observed closing transaction price.

This convention should be mentioned explicitly so that later users do not treat negative prc values as data corruption. The standard market-equity calculation therefore uses abs(prc) * shrout, subject to the project's security-level and company-level aggregation rules.

Minimum valid observations for rolling variables
Rolling market-based predictors should specify minimum data requirements. For example, a 12-2 momentum signal may use returns from months 
t
−
12
t−12 through 
t
−
2
t−2, but the documentation should also state how many valid monthly returns must be present before the signal is computed.

A practical rule is to require a minimum number of non-missing observations within the lookback window, such as at least 8 valid months for momentum, while leaving the exact threshold as a project-level design choice. The same logic applies to rolling volatility, turnover, beta, and related window-based variables.

Return aggregation convention for momentum
The momentum section should state the exact aggregation formula. A standard compounded-return definition is:

e
x
t
m
o
m
12
−
2
,
t
=
∏
k
=
2
12
(
1
+
r
t
−
k
)
−
1
extmom 
12−2,t
​
 = 
k=2
∏
12
​
 (1+r 
t−k
​
 )−1
An equivalent implementation can sum log returns over the same window and then exponentiate back to simple-return space. This should be stated explicitly because compounded returns, log-return accumulation, and naive arithmetic summation are not identical in the tails.

Forward target alignment and end-of-panel truncation
The document should state explicitly that the forward target on the row dated month 
t
t corresponds to the realized return in month 
t
+
1
t+1. In code, this is often implemented through a forward shift, but the implementation should be validated by checking that the target on month 
t
t matches the realized adjusted return minus the risk-free rate observed in month 
t
+
1
t+1.

The document should also note that the last available month for each firm cannot have a valid forward target by construction. Those end-of-history rows should therefore be excluded from the supervised-learning sample once the target variable has been formed.

Delisting months and target construction
If a firm delists in month 
t
+
1
t+1, then the observation at month 
t
t should use the delisting-adjusted forward return as its target. However, the delisting month itself typically does not remain in the final supervised-learning sample because there is no valid 
t
+
2
t+2 return to serve as the next-period target.

This interaction should be documented explicitly because it is easy to implement delisting adjustment correctly at the return level but still keep the wrong rows in the modeling sample.

Timing of the market-equity denominator in BM
The definition of book-to-market should specify the market-equity denominator timing. One choice is to pair book equity with market equity measured at the accounting-reference month or another fixed formation date, which is closer to the traditional Fama-French convention.

Another choice is to divide lagged book equity by contemporaneous monthly market equity, which creates a stale-book / fresh-market signal. Because these alternatives are economically different and can produce different portfolio sorts and predictive behavior, the chosen definition should be stated unambiguously.

Potential Compustat backfill-related selection issues
Compustat historical coverage can create potential selection concerns in some empirical designs because a firm may appear in the database with historical fundamentals that were not necessarily available in the same practical form to researchers at earlier dates. A project can optionally impose a minimum Compustat history requirement before a firm enters the usable accounting sample, but this should be treated as an optional robustness rule rather than as a universal requirement.

If such a rule is used, the documentation should specify the exact threshold, such as requiring the firm to have appeared in Compustat for more than one annual observation before its characteristics enter the panel. If no such rule is used, that should also be stated explicitly.

Short fiscal years and annual growth variables
Annual growth variables such as investment or asset growth can be distorted by short fiscal years and accounting-period changes. If the project uses these variables, the documentation should explain whether short fiscal periods are filtered out, identified through duration fields when available, or screened indirectly through the time spacing of consecutive datadate observations.

This matters because a simple year-over-year ratio may not have the intended economic interpretation if the two accounting observations do not correspond to roughly comparable reporting intervals.

Timing of low-price screens
If the project applies a low-price filter such as abs(prc) >= 5, the documentation should state when that filter is applied. A monthly investability screen, in which the filter is re-applied every month, is different from a formation-only screen used in portfolio sorts.

For monthly machine-learning prediction tasks, a monthly investability rule is often the more natural implementation. However, because this choice can affect the tail behavior of small-cap and distressed stocks, the rule should be documented explicitly rather than left implicit.

External benchmark validation
The diagnostics section can be strengthened by adding an external validation step in addition to internal consistency checks. A practical approach is to form simple size, value, or momentum sorted portfolios from the processed panel and compare their time-series behavior with public Kenneth French benchmark series.

The goal is not to reproduce the benchmark exactly under every design choice, but to ensure that the resulting signals have the expected direction and high correlation with public references when the methodology is intended to be comparable. If a self-constructed value or momentum long-short series correlates much less than expected with the corresponding benchmark, that often signals a problem in variable construction, date alignment, or sample filters.

Time-safe PCA and factor extraction
The time-series-safety principle should extend beyond scaling and imputation to any multivariate transformation such as PCA or latent-factor extraction. If downstream models use cross-sectional PCA, those transformations must be fit only on the training window and re-estimated at each rolling or expanding iteration rather than being learned once on the full sample.

This requirement is especially important because the covariance structure used by PCA depends on the full sample of inputs and can leak future information more subtly than univariate scaling.

Output ordering and indexing
The final exported panel should use a documented sorting convention. For example, sorting by permno, yyyymm is usually convenient for security-level rolling operations and grouped feature construction, while sorting by yyyymm, permno may be more convenient for cross-sectional monthly analysis.

The documentation should state the default ordering used in exported artifacts and, if relevant, whether a multi-index is set in memory before output. A consistent ordering rule improves reproducibility, merge stability, and debugging performance in downstream code.
