# Data Processing Specification: Dual-Channel Fund Crisis Early Warning System

**Document Version:** 2.0  
**Last Updated:** 2026-05  
**Repository:** `dual-channel-crisis-ews/docs/data_processing_specification.md`

---

## Table of Contents

1. [Document Overview](#document-overview)
2. [Stage 0: Infrastructure, Metadata Standards, and Regime Definitions](#stage-0-infrastructure-metadata-standards-and-regime-definitions)
3. [Stage 1: Fund Market Data Extraction and Cleaning (CRSP)](#stage-1-fund-market-data-extraction-and-cleaning-crsp)
4. [Stage 2: Holdings Data Extraction and Regime-Stratified Processing](#stage-2-holdings-data-extraction-and-regime-stratified-processing)
5. [Stage 3: Text Data Extraction and NLP Inference](#stage-3-text-data-extraction-and-nlp-inference)
6. [Stage 4: Macro and Structural Fragility Indicator Construction](#stage-4-macro-and-structural-fragility-indicator-construction)
7. [Stage 5: Signal Computation — Holdings Channel](#stage-5-signal-computation--holdings-channel)
8. [Stage 6: Signal Computation — Text Channel](#stage-6-signal-computation--text-channel)
9. [Stage 7: Crisis Label Construction](#stage-7-crisis-label-construction)
10. [Stage 8: Mixed-Frequency Alignment and Master Panel Construction](#stage-8-mixed-frequency-alignment-and-master-panel-construction)
11. [Stage 9: Analysis-Phase-Specific Data Products](#stage-9-analysis-phase-specific-data-products)
12. [Stage 10: Standardization, Missing Values, and Outlier Treatment](#stage-10-standardization-missing-values-and-outlier-treatment)
13. [Stage 11: Diagnostics, Validation, and Quality Assurance](#stage-11-diagnostics-validation-and-quality-assurance)
14. [Stage 12: Version Control and Reproducibility](#stage-12-version-control-and-reproducibility)
15. [Appendix A: Complete Final Panel Schema](#appendix-a-complete-final-panel-schema)
16. [Appendix B: Processing Order Summary](#appendix-b-processing-order-summary)
17. [Appendix C: Key Design Choices Checklist](#appendix-c-key-design-choices-checklist)
18. [Appendix D: Mapping to Proposal Research Questions and Hypotheses](#appendix-d-mapping-to-proposal-research-questions-and-hypotheses)

---

## Document Overview

This document provides a complete, stage-by-stage specification for all data processing in the Dual-Channel Fund Crisis Early Warning System project. The project fuses two information channels—portfolio holdings (private actions) and textual disclosures/news (public narratives)—to predict mutual fund crises across multiple regulatory regimes and crisis episodes.

The processing pipeline is organized into thirteen sequential stages (Stage 0 through Stage 12). Each stage produces well-defined intermediate artifacts that serve as inputs to subsequent stages. The design is governed by three overarching principles:

**Principle 1: Strict Point-in-Time Discipline.** No feature used for predicting a crisis at time \(t+h\) may incorporate information that was not publicly available by time \(t\). For SEC filings, the information availability date is the filing date (the date SEC accepted the document), not the reporting period end date.

**Principle 2: Regime Awareness.** The U.S. mutual fund disclosure landscape underwent two major structural breaks—the introduction of Form N-Q in 2004 and the transition to Form N-PORT in 2019. Each stage explicitly documents how it handles cross-regime comparability.

**Principle 3: Separation of Predictive and Diagnostic Signals.** Signals that require future-realized data (e.g., holdings-based returns) are strictly segregated from point-in-time predictive features. They serve only as ex post diagnostic tools for mechanism validation.

---

## Stage 0: Infrastructure, Metadata Standards, and Regime Definitions

### 0.1 Purpose and Rationale

Before any data extraction begins, this stage establishes the foundational definitions that all subsequent stages reference. Without these standards, inconsistencies in time alignment, identifier resolution, and sample boundaries would propagate throughout the pipeline and potentially introduce look-ahead bias—the most serious methodological threat to any predictive study.

### 0.2 Global Time Alignment Rules

The core anti-leakage principle requires a precise timestamp hierarchy for every piece of information entering the system:

| Timestamp Layer | Definition | Example |
|---|---|---|
| `report_date` | The end date of the period covered by a filing | 2019-12-31 |
| `filed_date` | The date SEC accepted the filing (appears on EDGAR) | 2020-02-28 |
| `effective_date` | Identical to `filed_date` — the moment information becomes public | 2020-02-28 |
| `effective_month` | The calendar month-end containing the `effective_date` | 202002 |
| `usable_from_month` | The first month for which this signal can serve as a prediction feature | 202003 |

The logic is straightforward: a signal becomes "known" in its `effective_month`. When constructing a prediction for month \(t+1\), all features with `effective_month <= t` are eligible. This ensures that even within the prediction month itself, no future information leaks in.

For news articles, the `filed_date` equivalent is the publication timestamp. For CRSP market data, monthly observations for month \(t\) become available at the close of the last trading day of month \(t\), and are thus usable for predictions targeting \(t+1\) onward.

### 0.3 Data Source Metadata Standard

Every observation entering the final panel must carry the following metadata fields. These fields enable post hoc auditing of information timing and provenance:

| Field | Purpose |
|---|---|
| `source_form` | Identifies the regulatory form or data type (N-PORT, N-Q, N-CSR-SOI, 13F, N-CSR-TEXT, NEWS) |
| `report_date` | The reporting period end date of the underlying document |
| `filed_date` | The SEC filing date or news publication date |
| `effective_month` | The first full month in which this signal is considered public knowledge |
| `coverage_level` | Granularity: fund, family, category, or market-wide |
| `regime_flag` | Regulatory regime identifier (pre-2004, NQ-era, NPORT-era) |
| `match_quality` | For linked data: exact, fuzzy, manual, or failed |

### 0.4 Regulatory Regime Definitions

**Regime 1: 13F Era (Exploratory, 1999–2003)**

During this period, no fund-level quarterly holdings disclosure was mandated. The only systematic source of portfolio information is Form 13F-HR, filed quarterly by institutional investment managers with over $100 million in Section 13(f) securities. The critical limitation is that 13F reports at the management company level (not individual fund level) and covers only U.S.-listed equities. This regime is used exclusively for exploratory analysis of the Dot-com crisis window.

**Regime 2: N-Q Era (Primary for GFC, 2004–February 2019)**

The SEC introduced Form N-Q in 2004, requiring registered investment companies to file complete portfolio schedules for the first and third fiscal quarters. Combined with the Schedule of Investments (SOI) already required within N-CSR (semi-annual and annual reports), this provided quarterly fund-level holdings disclosure for the first time. The data format is HTML/text-based, requiring custom parsing pipelines with non-trivial error rates.

**Regime 3: N-PORT Era (Primary for COVID, March 2019–Present)**

Form N-PORT replaced N-Q beginning in March 2019, requiring monthly portfolio reporting in structured XML format. This represents a quantum leap in data quality and frequency: standardized fields for liquidity classification, derivative exposures with delta reporting, and machine-readable formatting. However, an important nuance affects public availability: only the third month of each quarter's filing is immediately public; filings for months one and two are released with a delay.

---

## Stage 1: Fund Market Data Extraction and Cleaning (CRSP)

### 1.1 Purpose and Rationale

This stage extracts the foundational fund-level time series—returns, total net assets, and flows—from the CRSP Mutual Fund Database. These variables serve three critical roles: they define the prediction targets (drawdowns and outflows), they provide key predictive features (recent performance and flow momentum), and they supply the denominators and scaling factors needed for computing normalized signals in later stages.

### 1.2 Data Extraction

From the CRSP Mutual Fund Database (accessed via WRDS), extract the following monthly fields:

**Return and asset fields:** `crsp_fundno`, `crsp_portno`, `caldt`, `mret`, `mtna`, `mnav`

**Expense and activity fields:** `exp_ratio`, `turn_ratio`

**Classification fields:** `fund_name`, `mgmt_name`, `lipper_class`, `crsp_obj_cd`, `si_obj_cd`

**Asset allocation fields:** `per_com`, `per_pref`, `per_conv`, `per_corp`, `per_muni`, `per_govt`, `per_oth`, `per_cash`

**Time range:** January 1997 through the most recently available month.

### 1.3 Share Class to Portfolio Level Aggregation

Since the analytical unit of interest is the investment portfolio rather than the share class, all share-class-level data must be aggregated to the portfolio level.

**Total Net Assets:** Sum across all share classes within the same `crsp_portno` and month.

**Returns:** Compute a TNA-lagged-weighted average return:

\[
r_{p,t} = \frac{\sum_{s \in p} r_{s,t} \times \text{TNA}_{s,t-1}}{\sum_{s \in p} \text{TNA}_{s,t-1}}
\]

**Auxiliary attributes:** For fields that cannot be meaningfully aggregated, retain the values from the largest share class by TNA within each portfolio-month.

**Record aggregation metadata:** Store `share_class_count` for each portfolio-month.

### 1.4 Fund Flow Calculation

**Raw dollar flow:**

$$
\text{Flow}^{\$}_{i,t} = \text{TNA}_{i,t} - \text{TNA}_{i,t-1} \times (1 + r_{i,t})
$$

**Percentage flow:**

$$
\text{flow}_{i,t} = \frac{\text{Flow}^{\$}_{i,t}}{\text{TNA}_{i,t-1}}
$$

### 1.5 Example Python Snippet: Portfolio-Month Identifier

```python
fund["yyyymm"] = fund["caldt"].dt.year * 100 + fund["caldt"].dt.month
```

### 1.6 Stage 1 Output Artifacts

```text
artifacts/stage1/
├── fund_month_panel_raw.parquet
├── fund_month_panel_portfolio.parquet
├── fund_month_panel_classified.parquet
├── fund_death_records.parquet
└── stage1_diagnostics.json
```

---

## Stage 2: Holdings Data Extraction and Regime-Stratified Processing

### 2.1 Purpose and Rationale

This stage extracts the raw security-level portfolio holdings that form the basis of the holdings channel. Because the regulatory disclosure framework changed dramatically over the sample period, this stage must handle three distinct filing types with different formats, frequencies, coverage, and quality characteristics.

### 2.2 N-PORT Extraction (March 2019 – Present)

**Data source:** SEC EDGAR bulk N-PORT-P XML filings, accessed either through direct EDGAR download or via WRDS SEC Analytics Suite.

**Extraction fields (per fund × month × security):**

| Field | Description | Notes |
|---|---|---|
| `registrant_cik` | CIK of the registered investment company | Links to fund family |
| `series_id` | Series identifier within the registrant | Links to specific fund |
| `report_date` | Reporting period end date | Usually month-end |
| `filed_date` | Date SEC accepted the filing | Typically `report_date + 55–65 days` |
| `cusip` | 9-digit CUSIP identifier | Standard issue-level identifier |
| `issuer_name` | Name of security issuer | Backup for matching |
| `title` | Security title/description | Useful for classification |
| `asset_type` | Asset category | Standardized in XML schema |
| `quantity` | Number of shares/units held |  |
| `value` | Fair market value in USD | Basis for weight computation |
| `pct_val` | Percentage of net assets | Cross-check with computed weight |
| `liquidity_class` | Liquidity classification | N-PORT-exclusive field |
| `is_derivative` | Boolean derivative flag |  |
| `derivative_type` | Derivative type | If applicable |
| `delta` | Option/derivative delta | If applicable |
| `notional_amount` | Notional value | If applicable |

**Public availability rules:**

```python
def get_nport_public_available_date(report_month_position_in_fiscal_quarter, filed_date, filed_date_of_third_month_filing):
    if report_month_position_in_fiscal_quarter == 3:
        return filed_date
    return filed_date_of_third_month_filing
```

**Design choice:** The baseline implementation uses all monthly N-PORT filings with `public_available_date` as the effective date. A robustness variant restricts to immediately public third-month filings only.

### 2.3 N-Q and N-CSR Schedule of Investments Extraction (2004 – February 2019)

**Data source:** SEC EDGAR filings parsed through a T2MD pipeline that converts HTML and text-formatted portfolio schedules into structured tabular data.

Critical differences from N-PORT:

- No standardized liquidity classification.
- No standardized derivative reporting.
- Higher parsing error rates.

**Parsing quality control:**

Parsing quality control:

$$
\text{ParseQualityRatio} = \frac{\sum_j \text{parsedValue}_j}{\text{reportedNAV} \times \text{sharesOutstanding}}
$$

If this ratio deviates from 1.0 by more than 5%, flag the filing as `parse_quality = "low"`. If the deviation exceeds 20%, mark it as `parse_quality = "failed"`.

### 2.4 13F Extraction (1999 – Present, Exploratory Sample Only)

Because of its limitations, 13F data cannot enter the fund-level prediction panel on equal footing with N-Q or N-PORT data. Its role is strictly limited to:

- Exploratory event-study analysis for the Dot-com crisis window.
- Family-level robustness checks of concentration and herding patterns.
- Auxiliary estimation of passive ownership shares for the Fragility Index.

### 2.5 CUSIP-to-PERMNO Linkage

```python
def standardize_cusip(cusip: str) -> str | None:
    if cusip is None:
        return None
    cusip = str(cusip).strip().upper().replace(" ", "")
    if len(cusip) >= 8:
        return cusip[:8]
    return None
```

**Date-aware mapping rule:**

For each holding with CUSIP `c` observed at `report_date d`, search CRSP security name history for records satisfying:

- `ncusip == c`
- `namedt <= d <= nameendt`

### 2.6 Stage 2 Output Artifacts

```text
artifacts/stage2/
├── holdings_nport/
│   ├── holdings_nport_security_level.parquet
│   ├── holdings_nport_matched.parquet
│   └── nport_filing_metadata.parquet
├── holdings_nq/
│   ├── holdings_nq_security_level.parquet
│   ├── holdings_nq_matched.parquet
│   └── nq_filing_metadata.parquet
├── holdings_13f/
│   ├── holdings_13f_family_level.parquet
│   ├── holdings_13f_matched.parquet
│   └── 13f_filing_metadata.parquet
├── linkage_diagnostics/
│   ├── cusip_match_rates_by_regime.json
│   ├── unmatchable_assets_log.parquet
│   └── derivative_coverage_summary.json
└── stage2_diagnostics.json
```

---

## Stage 3: Text Data Extraction and NLP Inference

### 3.1 Purpose and Rationale

This stage processes two distinct text data streams: financial news and fund regulatory filings. The key methodological challenge is ensuring that FinBERT transfers effectively to the specific domain of mutual fund shareholder communications.

### 3.2 Pilot Validation Workflow

Before full-scale NLP processing, the project requires a validation study to assess FinBERT's applicability to fund-specific text.

### 3.3 Filing-Date Alignment for Report Text

```python
text_report_date = filing["report_period_end"]      # e.g., 2019-12-31
text_filed_date = filing["sec_acceptance_date"]     # e.g., 2020-03-02
text_effective_month = text_filed_date.year * 100 + text_filed_date.month
```

This signal is usable as a known feature starting in `text_effective_month`.

### 3.4 Stage 3 Output Artifacts

```text
artifacts/stage3/
├── pilot_validation/
│   ├── pilot_sample_metadata.csv
│   ├── human_annotations_annotator1.csv
│   ├── human_annotations_annotator2.csv
│   ├── finbert_predictions_pilot.csv
│   ├── pilot_evaluation_report.json
│   └── decision_log_pilot.md
├── news/
│   ├── news_daily_market.parquet
│   ├── news_daily_fund_linked.parquet
│   ├── news_weekly_aggregated.parquet
│   ├── news_monthly_aggregated.parquet
│   └── news_coverage_diagnostics.json
├── fund_reports/
│   ├── report_filing_metadata.parquet
│   ├── report_sentiment_by_section.parquet
│   ├── report_keywords_by_group.parquet
│   ├── report_sentiment_changes.parquet
│   └── report_text_quality_flags.parquet
└── stage3_diagnostics.json
```
## Stage 4: Macro and Structural Fragility Indicator Construction

### 4.1 Purpose and Rationale

This stage constructs the macro-financial and structural fragility variables that capture the broader environment in which funds operate. These variables serve two roles: (i) as control covariates to disentangle fund-specific behavior from market-wide stress and (ii) as inputs to a structural fragility index that proxies for the vulnerability of the ecosystem to passive flows and index concentration.

All macro variables must respect point-in-time constraints by incorporating appropriate publication lags.

### 4.2 Data Sources

The baseline implementation uses the following public data sources (via FRED or equivalent vendors):

- **Volatility:** CBOE Volatility Index (VIX).
- **Credit Spread:** Moody’s BAA Corporate Bond Yield minus 10-Year Treasury Constant Maturity.
- **Yield Curve Slope:** 10-Year Treasury Constant Maturity minus 2-Year Treasury.
- **Passive AUM:** Time series of passive mutual fund and ETF assets under management (e.g., from ICI or Morningstar).
- **Index Concentration:** Herfindahl-Hirschman Index (HHI) of major equity indices’ constituent weights (e.g., S&P 500).

Each time series is downloaded at a **daily or monthly frequency** and then aggregated to the **end-of-month** frequency consistent with the fund-month panel.

### 4.3 Publication Lag and Point-in-Time Alignment

For each macro series \(x_t\), the following alignment rule is applied:

- If \(x_t\) is available contemporaneously (e.g., VIX closing value on the last trading day of month \(t\)), then \(x_t\) is eligible for prediction of month \(t+1\) onward.
- If a series is published with delay (e.g., macro releases that appear in the first week of month \(t+1\)), a one-month lag is imposed to guarantee that the value used for month \(t\) was known by the end of month \(t\).

Implementation sketch:

```python
macro["yyyymm"] = macro["date"].dt.year * 100 + macro["date"].dt.month
macro_monthly = (
    macro.groupby("yyyymm")
         .agg({
             "vix": "last",
             "credit_spread": "last",
             "yield_slope": "last",
             "passive_aum": "last",
             "index_hhi": "last",
         })
         .reset_index()
)

# Impose one-month lag where necessary
macro_monthly[["vix", "credit_spread", "yield_slope"]] = \
    macro_monthly[["vix", "credit_spread", "yield_slope"]].shift(1)
```

The lagged values are then merged onto the `(fund_id, yyyymm)` backbone in Stage 8.

### 4.4 Structural Fragility Index Construction

The structural fragility index is designed to capture the interaction between the growth of passive investing and the concentration of benchmark indices.

Baseline specification:

$$
\text{FragilityIdx}_t = z\left(\Delta \log(\text{PassiveAUM}_t)\right) + z\left(\text{IndexHHI}_t\right)
$$

where $z(\cdot)$ denotes a standardized version of the series (mean 0, unit variance) computed over the historical window up to time $t$.

Implementation:

where \(z(\cdot)\) denotes a standardized version of the series (mean 0, unit variance) computed over the historical window up to time \(t\).

Implementation:

```python
macro_monthly["passive_aum_growth"] = (
    np.log(macro_monthly["passive_aum"]) -
    np.log(macro_monthly["passive_aum"].shift(12))
)

# Standardize within-sample up to each month (recursive z-score)
mean_aum = macro_monthly["passive_aum_growth"].expanding().mean()
std_aum  = macro_monthly["passive_aum_growth"].expanding().std().replace(0, 1.0)

mean_hhi = macro_monthly["index_hhi"].expanding().mean()
std_hhi  = macro_monthly["index_hhi"].expanding().std().replace(0, 1.0)

macro_monthly["fragility_idx"] = (
    (macro_monthly["passive_aum_growth"] - mean_aum) / std_aum +
    (macro_monthly["index_hhi"] - mean_hhi) / std_hhi
)
```

### 4.5 Stage 4 Output Artifacts

```text
artifacts/stage4/
├── macro_monthly_raw.parquet
├── macro_monthly_lagged.parquet
├── fragility_index_monthly.parquet
└── stage4_diagnostics.json
```

---

## Stage 5: Signal Computation — Holdings Channel

### 5.1 Purpose and Rationale

Stage 2 delivers security-level holdings snapshots across regimes. Stage 5 aggregates these holdings into fund-level signals that describe concentration, liquidity exposure, crowding, and tail-risk characteristics. A strict distinction is maintained between:

- **Predictive signals**: computable using information available at or before month \(t\).
- **Ex post diagnostic signals**: require future return realizations and are used only for mechanism validation (not as predictors).


### 5.2 Portfolio Weights and Basic Aggregation

For each fund $i$, month $t$, and security $j$ with market value $v_{ijt}$:

$$
w_{i,j,t} = \frac{v_{i,j,t}}{\sum_{k} v_{i,k,t}}
$$

Using these weights, all security-level characteristics $x_{j,t}$ are aggregated to the fund level as:

$$
x_{i,t}^{port} = \sum_j w_{i,j,t} \, x_{j,t}
$$
### 5.3 Concentration Measures

**Herfindahl-Hirschman Index (HHI):**

$$
\text{HHIPort}_{i,t} = \sum_j w_{i,j,t}^2
$$

**Top 10 Holdings Share:**

1. Sort securities by $w_{i,j,t}$ in descending order.
2. Sum the top 10 weights:

$$
\text{Top10Share}_{i,t} = \sum_{j \in \text{Top 10}} w_{i,j,t}
$$
### 5.4 Liquidity Exposure: Portfolio Amihud

Using CRSP stock-level daily data, compute the daily Amihud illiquidity measure for each stock $j$:

$$
\text{Illiq}_{j,d} = \frac{|r_{j,d}|}{\text{DollarVolume}_{j,d}}
$$

Then average over all trading days in month $t$:

$$
\text{Illiq}_{j,t} = \frac{1}{D_t} \sum_{d \in t} \text{Illiq}_{j,d}
$$

Portfolio-level Amihud:

$$
\text{AmihudPort}_{i,t} = \sum_j w_{i,j,t} \, \text{Illiq}_{j,t}
$$

By construction, this measure is known at the end of month $t$ and can be used to predict $t+1$ onward.

### 5.5 Crowding / Similarity: Category Overlap

For each category $c$ and month $t$, define the category-average portfolio:

$$
\bar{w}_{c,j,t} = \frac{1}{N_{c,t}} \sum_{i \in c} w_{i,j,t}
$$

Compute the cosine similarity between fund $i$’s weight vector $w_{i,\cdot,t}$ and the category-average vector $\bar{w}_{c,\cdot,t}$:

$$
\text{OverlapCat}_{i,t} =
\frac{\sum_j w_{i,j,t} \bar{w}_{c,j,t}}
     {\sqrt{\sum_j w_{i,j,t}^2} \sqrt{\sum_j \bar{w}_{c,j,t}^2}}
$$

This measures how “crowded” a fund is relative to its peers.

### 5.6 Tail-Risk Exposure: Portfolio MES (Marginal Expected Shortfall)

Using stock-level daily returns $r_{j,d}$ and a chosen market factor $r_{M,d}$ (e.g., S&P 500):

1. Estimate each stock’s MES as:

$$
\text{MES}_{j} = \mathbb{E}\left[ r_{j,d} \,\middle|\, r_{M,d} \leq q_{0.05}(r_{M}) \right]
$$

where $q_{0.05}(r_M)$ is the 5% left-tail quantile of the market return.

2. Aggregate to the portfolio:

$$
\text{MESPort}_{i,t} = \sum_j w_{i,j,t} \, \text{MES}_j
$$

MES is estimated using a rolling window of daily returns up to month $t$, ensuring point-in-time validity.

### 5.7 Turnover and Turnover Shock

If reported turnover ratios are available at year or semi-annual frequency:

- **Level:** Use reported `turn_ratio` mapped to the corresponding months.
- **Quarter-over-quarter change:**

$$
\text{TurnoverChg}_{i,t} = \text{Turnover}_{i,t} - \text{Turnover}_{i,t-3}
$$

A large positive value indicates an unusual surge in trading activity.

### 5.8 Ex Post Diagnostics: Holdings-Based Return and Return Gap

These variables are **strictly diagnostic** and must never be included in the predictive feature set.

1. **Holdings-Based Return (HBR):**

Use end-of-month weights $w_{i,j,t}$ and next-month stock returns $r_{j,t+1}$:

$$
\text{HBRExPost}_{i,t+1} = \sum_j w_{i,j,t} \, r_{j,t+1}
$$

2. **Return Gap:**

$$
\text{ReturnGapExPost}_{i,t+1} = r_{i,t+1}^{\text{reported}} - \text{HBRExPost}_{i,t+1}
$$

These ex post measures are used to study whether “abnormal” trading or unobserved positions explain deviations between reported returns and holdings-implied returns.

### 5.9 Stage 5 Output Artifacts
```text
artifacts/stage5/
├── holdings_signals_monthly.parquet
├── holdings_signals_ex_post_diagnostics.parquet
└── stage5_diagnostics.json
```

---

## Stage 6: Signal Computation — Text Channel

### 6.1 Purpose and Rationale

Stage 3 produces sentence-level or section-level sentiment and keyword intensities for both news and fund reports. Stage 6 aggregates these fine-grained outputs into panel-ready, monthly fund-level and market-level features.

### 6.2 Market-Level News Sentiment (`news_sent_market`)

Starting from a daily news-level dataset with FinBERT scores:

- Each news item \(k\) has:
  - publication date \(d_k\),
  - FinBERT sentiment score \(s_k \in [-1,1]\),
  - optional relevance weight \(w_k\) (e.g., based on source quality or article length).

For each month $t$, define:

$$
\text{NewsSentMarket}_t =
\frac{\sum_{k \in t} w_k s_k}{\sum_{k \in t} w_k}
$$

with $w_k = 1$ as the baseline.

Implementation:

```python
news_daily["yyyymm"] = news_daily["date"].dt.year * 100 + news_daily["date"].dt.month
monthly_market = (
    news_daily.groupby("yyyymm")
              .apply(lambda df: np.average(df["sentiment"], weights=None))
              .reset_index(name="news_sent_market")
)
```

This series is then lagged by one month for prediction.

### 6.3 Fund-Linked and Category-Linked News Sentiment

For articles that can be linked to specific funds or categories (via ticker mapping, name matching, or metadata):

- **Fund-level:**

$$
\text{NewsSentFund}_{i,t} =
\frac{\sum_{k \in \mathcal{N}(i,t)} s_k}{\lvert \mathcal{N}(i,t) \rvert}
$$

where \(\mathcal{N}(i,t)\) is the set of news items linked to fund \(i\) in month \(t\).

- **Category-level:** analogous aggregation over categories.

Missing values (no linked news) are carried as NaN and can later be imputed or left as missing.

### 6.4 Report-Level Sentiment Aggregation

From Stage 3’s `report_sentiment_by_section`, each filing has:

- `fund_id`, `report_date`, `filed_date`, `effective_month`,
- section-level sentiment scores \(s_{\text{MDA}}, s_{\text{Risk}}, s_{\text{Letter}}\),
- a total sentiment score computed as a length-weighted average:

$$
\text{ReportSentTotal} =
\frac{\sum_{\ell} L_{\ell} s_{\ell}}
     {\sum_{\ell} L_{\ell}}
$$

where \(L_{\ell}\) is the token or sentence count of section \(\ell\).

**Monthly mapping:** For each fund and its filing with `effective_month = m`, set:

\[
\text{report\_sent\_total}_{i,m} = \text{report\_sent\_total}(\text{filing } i).
\]

Between filings, the latest available filing sentiment is carried forward (step function) starting from `effective_month`, consistent with the point-in-time rule.

### 6.5 Sentiment Change (`report_delta_sent`)

For each fund \(i\) and each filing \(k\) with effective month \(m_k\):

$$
\text{ReportDeltaSent}_{i,m_k} =
\text{ReportSentTotal}_{i,m_k} -
\text{ReportSentTotal}_{i,m_{k-1}}
$$

where \(m_{k-1}\) is the effective month of the previous filing of the same fund. This delta is attached to month \(m_k\) and carried forward until the next filing.

### 6.6 Keyword-Based Indicators

Keyword intensity variables follow the same pattern:

- For each filing and keyword group \(g\) (e.g., liquidity, redemption, volatility), compute:

$$
\text{KWIntensity}_{g} =
\frac{\text{count of tokens in group } g}
     {\text{total tokens in relevant sections}}
$$

- Map to `effective_month` and carry forward as step functions, analogous to `report_sent_total`.

### 6.7 Stage 6 Output Artifacts

```text
artifacts/stage6/
├── text_signals_market_monthly.parquet
├── text_signals_fund_monthly.parquet
├── text_signals_category_monthly.parquet
└── stage6_diagnostics.json
```

---

## Stage 7: Crisis Label Construction

### 7.1 Purpose and Rationale

This stage constructs the target variables for crisis prediction:

- `crisis_dd_fwd`: forward-looking drawdown-based crisis indicator.
- `crisis_flow_fwd`: forward-looking outflow-based crisis indicator.

The key design constraint is to ensure that labels are defined strictly using future realizations relative to the prediction month, while feature construction uses only information available up to the prediction month, thereby avoiding target leakage.

### 7.2 Drawdown Crisis Label (`crisis_dd_fwd`)

For each fund $i$ and month $t$, define the cumulative return over a forward window of $H^{(dd)}$ months:

$$
R_{i,t}^{(H^{(dd)})} = \prod_{\tau = t+1}^{t+H^{(dd)}} (1 + r_{i,\tau}) - 1
$$

Define a drawdown threshold $\theta_{dd} < 0$ (e.g., $-0.3$ for a 30% drop). Then:

$$
\text{CrisisDDFwd}_{i,t} =
\begin{cases}
1, & \text{if } R_{i,t}^{(H^{(dd)})} \leq \theta_{dd}, \\
0, & \text{otherwise.}
\end{cases}
$$

Typical choices:

- $H^{(dd)} = 6$ or $12$ months.
- $\theta_{dd} \in [-0.2, -0.4]$ depending on desired severity.

Implementation sketch:

```python
H_DD = 6
THRESH_DD = -0.3

fund_panel = fund_panel.sort_values(["fund_id", "yyyymm"])
fund_panel["cum_ret_fwd_dd"] = (
    fund_panel.groupby("fund_id")["mret"]
              .apply(lambda x: (1 + x).rolling(H_DD).apply(np.prod, raw=True) - 1)
              .shift(-H_DD)  # shift back to align with prediction month t
)

fund_panel["crisis_dd_fwd"] = (
    (fund_panel["cum_ret_fwd_dd"] <= THRESH_DD).astype("Int64")
)
```

### 7.3 Outflow Crisis Label (`crisis_flow_fwd`)

Similarly, define cumulative net flow over a forward window \(H^{(flow)}\):

\[
F_{i,t}^{(H^{(flow)})} = \sum_{\tau = t+1}^{t+H^{(flow)}} \text{flow\_clean}_{i,\tau}.
\]

Define an outflow threshold \(\theta_{flow} < 0\) (e.g., \(-0.5\) for a cumulative 50% outflow relative to lagged TNA). Then:

\[
\text{crisis\_flow\_fwd}_{i,t} =
\begin{cases}
1, & \text{if } F_{i,t}^{(H^{(flow)})} \leq \theta_{flow}, \\
0, & \text{otherwise.}
\end{cases}
\]

Implementation sketch:

```python
H_FLOW = 6
THRESH_FLOW = -0.5

fund_panel["cum_flow_fwd"] = (
    fund_panel.sort_values(["fund_id", "yyyymm"])
              .groupby("fund_id")["flow_clean"]
              .apply(lambda x: x.rolling(H_FLOW).sum())
              .shift(-H_FLOW)
)

fund_panel["crisis_flow_fwd"] = (
    (fund_panel["cum_flow_fwd"] <= THRESH_FLOW).astype("Int64")
)
```

### 7.4 Label Validity and Sample Truncation

For months near the sample end where the full \(H\)-month forward window is not observable, labels are set to missing:

- Define `label_valid_flag = 1` if both `cum_ret_fwd_dd` and `cum_flow_fwd` are observed.
- Final modeling for Phase 3 only uses rows with `label_valid_flag == 1`.

### 7.5 Stage 7 Output Artifacts

```text
artifacts/stage7/
├── crisis_labels_drawdown.parquet
├── crisis_labels_flow.parquet
└── stage7_diagnostics.json
```

---

## Stage 9: Analysis-Phase-Specific Data Products

### 9.1 Purpose and Rationale

Different empirical analyses in the project impose different sample requirements and aggregation schemes. Stage 9 defines phase-specific derived datasets and the corresponding eligibility flags:

- `eligible_phase1`: event-study / single-channel signal existence tests.
- `eligible_phase2`: category-level VAR / network analysis of cross-channel dynamics.
- `eligible_phase3`: panel for Temporal Fusion Transformer (TFT) training and testing.

### 9.2 Phase 1: Event Study Eligibility (`eligible_phase1`)

Phase 1 evaluates whether individual signals (e.g., holdings concentration, text sentiment) show abnormal behavior around crises.

Eligibility rules:

- The fund must have:
  - At least \(L_{\text{pre}}\) months of pre-event data (e.g., 24 months).
  - At least \(L_{\text{post}}\) months of post-event data (e.g., 12 months).
- The crisis event (drawdown or outflow) is clearly identified (i.e., `crisis_dd_fwd` or `crisis_flow_fwd` triggers within a well-defined window).

Implementation:

```python
PRE = 24
POST = 12

def mark_phase1_eligible(panel):
    panel = panel.sort_values(["fund_id", "yyyymm"])
    panel["eligible_phase1"] = 0

    # Example: require continuous data around crisis windows
    # (Pseudo-code; exact implementation may track specific crisis dates)
    return panel
```

In practice, `eligible_phase1` is set to 1 for all fund-months belonging to event windows that satisfy the coverage requirements.

### 9.3 Phase 2: Category-Level VAR / Lead–Lag Analysis (`eligible_phase2`)

Phase 2 aggregates signals to the category-month level and studies the lead–lag structure between:

- Holdings-based signals,
- Text-based signals,
- Macro/fragility variables.

Eligibility rules for category \(c\) and month \(t\):

- Minimum number of active funds in category \(c\) at month \(t\) (e.g., \(N_{c,t} \geq 10\)).
- Sufficient time span of data for that category (e.g., at least 60 months of contiguous observations).

Category-level aggregation:

\[
\bar{x}_{c,t} = \frac{1}{N_{c,t}} \sum_{i \in c} x_{i,t},
\]

where \(x_{i,t}\) can be `hhi_port`, `amihud_port`, `news_sent_market`, `report_sent_total`, etc.

`eligible_phase2` is then set to 1 for all `(fund_id, yyyymm)` pairs where the corresponding `(category, yyyymm)` is included in the category-level time series used in VAR estimation.

### 9.4 Phase 3: TFT Prediction Panel (`eligible_phase3`)

Phase 3 builds the prediction panel for the Temporal Fusion Transformer.

Eligibility criteria:

- `label_valid_flag == 1` (crisis labels are fully observable).
- Sufficient history length for model input window (e.g., at least \(L_{\text{hist}} = 24\) months of past features).
- The fund belongs to the core asset class sample (e.g., actively managed U.S. equity mutual funds) and passes basic data quality filters (no extreme missingness in key predictors).

Implementation sketch:

```python
L_HIST = 24

panel_sorted = master_panel.sort_values(["fund_id", "yyyymm"])
panel_sorted["history_length"] = (
    panel_sorted.groupby("fund_id").cumcount()
)

master_panel["eligible_phase3"] = (
    (master_panel["label_valid_flag"] == 1) &
    (master_panel["history_length"] >= L_HIST)
).astype("Int64")
```

### 9.5 Phase-Specific Exported Datasets

```text
artifacts/stage9/
├── phase1_event_study_panel.parquet
├── phase2_category_month_panel.parquet
├── phase3_tft_training_panel.parquet
└── stage9_diagnostics.json
```
---

## Stage 8: Mixed-Frequency Alignment and Master Panel Construction

### 8.1 Monthly Backbone Construction

```python
backbone = [
    (fund_id, yyyymm)
    for fund_id in primary_prediction_sample
    for yyyymm in range(sample_start_month, sample_end_month + 1)
    if fund_active_in_month(fund_id, yyyymm)
]
```

### 8.2 Holdings Signals Merge Rule

For each backbone observation `(fund_id, yyyymm)`, find the most recent holdings signal with `effective_month <= yyyymm`.

```python
panel = merge_asof(
    left=backbone.sort_values(["fund_id", "yyyymm"]),
    right=holdings_signals.sort_values(["fund_id", "effective_month"]),
    by="fund_id",
    left_on="yyyymm",
    right_on="effective_month",
    direction="backward"
)
```

### 8.3 Uniqueness Validation

```python
assert master_panel.groupby(["fund_id", "yyyymm"]).size().max() == 1, \
    "Duplicate fund-month observations detected — merge logic error"
```

---

## Stage 10: Standardization, Missing Values, and Outlier Treatment

### 10.1 Winsorization

```python
for each_month in sample_months:
    for feature in continuous_features:
        lower_bound = percentile(feature_values_in_month(each_month), 1)
        upper_bound = percentile(feature_values_in_month(each_month), 99)
        winsorized_feature[each_month] = clip(
            raw_feature[each_month],
            lower_bound,
            upper_bound
        )
```

### 10.2 Walk-Forward Standardization

```python
for step in walk_forward_steps:
    training_data = panel[panel.yyyymm <= training_cutoff[step]]

    for feature in continuous_features:
        mean_f = training_data[feature].mean(skipna=True)
        std_f = training_data[feature].std(skipna=True)

        if std_f < 1e-8:
            std_f = 1.0

        panel_scaled.loc[:, feature] = (panel[feature] - mean_f) / std_f

    save_scaler_params(step=step, means=mean_f, stds=std_f)
```
## Stage 11: Diagnostics, Validation, and Quality Assurance

### 11.1 Purpose and Rationale

Before the final panel is handed over for modeling, it must pass a rigorous suite of diagnostic checks. Silent failures in financial data preprocessing (e.g., merging errors, regime-break artifacts, unhandled survivorship bias) often produce models that look successful but are conceptually flawed. This stage enforces quantitative and qualitative validation.

### 11.2 Internal Consistency Checks

These checks verify the mechanical integrity of the data pipeline:

*   **Row Counts and Attrition:** Verify that the number of funds per month remains stable and does not exhibit unexplained dropouts, particularly around the 2019 transition from Form N-Q to N-PORT.
*   **Uniqueness:** Confirm that `(fund_id, yyyymm)` uniquely identifies every row in the master panel.
*   **Missingness Distribution:** Compute the percentage of missing values for every signal across time. A sudden spike in missing values usually indicates a broken CUSIP linkage or a parsing failure in the EDGAR extraction pipeline.
*   **Filing Lag Distribution:** Calculate `filed_date - report_date`. Verify that the median gap aligns with SEC rules (e.g., roughly 60 days for N-PORT and N-CSR). Flag any filings with negative lags or lags exceeding 90 days.

```python
# Example consistency check
def check_filing_lags(panel):
    lag_days = (panel['holdings_filed_date'] - panel['holdings_report_date']).dt.days
    assert lag_days.min() >= 0, "Error: Filed date cannot precede report date"
    print(f"Median filing lag: {lag_days.median()} days")
```

### 11.3 Signal Plausibility Checks

These checks confirm that the computed signals reflect known historical realities:

*   **Concentration Dynamics:** Verify that aggregate `hhi_port` and `top10_share` show upward trends during known periods of market crowding (e.g., the \"FAANG\" concentration build-up prior to the COVID crash).
*   **Sentiment Dynamics:** Confirm that `news_sent_market` drops sharply precisely at known crisis onset dates (e.g., September 2008, February 2020).
*   **Structural Fragility Trend:** Check that the `fragility_idx` exhibits a structural upward trend in the post-2015 era, corresponding to the well-documented rise of passive investing.

### 11.4 External Benchmark Validation

*   **Aggregate Flow vs. ICI:** Sum the `flow_clean` across all funds in the panel for each month and calculate the correlation with the official Investment Company Institute (ICI) mutual fund flow statistics. A high correlation (>0.85) confirms that the CRSP-based flow extraction and merger-cleaning logic is sound.
*   **Illiquidity vs. Market Events:** Verify that the portfolio-weighted `amihud_port` measure spikes concurrently with known market-wide liquidity freezes (e.g., March 2020 Treasury market dysfunction).

### 11.5 Survivorship Bias Checks

*   **Death Rate Spikes:** Plot the count of `terminal_flag = 1` by quarter. Ensure there are visible spikes during and immediately following the Dot-com and GFC episodes. If deaths are uniform or missing, the CRSP delisting linkage failed.
*   **Terminal Outflows:** Verify that funds nearing their termination date exhibit severe negative values in `flow_raw` and are properly captured in the target label before exiting the sample.

---

## Stage 12: Version Control and Reproducibility

### 12.1 Purpose and Rationale

Financial databases (CRSP, Morningstar) and external APIs (GDELT, RavenPack) are routinely updated, backfilled, and revised. A predictive signal generated today might look different if the same database is queried a year later. To ensure absolute reproducibility, the pipeline must log exact data vintages and software environments.

### 12.2 Data Vintage Tracking

A `reproducibility_log.json` artifact must be generated alongside the final panel, recording:

*   **WRDS Extraction Date:** The exact date CRSP Mutual Fund and CRSP Stock files were queried, along with the specific table versions (e.g., `crsp_q_mutualfunds`).
*   **SEC EDGAR Scope:** The date the bulk download was executed and the exact range of quarters included.
*   **News Corpus Vintage:** The cutoff date and version of the RavenPack or GDELT dataset used.
*   **Macro Data Vintage:** The exact retrieval date for FRED and ICI series, acknowledging that macro series are often subject to retrospective revisions.

### 12.3 Software and Pipeline Versioning

*   **FinBERT Model Version:** Document whether the off-the-shelf `ProsusAI/finbert` was used or a custom fine-tuned checkpoint. Record the Hugging Face model hash.
*   **T2MD Pipeline Commit:** Record the Git commit hash of the Text-to-Machine-Data (T2MD) repository used to parse N-CSR and N-Q HTML filings.
*   **Preprocessing Script Commit:** Record the Git commit hash of the primary processing repository.

### 12.4 Execution Environment

Provide a standard `requirements.txt`, `environment.yml`, or `poetry.lock` defining the exact versions of Pandas, PyTorch, Transformers, and scikit-learn used during extraction and merging.

```json
{
  "pipeline_execution": {
    "timestamp": "2025-XX-XXT14:32:00Z",
    "git_commit": "a1b2c3d4e5f6...",
    "t2md_commit": "9f8e7d6c5b4a...",
    "finbert_model": "ProsusAI/finbert",
    "data_vintages": {
      "crsp_mutual_fund": "2025-01-15",
      "sec_edgar_through": "2024-12-31",
      "fred_api_pull": "2025-02-01"
    }
  }
}
```

The pipeline execution is considered complete only when this reproducibility log is successfully written to the `artifacts/final/` directory alongside the master panel.
---
## Appendix A: Complete Final Panel Schema

The final output of the preprocessing pipeline is a unified fund-month master panel. Every observation represents one portfolio in one calendar month.

| Column Name | Data Type | Description | Origin / Stage |
|---|---|---|---|
| `fund_id` | String | Portfolio-level identifier (e.g., `crsp_portno`) | Stage 1 |
| `share_class_count` | Integer | Number of share classes aggregated | Stage 1 |
| `fund_family_id` | String | Management company / family identifier | Stage 1 |
| `yyyymm` | Integer | Calendar month (YYYYMM format) | Stage 1 |
| `date` | Date | Month-end date | Stage 1 |
| `mret` | Float | Monthly net return (TNA-weighted across classes) | Stage 1 |
| `mtna` | Float | Portfolio-level Total Net Assets (TNA) | Stage 1 |
| `flow_raw` | Float | Implied raw fund flow | Stage 1 |
| `flow_clean` | Float | Cleaned flow (mergers/reorganizations handled) | Stage 1 |
| `flow_hist_n` | Integer | Number of prior months used for flow baseline | Stage 1 |
| `news_sent_market` | Float | Market-wide news sentiment (monthly aggregated) | Stage 3 |
| `news_sent_fund` | Float | Fund-linked news sentiment | Stage 3 |
| `report_sent_total` | Float | Total filing-level fund-report sentiment | Stage 3 |
| `report_delta_sent` | Float | Change in sentiment versus prior filing | Stage 3 |
| `hhi_port` | Float | Holdings concentration (Herfindahl-Hirschman Index) | Stage 2 |
| `top10_share` | Float | Top 10 holdings share | Stage 2 |
| `amihud_port` | Float | Portfolio-weighted Amihud illiquidity | Stage 2/5 |
| `overlap_cat` | Float | Cosine similarity with category-average portfolio | Stage 2/5 |
| `mes_port` | Float | Portfolio Marginal Expected Shortfall | Stage 2/5 |
| `turnover_chg` | Float | Turnover anomaly (QoQ change) | Stage 1/5 |
| `hbr_ex_post` | Float | Ex post holdings-based return (Diagnostic only) | Stage 5 |
| `return_gap_ex_post`| Float | Reported return minus ex post HBR (Diagnostic) | Stage 5 |
| `holdings_source` | String | Form source (`13F`, `N-Q`, `N-CSR-SOI`, `N-PORT`) | Stage 2 |
| `holdings_coverage` | String | Coverage level (`fund` or `family`) | Stage 2 |
| `holdings_eff_mo` | Integer | First month the holdings signal is usable | Stage 2 |
| `text_eff_mo` | Integer | First month the report-text signal is usable | Stage 3 |
| `fragility_idx` | Float | Lagged structural fragility index | Stage 4 |
| `vix` | Float | Observed macro input: CBOE Volatility Index | Stage 4 |
| `credit_spread` | Float | Observed macro input: BAA minus Treasury | Stage 4 |
| `yield_slope` | Float | Observed macro input: 10Y minus 2Y Treasury | Stage 4 |
| `category` | String | Morningstar fund category | Stage 1 |
| `crisis_dd_fwd` | Integer | Target: Forward drawdown crisis label (0/1) | Stage 7 |
| `crisis_flow_fwd` | Integer | Target: Forward outflow crisis label (0/1) | Stage 7 |
| `eligible_phase1` | Boolean | Event-study eligibility flag | Stage 9 |
| `eligible_phase2` | Boolean | VAR aggregation eligibility flag | Stage 9 |
| `eligible_phase3` | Boolean | TFT training/testing eligibility flag | Stage 9 |

---
## Appendix B: Processing Order Summary

1. Extract CRSP mutual fund monthly data.
2. Aggregate share classes to portfolio level.
3. Compute flows and classify flow events.
4. Apply sample filters and survivorship rules.
5. Extract holdings across N-PORT, N-Q/N-CSR, and 13F regimes.
6. Standardize CUSIPs and map to PERMNO where possible.
7. Compute holdings-based signals.
8. Extract and score textual signals.
9. Construct macro and structural fragility variables.
10. Build crisis labels.
11. Merge all channels onto the monthly backbone.
12. Create phase-specific datasets.
13. Apply winsorization, missing-value tagging, and standardization.
14. Run diagnostics and export artifacts.
---
## Appendix C: Key Design Choices Checklist

This checklist documents the critical methodological decisions implemented in this pipeline to ensure robust and unbiased crisis prediction:

- [x] **Point-in-Time Discipline:** `filed_date` is strictly used over `report_date` to determine when information entered the public domain.
- [x] **Target Leakage Prevention:** Future-dependent variables (like Holdings-Based Returns and Return Gaps) are classified as `ex_post` and excluded from Phase 3 (TFT) predictive inputs.
- [x] **Survivorship Bias Mitigation:** Dead, merged, and liquidated funds are retained in the panel through their terminal month.
- [x] **Mixed-Frequency Alignment:** Holdings and textual signals are carried forward as step functions starting *only* from their `effective_month`.
- [x] **Regime Awareness:** Signals explicitly track their regulatory source (`N-PORT`, `N-Q`, `13F`) to prevent machine learning models from learning spurious regulatory regime shifts.
- [x] **Outlier Treatment:** Winsorization and standard scaling are performed dynamically within each walk-forward training window, never using full-sample distributions.

---

## Appendix D: Mapping to Proposal Research Questions and Hypotheses

The data artifacts generated by this pipeline directly support the three core research questions outlined in the proposal:

### RQ1 / Phase 1: Signal Existence (Single-Channel Validation)
*   **Mechanism:** Tests whether individual textual and holdings signals deviate significantly from baselines prior to the Dot-com, GFC, and COVID-19 crises.
*   **Pipeline Inputs:** 
    *   Text: `news_sent_market`, `report_delta_sent`
    *   Holdings: `hhi_port`, `amihud_port`, `overlap_cat`, `mes_port`
    *   Eligibility: `eligible_phase1 == True`

### RQ2 / Phase 2: Cross-Channel Dynamics (Lead-Lag Analysis)
*   **Mechanism (H1 & H2):** Tests whether private actions (holdings adjustments) precede public disclosures (report tone), and whether both are Granger-caused by external news.
*   **Pipeline Inputs:** 
    *   Monthly category-level aggregations of `news_sent_market`, `report_sent_total`, and holdings signals.
    *   Structural Covariate: `fragility_idx`.
    *   Eligibility: `eligible_phase2 == True`

### RQ3 / Phase 3: Fusion Superiority (Temporal Fusion Transformer)
*   **Mechanism (H3 & H4):** Tests whether fusing the text and holdings channels achieves superior out-of-sample crisis prediction compared to single-channel baselines.
*   **Pipeline Inputs:** 
    *   All observed predictive features (standardized walk-forward).
    *   Targets: `crisis_dd_fwd` and `crisis_flow_fwd`.
    *   Eligibility: `eligible_phase3 == True` (excluding the final 6 months of the sample where targets cannot be fully observed).
    ## Appendix E: Computational Requirements & Data Volume Estimates

This project processes over 20 years of high-frequency market data, unstructured regulatory texts, and security-level portfolio holdings. Due to the massive scale of the SEC EDGAR extractions and deep learning (NLP) inference, executing this pipeline requires workstation-grade or server-grade computational resources.

### 1. Estimated Data Volume & Processing Scale

The following table outlines the expected data scale at each major pipeline stage. 

| Data Source / Pipeline Stage | Granularity | Estimated Raw Size | Estimated Record Count | Processing Bottlenecks & Notes |
| :--- | :--- | :--- | :--- | :--- |
| **CRSP Mutual Fund** | Fund-Month | ~2 GB | 10M – 15M rows | Light. TNA-weighted aggregation and survivorship bias handling. |
| **CRSP Stock Daily** | Stock-Day | ~50 GB | ~100M rows | Moderate. Used for daily liquidity (Amihud) and ex-post HBR calculations. |
| **SEC Holdings (N-PORT, N-Q)** | Fund-Month-Sec | 150 GB – 300 GB | 100M – 500M rows | **Heavy Memory I/O.** Parsing historical HTML (N-Q); joining CUSIP holdings with daily stock data. |
| **SEC Text Corpus (N-CSR)** | Fund-SemiAnnual| 50 GB – 100 GB | > 100,000 filings | Moderate I/O. Extracting specific sections (MD&A, Risk) via regex/T2MD. |
| **FinBERT Inference** | Sentence-level | - | 20M – 50M sentences | **Heavy GPU Compute.** ~100-200 hours of pure inference time on a single RTX 3090. |
| **Final Master Panel** | Portfolio-Month | ~150 MB | ~1,000,000 rows | Light. The final unified panel (50+ features) ready for TFT modeling. |

### 2. Hardware Requirements

To avoid Out-Of-Memory (OOM) errors during the `Holdings-to-Stock` joins and to process the FinBERT NLP pipeline within a reasonable timeframe (e.g., 2-3 weeks of continuous run time), the following hardware specifications are required:

| Component | Minimum Specification | Recommended Specification | Reason / Bottleneck |
| :--- | :--- | :--- | :--- |
| **Memory (RAM)** | 64 GB | **128 GB or higher** | Crucial for in-memory joins of holding-level data (Pandas/Polars). |
| **Storage (Disk)** | 1 TB NVMe SSD | **2 TB+ NVMe SSD** | Fast I/O is mandatory; HDDs will severely bottleneck EDGAR XML/HTML parsing. |
| **Processor (CPU)** | 8-core / 16-thread | **16-core+ / 32-thread** | Required for parallel downloading and multi-processing of SEC files. |
| **Graphics (GPU)** | 1x NVIDIA T4 / RTX 3090 (16GB+) | **1x NVIDIA A100** or **2x RTX 4090** | Dramatically accelerates FinBERT sentiment inference for 50M+ sentences. |

> **HPC / Cloud Recommendation:** If processing on a local machine is not feasible, allocating an AWS EC2 instance (e.g., `p3.2xlarge` for NLP, `r5.4xlarge` for memory-heavy data joins) or utilizing the university's High-Performance Computing (HPC) cluster is strongly advised.

### 3. Software & Environment Stack

The pipeline relies on a modern data science and deep learning stack. Using `Parquet` instead of `CSV` for all intermediate files is strictly enforced to manage the large holdings files efficiently.

| Category | Recommended Technologies, Libraries & Versions |
| :--- | :--- |
| **Operating System** | Ubuntu 20.04 / 22.04 LTS (Recommended for HPC compatibility) |
| **Core Language** | Python 3.10+ |
| **Data Extraction** | `wrds` (WRDS Python API), `sec-edgar-downloader`, `requests` |
| **Data Manipulation** | `pandas >= 2.0`, `numpy`, `polars` (highly recommended for large holdings merges) |
| **Text Parsing** | `BeautifulSoup4`, `lxml` (for N-Q/N-CSR HTML parsing), `regex` |
| **Deep Learning / NLP** | `pytorch >= 2.0` (with CUDA support), `transformers` (Hugging Face) |
| **Pre-trained Model** | `ProsusAI/finbert` (Financial Sentiment Analysis) |
| **Forecasting Model** | `pytorch-forecasting` (for Temporal Fusion Transformer implementation) |
| **Data Storage Format**| `pyarrow`, `fastparquet` |
| **Experiment Tracking**| `wandb` (Weights & Biases) or `mlflow` (for tracking TFT training metrics) |

