# Data Processing Specification: Dual-Channel Fund Crisis Early Warning System

**Document Version:** 2.0  
**Last Updated:** 2025-XX-XX  
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

\[
\text{Flow}^{\$}_{i,t} = \text{TNA}_{i,t} - \text{TNA}_{i,t-1} \times (1 + r_{i,t})
\]

**Percentage flow:**

\[
\text{flow}_{i,t} = \frac{\text{Flow}^{\$}_{i,t}}{\text{TNA}_{i,t-1}}
\]

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

\[
\text{Parse Quality Ratio} =
\frac{\sum_j \text{parsed\_value}_j}{\text{reported\_NAV} \times \text{shares\_outstanding}}
\]

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
