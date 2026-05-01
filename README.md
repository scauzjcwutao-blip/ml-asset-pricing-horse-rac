# ML Horse-Race Asset Pricing Pipeline

**Replicating & Extending Weigert et al. (RFS 2023 & JF 2024)**

A clean, reproducible implementation of machine learning frameworks for asset pricing.

### Key Features
- **Models**: LASSO, Elastic Net, PCR/PLS, Random Forest, Gradient Boosted Trees, XGBoost, LightGBM, Feed-forward Neural Network, Ensemble
- **Validation**: Strict rolling-window Out-of-Sample (OOS) testing
- **Economic Significance**: Decile long-short portfolios, Sharpe ratio, t-stat, annualized returns
- **Explainability**: Full SHAP analysis for every model (coming soon)
- **Ready for extension**: Designed to integrate seamlessly with textual features from `text2md` + `FinBERT-Valuation-Signals`

### Quick Start

1. Clone the repository
   ```bash
   git clone https://github.com/scauzjcwutao-blip/ml-asset-pricing-horse-rac.git
   cd ml-asset-pricing-horse-rac
   ```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Download the required data
```bash
python data_download.py          # Automatically downloads Fama-French 5 Factors (daily)
```
```bash
# 2023 RFS Paper (Option Return Predictability)
python replicate_weigert_rfs2023.py
# 2024 JF Paper (Unobserved Performance of Hedge Funds)
python replicate_weigert_up_2024.py
```
```bash
jupyter lab
```
### Project Structure

The repository is organized as follows:
```bash
ml-asset-pricing-horse-rac/
├── notebooks/                    # Jupyter notebooks
├── src/                          # Core source code
├── data/                         # Automatically created
│   ├── ff5_factors.csv
│   └── results/                  # Generated results
├── .gitignore
├── README.md
├── generate_demo_data.py
├── data_download.py
├── replicate_weigert_rfs2023.py
├── replicate_weigert_up_2024.py
├── requirements.txt
└── (more modules coming soon)
### Current Status

Both replication scripts are ready to run with demo data.  
Full pipeline with real WRDS/OptionMetrics data is in progress.

### Replication Scripts

- **`replicate_weigert_rfs2023.py`** — Replicates **Turan G Bali, Heiner Beckmeyer, Mathis Mörke, Florian Weigert, Option Return Predictability with Machine Learning and Big Data, The Review of Financial Studies, Volume 36, Issue 9, September 2023, Pages 3548–3602, https://doi.org/10.1093/rfs/hhad017
- **`replicate_weigert_up_2024.py`** — Replicates AGARWAL, V., RUENZI, S. and WEIGERT, F. (2024), Unobserved Performance of Hedge Funds. J Finance, 79: 3203-3259. https://doi.org/10.1111/jofi.13368
```
### Demo Video

<video width="800" controls autoplay muted loop>
  <source src="https://raw.githubusercontent.com/scauzjcwutao-blip/ml-asset-pricing-horse-rac/main/DEMO.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<p><em>If the video does not play, please try refreshing the page or click <a href="https://raw.githubusercontent.com/scauzjcwutao-blip/ml-asset-pricing-horse-rac/main/DEMO.mp4" target="_blank">here to open directly</a>.</em></p>
