"""
run_horse_race.py
One-click full ML Horse-Race Pipeline for Weigert et al. (RFS 2023)
Author: Tao Wu
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Core modules
from src.models import AssetPricingModels
from src.oos_validation import OOSValidator
from src.portfolio import PortfolioConstructor
from src.shap_explain import SHAPExplainer


def main():
    parser = argparse.ArgumentParser(description="Run full ML Horse-Race Pipeline")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of OOS splits")
    parser.add_argument("--test_size", type=int, default=252, help="Test window size")
    parser.add_argument("--shap_model", type=str, default="XGBoost",
                        help="Model to run SHAP on (default: XGBoost)")
    parser.add_argument("--no_shap", action="store_true", help="Skip SHAP analysis")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Starting ML Horse-Race Asset Pricing Pipeline")
    print("Replicating Weigert et al. (RFS 2023)")
    print("=" * 80)

    # 1. Load data - Support both demo stock panel and FF5
    demo_path = Path("/home/user/Downloads/ml-asset-pricing-horse-rac-main/data/demo_stock_panel.csv")
    ff5_path = Path("/home/user/Downloads/ml-asset-pricing-horse-rac-main/data/ff5_factors.csv")

    if demo_path.exists():
        print("✅ Using generated demo stock panel data")
        df = pd.read_csv(demo_path, parse_dates=['date'])
        # Rename 'return' to 'target' to match pipeline expectation
        if 'return' in df.columns:
            df = df.rename(columns={'return': 'target'})
    elif ff5_path.exists():
        print("✅ Using Fama-French 5 Factors demo data")
        ff5 = pd.read_csv(ff5_path, index_col=0, parse_dates=True)
        df = ff5.copy()
        df['target'] = df['Mkt-RF'].shift(-1)
        df = df.dropna()
    else:
        print("❌ No demo data found.")
        print("   Please run one of the following first:")
        print("   python generate_demo_data.py")
        print("   python data_download.py")
        return

    # Prepare features and target
    y = df['target']
    # Keep only numeric columns (drop date, ticker, etc.)
    X = df.drop(columns=['target']).select_dtypes(include=[np.number])

    print(f"Features: {X.shape[1]} | Samples: {len(X)}")

    # 2. Initialize components
    models = AssetPricingModels(random_state=42)
    validator = OOSValidator(n_splits=args.n_splits, test_size=args.test_size)
    portfolio = PortfolioConstructor(n_quantiles=10, periods_per_year=12)
    explainer = SHAPExplainer(output_dir="output/shap")

    # 3. Run horse-race
    print("\n🔥 Running full horse-race...")
    model_names = ['LASSO', 'ElasticNet', 'PCR', 'PLS',
                   'RandomForest', 'GBM', 'XGBoost', 'LightGBM']

    results = {}
    for name in model_names:
        print(f"\nTraining {name}...")
        model = models.get_model(name.lower())

        oos_preds = []
        for X_train, X_test, y_train, y_test in validator.split(X, y):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            oos_preds.extend(pred)

        port_df, ls_returns = portfolio.long_short_portfolios(
            pd.Series(oos_preds, index=y.index[-len(oos_preds):]),
            y.iloc[-len(oos_preds):]
        )

        # Evaluate performance using long-short return time series
        perf = portfolio.evaluate_performance(ls_returns)
        results[name] = perf

    # Save and display summary
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results).T
    summary.to_csv(output_dir / "horse_race_summary.csv")
    print("\n🎯 Final Horse-Race Performance Summary (Long-Short)")
    print(summary.round(4))

    # 4. SHAP (optional)
    if not args.no_shap:
        print(f"\n📊 Running SHAP for {args.shap_model}...")
        model_shap = models.get_model(args.shap_model.lower())
        model_shap.fit(X.iloc[:-252], y.iloc[:-252])

        result = explainer.explain_model(
            model=model_shap,
            X_train=X.iloc[:-252],
            X_test=X.iloc[-252:],
            model_name=args.shap_model
        )
        print(f"✅ SHAP completed! Files saved to output/shap/")

    print("\n" + "="*80)
    print("🎉 All done! Results saved to output/")
    print("="*80)


if __name__ == "__main__":
    main()
