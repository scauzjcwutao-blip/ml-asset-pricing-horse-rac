"""
Option Return Predictability with Machine Learning and Big Data
Weigert et al. (Review of Financial Studies, 2023)
Python Replication Script
Author: Tao Wu
Date: April 2026

This script implements the linear vs. nonlinear ML horse-race framework
for predicting option returns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# ========================== CONFIG ==========================
np.random.seed(42)
DATA_DIR = Path("data")
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("Replicating Weigert et al. (RFS 2023) — Option Return Predictability with ML")
print("=" * 90)

# ========================== 1. Load & Prepare Data ==========================
def load_demo_data():
    """Demo data only. In real replication, load OptionMetrics + CRSP + FF5 here."""
    print("Loading demonstration data (replace with real option panel data)...")
    
    dates = pd.date_range(start='1996-01-01', end='2020-12-31', freq='ME')  # 新版 pandas 推荐 'ME'
    n_obs = len(dates) * 500
    
    df = pd.DataFrame({
        'date': np.repeat(dates, 500),
        'option_id': np.tile(range(500), len(dates)),
        'iv': np.random.normal(0.35, 0.15, n_obs),
        'delta': np.random.normal(0.5, 0.3, n_obs),
        'gamma': np.random.normal(0.05, 0.03, n_obs),
        'vega': np.random.normal(0.1, 0.05, n_obs),
        'moneyness': np.random.normal(1.0, 0.2, n_obs),
        'maturity': np.random.uniform(30, 180, n_obs),
        'mktrf': np.random.normal(0.006, 0.04, n_obs),
        'smb': np.random.normal(0.002, 0.03, n_obs),
        'hml': np.random.normal(0.003, 0.03, n_obs),
        'rmw': np.random.normal(0.002, 0.02, n_obs),
        'cma': np.random.normal(0.001, 0.02, n_obs),
        'ret': np.random.normal(0.01, 0.12, n_obs)
    })
    
    # Load real FF5 if available
    ff5_path = DATA_DIR / "ff5_factors.csv"
    if ff5_path.exists():
        ff5 = pd.read_csv(ff5_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded real FF5 factors: {ff5.shape}")
    else:
        print("⚠️  ff5_factors.csv not found (run data_download.py first)")
    
    print(f"✅ Simulated option panel shape: {df.shape}")
    return df


# ========================== 2. Model Horse-Race (Fixed) ==========================
def run_horse_race(df: pd.DataFrame):
    """Main ML horse-race: each model gets its own predictions."""
    print("\n🚀 Starting ML Horse-Race (Rolling OOS)...")
    
    feature_cols = ['iv', 'delta', 'gamma', 'vega', 'moneyness', 'maturity',
                    'mktrf', 'smb', 'hml', 'rmw', 'cma']
    X = df[feature_cols].copy()
    y = df['ret'].copy()
    
    # Feature scaling (important for linear models & NN)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    
    tscv = TimeSeriesSplit(n_splits=8)   # 增加分割次数，更合理
    
    predictions = {}   # 每个模型单独保存预测
    
    models = {
        'LASSO': Lasso(alpha=0.001, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
        'PCR': None,   # 特殊处理
        'PLS': PLSRegression(n_components=5),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
        'NeuralNet': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model_preds = pd.Series(index=df.index, dtype=float)
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            
            if name == 'PCR':
                pca = PCA(n_components=5)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                reg = Lasso(alpha=0.001, max_iter=10000)
                reg.fit(X_train_pca, y_train)
                pred = reg.predict(X_test_pca)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            
            model_preds.iloc[test_idx] = pred
        
        predictions[name] = model_preds
        df[f'pred_{name}'] = model_preds
    
    print(f"✅ Horse-race completed. Predictions generated for {len(models)} models.")
    return df


# ========================== 3. Portfolio Construction & Evaluation ==========================
def form_decile_portfolios(df: pd.DataFrame):
    """Form 10-1 long-short portfolios for each model."""
    print("\n📊 Forming decile long-short portfolios for each model...")
    
    results = {}
    for col in [c for c in df.columns if c.startswith('pred_')]:
        model_name = col.replace('pred_', '')
        temp = df.dropna(subset=[col]).copy()
        temp['decile'] = temp.groupby('date')[col].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
        )
        
        port_returns = temp.groupby(['date', 'decile'])['ret'].mean().unstack()
        long_short = port_returns[10] - port_returns[1]
        
        mean_ret = long_short.mean()
        results[model_name] = {
            'Mean Monthly': mean_ret,
            'Annualized': mean_ret * 12,
            'Sharpe': mean_ret / long_short.std() * np.sqrt(12),
            't-stat': mean_ret / (long_short.std() / np.sqrt(len(long_short))) if len(long_short) > 0 else np.nan
        }
    
    # Print summary
    summary = pd.DataFrame(results).T
    print("\n🎯 Model Horse-Race Performance (10-1 Long-Short):")
    print(summary.round(4))
    
    # Save
    summary.to_csv(RESULTS_DIR / "model_performance_summary.csv")
    df.to_csv(RESULTS_DIR / "predictions_with_deciles.csv", index=False)
    
    return summary


# ========================== Main Execution ==========================
if __name__ == "__main__":
    df = load_demo_data()
    df = run_horse_race(df)
    performance = form_decile_portfolios(df)
    
    print("\n" + "="*90)
    print("🎉 Replication of Weigert et al. (RFS 2023) completed successfully!")
    print("💡 Replace demo data with real OptionMetrics + CRSP panel for full replication.")
    print(f"📁 All results saved to: {RESULTS_DIR}")
    print("="*90)
