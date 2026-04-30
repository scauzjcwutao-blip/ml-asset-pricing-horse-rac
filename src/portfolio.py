"""
Portfolio Construction Module (Final Fixed Version)
Implements decile portfolio sorting and performance evaluation.
Fixed: extreme max_drawdown caused by incorrect reindex/fill_value=0
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class PortfolioConstructor:
    """
    Construct decile portfolios and evaluate performance.
    Fixed version: safe long-short return calculation to prevent extreme drawdowns.
    """

    def __init__(self, n_quantiles: int = 10, periods_per_year: int = 12):
        self.n_quantiles = n_quantiles
        self.periods_per_year = periods_per_year

    def long_short_portfolios(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create decile portfolios and compute long-short returns.
        Safe version: no dangerous fill_value=0 reindex.
        """
        # Align and clean data
        df = pd.DataFrame({
            'pred': predictions,
            'actual': actual_returns
        }).dropna()

        if len(df) == 0:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Assign deciles (cross-sectional if possible, otherwise time-series)
        df['decile'] = pd.qcut(df['pred'], q=self.n_quantiles,
                               labels=False, duplicates='drop')

        # 1. Summary: average return per decile
        decile_avg = df.groupby('decile')['actual'].mean()
        long_short_mean = decile_avg.iloc[-1] - decile_avg.iloc[0]

        summary = pd.DataFrame({
            'Decile': [f'D{i+1}' for i in range(len(decile_avg))],
            'Mean_Return': decile_avg.values
        })
        ls_row = pd.DataFrame({'Decile': ['Long-Short'], 'Mean_Return': [long_short_mean]})
        summary = pd.concat([summary, ls_row], ignore_index=True)

        # 2. Time-series long-short returns (fixed safe version)
        # Group by date first, then compute top - bottom
        daily = df.groupby([df.index, 'decile'])['actual'].mean().unstack()

        if len(daily.columns) < 2:
            ls_returns = pd.Series(0.0, index=df.index, name='long_short')
        else:
            top = daily.columns[-1]
            bottom = daily.columns[0]
            ls_returns = daily[top] - daily[bottom]
            ls_returns = ls_returns.dropna()   # 只保留 top 和 bottom 同时存在的日期

        return summary, ls_returns

    def evaluate_performance(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate key performance metrics.
        """
        returns = portfolio_returns.dropna()
        if len(returns) < 2:
            return {
                'annualized_mean_return': np.nan,
                'annualized_volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'n_periods': 0
            }

        # Excess return for Sharpe
        if risk_free_rate is not None:
            rf = risk_free_rate.reindex(returns.index).fillna(0)
            excess = returns - rf
        else:
            excess = returns

        mean_ret = returns.mean() * self.periods_per_year
        vol = returns.std() * np.sqrt(self.periods_per_year)
        sharpe = (excess.mean() * self.periods_per_year) / vol if vol > 0 else np.nan

        # Safe max drawdown
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()

        return {
            'annualized_mean_return': mean_ret,
            'annualized_volatility': vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'n_periods': len(returns)
        }

    def decile_analysis(self, predictions: pd.Series, actual_returns: pd.Series) -> pd.DataFrame:
        """Optional: detailed decile statistics"""
        df = pd.DataFrame({'pred': predictions, 'actual': actual_returns}).dropna()
        df['decile'] = pd.qcut(df['pred'], q=self.n_quantiles, labels=False, duplicates='drop')
        stats = df.groupby('decile')['actual'].agg(['mean', 'std', 'count']).reset_index()
        stats.columns = ['decile', 'mean_return', 'std_return', 'n_obs']
        return stats
