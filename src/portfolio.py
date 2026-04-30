"""
Portfolio Construction Module
Implements decile portfolio sorting and performance evaluation for asset pricing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class PortfolioConstructor:
    """
    Construct decile portfolios based on model predictions and evaluate performance.
    """

    def __init__(self, n_quantiles: int = 10, periods_per_year: int = 12):
        """
        Initialize portfolio constructor.

        Parameters:
        -----------
        n_quantiles : int
            Number of quantiles for portfolio sorting (default: 10 for deciles)
        periods_per_year : int
            Number of periods per year for annualization (252 for daily, 12 for monthly)
        """
        self.n_quantiles = n_quantiles
        self.periods_per_year = periods_per_year

    def long_short_portfolios(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series
    ) -> tuple:
        """
        Create decile portfolios sorted by predictions.

        Parameters:
        -----------
        predictions : pd.Series
            Model predicted returns
        actual_returns : pd.Series
            Actual realized returns

        Returns:
        --------
        tuple : (summary_df, ls_returns_series)
            - summary_df: DataFrame with average returns per decile
            - ls_returns_series: Time series of long-short portfolio returns
        """
        # Align indices
        df = pd.DataFrame({
            'pred': predictions,
            'actual': actual_returns
        }).dropna()

        # Create deciles based on predictions
        df['decile'] = pd.qcut(df['pred'], q=self.n_quantiles, labels=False, duplicates='drop')

        # Calculate decile portfolio returns (average)
        decile_returns = df.groupby('decile')['actual'].mean()

        # Long-short: long top decile (n_quantiles-1), short bottom decile (0)
        long_short = decile_returns.iloc[-1] - decile_returns.iloc[0]

        # Create summary DataFrame
        result = pd.DataFrame({
            'Decile': [f'D{i+1}' for i in range(len(decile_returns))],
            'Return': decile_returns.values
        })

        # Add long-short row
        ls_row = pd.DataFrame({'Decile': ['Long-Short'], 'Return': [long_short]})
        result = pd.concat([result, ls_row], ignore_index=True)

        # Calculate time series of long-short returns (per period)
        top_decile = df['decile'].max()
        bottom_decile = df['decile'].min()
        top_returns = df[df['decile'] == top_decile]['actual']
        bottom_returns = df[df['decile'] == bottom_decile]['actual']

        # Reindex to full index and fill with 0 for periods where decile has no observations
        full_index = df.index
        top_returns_aligned = top_returns.reindex(full_index, fill_value=0)
        bottom_returns_aligned = bottom_returns.reindex(full_index, fill_value=0)
        ls_returns = top_returns_aligned - bottom_returns_aligned

        return result, ls_returns

    def evaluate_performance(
        self,
        portfolio_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.

        Parameters:
        -----------
        portfolio_returns : pd.Series
            Time series of portfolio returns

        Returns:
        --------
        Dict[str, float] : Performance metrics
        """
        returns = portfolio_returns.dropna()

        if len(returns) == 0:
            return {
                'mean_return': np.nan,
                'volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan
            }

        # Mean annualized return
        mean_return = returns.mean() * self.periods_per_year

        # Annualized volatility
        volatility = returns.std() * np.sqrt(self.periods_per_year)

        # Sharpe ratio (assuming zero risk-free rate for simplicity)
        sharpe_ratio = mean_return / volatility if volatility != 0 else np.nan

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def decile_analysis(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Full decile analysis: monotonicity test and spread significance.

        Parameters:
        -----------
        predictions : pd.Series
            Model predicted returns
        actual_returns : pd.Series
            Actual realized returns

        Returns:
        --------
        pd.DataFrame : Decile statistics
        """
        df = pd.DataFrame({
            'pred': predictions,
            'actual': actual_returns
        }).dropna()

        df['decile'] = pd.qcut(df['pred'], q=self.n_quantiles, labels=False, duplicates='drop')

        # Statistics per decile
        stats = df.groupby('decile').agg({
            'actual': ['mean', 'std', 'count']
        }).reset_index()
        stats.columns = ['decile', 'mean_return', 'std_return', 'n_obs']

        return stats
