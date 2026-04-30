"""
SHAP Model Explainability — Bali et al. (RFS, 2023)
Provides model-agnostic and model-specific SHAP explanations
for all estimators used in the asset pricing horse-race.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline


# ── Model-family detection ───────────────────────────────────────────

_TREE_TYPES: tuple = (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

_LINEAR_TYPES: tuple = (
    Lasso,
    ElasticNet,
    LinearRegression,
)

try:
    import xgboost as xgb
    _TREE_TYPES = (*_TREE_TYPES, xgb.XGBRegressor)
except ImportError:
    pass

try:
    import lightgbm as lgb
    _TREE_TYPES = (*_TREE_TYPES, lgb.LGBMRegressor)
except ImportError:
    pass


def _unwrap_pipeline(model):
    """If model is a sklearn Pipeline, return the final estimator."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


class SHAPExplainer:
    """Generate SHAP explanations for any model in the horse-race."""

    def __init__(
        self,
        output_dir: str = ".",
        background_size: int = 200,
        figsize: tuple = (12, 8),
        dpi: int = 200,
    ):
        """
        Parameters
        ----------
        output_dir      : Directory where plots are saved.
        background_size : Number of background samples for KernelExplainer.
        figsize         : Matplotlib figure size.
        dpi             : Resolution of saved figures.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.background_size = background_size
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    #  Select the most efficient explainer for the model family
    # ------------------------------------------------------------------
    def _build_explainer(self, model, X_background: pd.DataFrame):
        core = _unwrap_pipeline(model)

        if isinstance(core, _TREE_TYPES):
            return shap.TreeExplainer(core)

        if isinstance(core, _LINEAR_TYPES):
            return shap.LinearExplainer(core, X_background)

        # Fallback: KernelExplainer (model-agnostic, slower)
        background = shap.sample(X_background, min(self.background_size, len(X_background)))
        return shap.KernelExplainer(model.predict, background)

    # ------------------------------------------------------------------
    #  Compute SHAP values
    # ------------------------------------------------------------------
    def compute_shap_values(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> shap.Explanation:
        """
        Compute SHAP values for X_test.

        Parameters
        ----------
        model   : A fitted estimator (sklearn / xgb / lgb / Pipeline).
        X_train : Training data used as background reference.
        X_test  : Samples to explain.

        Returns
        -------
        shap.Explanation object.
        """
        explainer = self._build_explainer(model, X_train)
        shap_values = explainer(X_test)

        # KernelExplainer returns a raw numpy array; wrap it
        if not isinstance(shap_values, shap.Explanation):
            shap_values = shap.Explanation(
                values=np.array(shap_values),
                data=X_test.values,
                feature_names=list(X_test.columns),
            )

        return shap_values

    # ------------------------------------------------------------------
    #  Feature importance (mean |SHAP|) as a sorted DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def feature_importance(shap_values: shap.Explanation) -> pd.DataFrame:
        """Return mean absolute SHAP value per feature, sorted descending."""
        vals = np.abs(shap_values.values)
        importance = pd.DataFrame({
            'feature': shap_values.feature_names,
            'mean_abs_shap': vals.mean(axis=0),
        })
        return importance.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    #  Plots
    # ------------------------------------------------------------------
    def summary_plot(
        self,
        shap_values: shap.Explanation,
        X_test: pd.DataFrame,
        model_name: str = "Model",
        show: bool = False,
    ) -> Path:
        """Beeswarm summary plot. Returns path to saved figure."""
        shap.summary_plot(
            shap_values,
            X_test,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(self.figsize)
        fig.suptitle(f"SHAP Summary — {model_name}", y=1.02)
        fig.tight_layout()

        path = self.output_dir / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        if not show:
            plt.close(fig)

        return path

    def bar_plot(
        self,
        shap_values: shap.Explanation,
        model_name: str = "Model",
        max_display: int = 20,
        show: bool = False,
    ) -> Path:
        """Global feature importance bar plot. Returns path to saved figure."""
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(self.figsize)
        fig.suptitle(f"SHAP Feature Importance — {model_name}", y=1.02)
        fig.tight_layout()

        path = self.output_dir / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        if not show:
            plt.close(fig)

        return path

    # ------------------------------------------------------------------
    #  Convenience: compute + plot + return importance table
    # ------------------------------------------------------------------
    def explain_model(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str = "Model",
        show: bool = False,
    ) -> dict:
        """
        Full explanation pipeline: SHAP values, summary plot, bar plot,
        and feature importance table.

        Returns
        -------
        dict with keys: 'shap_values', 'importance', 'summary_path', 'bar_path'.
        """
        shap_values = self.compute_shap_values(model, X_train, X_test)
        importance = self.feature_importance(shap_values)
        summary_path = self.summary_plot(shap_values, X_test, model_name, show=show)
        bar_path = self.bar_plot(shap_values, model_name, show=show)

        return {
            'shap_values': shap_values,
            'importance': importance,
            'summary_path': summary_path,
            'bar_path': bar_path,
        }
