"""
Asset Pricing Models for Weigert et al. (RFS 2023)
Implements all models used in the linear vs. nonlinear horse-race.
"""

from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb


class PLSWrapper(PLSRegression):
    """Wrapper for PLSRegression to ensure predict() returns 1D array (n_samples,)."""
    def fit(self, X, y):
        self.n_components = min(self.n_components, X.shape[0], X.shape[1])
        return super().fit(X, y)

    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


class AssetPricingModels:
    """All machine learning models used in Weigert et al. (RFS 2023) horse-race."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def get_lasso(self, alpha: float = 0.001):
        """LASSO (L1 regularization)"""
        return Lasso(alpha=alpha, max_iter=10000, random_state=self.random_state)

    def get_elasticnet(self, alpha: float = 0.001, l1_ratio: float = 0.5):
        """Elastic Net (L1 + L2 regularization)"""
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                         max_iter=10000, random_state=self.random_state)

    def get_pcr(self, n_components: int = 10):
        """Principal Component Regression (PCR)"""
        # Wrap PCA to auto-cap n_components at feature count
        class AdaptivePCA(PCA):
            def fit_transform(self, X, y=None):
                self.n_components = min(self.n_components, X.shape[1])
                return super().fit_transform(X, y)
            def transform(self, X):
                return super().transform(X)
        return Pipeline([
            ('pca', AdaptivePCA(n_components=n_components)),
            ('ols', LinearRegression())
        ])

    def get_pls(self, n_components: int = 10):
        """Partial Least Squares (PLS) — wrapped to return 1D predictions"""
        return PLSWrapper(n_components=n_components)

    def get_random_forest(self, n_estimators: int = 200, max_depth: int = None):
        """Random Forest"""
        return RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   random_state=self.random_state)

    def get_gbm(self, n_estimators: int = 200, learning_rate: float = 0.1):
        """Gradient Boosted Trees (GBM)"""
        return GradientBoostingRegressor(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       random_state=self.random_state)

    def get_xgboost(self, n_estimators: int = 200, learning_rate: float = 0.05):
        """XGBoost"""
        return xgb.XGBRegressor(n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              random_state=self.random_state)

    def get_lightgbm(self, n_estimators: int = 200, learning_rate: float = 0.05):
        """LightGBM (verbose disabled)"""
        return lgb.LGBMRegressor(n_estimators=n_estimators,
                               learning_rate=learning_rate,
                               random_state=self.random_state,
                               verbose=-1)

    def get_neuralnet(self, hidden_layer_sizes=(64, 32), max_iter=500):
        """Feed-forward Neural Network"""
        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                          max_iter=max_iter,
                          random_state=self.random_state)

    def get_ensemble(self):
        """Voting Ensemble (average of selected strong models)"""
        estimators = [
            ('lasso', self.get_lasso()),
            ('rf', self.get_random_forest()),
            ('xgb', self.get_xgboost()),
            ('lgb', self.get_lightgbm()),
        ]
        return VotingRegressor(estimators=estimators)

    def get_model(self, name: str, **kwargs):
        """Unified interface to get any model by name (recommended)"""
        name = name.lower().strip().replace('_', '').replace('-', '')
        if name == 'lasso':
            return self.get_lasso(**kwargs)
        elif name == 'elasticnet':
            return self.get_elasticnet(**kwargs)
        elif name in ['pcr', 'principalcomponentregression']:
            return self.get_pcr(**kwargs)
        elif name == 'pls':
            return self.get_pls(**kwargs)
        elif name in ['randomforest', 'rf']:
            return self.get_random_forest(**kwargs)
        elif name in ['gbm', 'gradientboosting']:
            return self.get_gbm(**kwargs)
        elif name == 'xgboost':
            return self.get_xgboost(**kwargs)
        elif name == 'lightgbm':
            return self.get_lightgbm(**kwargs)
        elif name in ['nn', 'neuralnet', 'mlp']:
            return self.get_neuralnet(**kwargs)
        elif name in ['ensemble', 'voting']:
            return self.get_ensemble()
        else:
            raise ValueError(f"Unknown model: {name}\n"
                           f"Available: lasso, elasticnet, pcr, pls, randomforest, "
                           f"gbm, xgboost, lightgbm, nn, ensemble")
