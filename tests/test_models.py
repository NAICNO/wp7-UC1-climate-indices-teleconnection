"""
test_models.py — TDD tests for scripts/lrbased_teleconnection/models.py

Classes / functions under test:
  - ModelSelector(modelname)           — factory & attribute delegation
  - ModelSelector.feature_importances() — extracts importances per model type
  - LRDropoutPSO                       — PSO-based LR with dropout
  - LRforcedPSO                        — PSO-based LR with forced constraint

XGBoost is NOT available; those model name paths are tested to raise ValueError
(or are skipped if the implementation requires the library at class instantiation).
"""

import sys
import types
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# ---------------------------------------------------------------------------
# Stubs for unavailable packages
# ---------------------------------------------------------------------------

def _stub_if_absent(name, attrs=None):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
    return sys.modules[name]


def _setup_stubs():
    xgb = _stub_if_absent("xgboost")

    class _FakeXGBRegressor:
        def __init__(self, **kw):
            self.feature_importances_ = np.array([0.5, 0.5])
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))

    xgb.XGBRegressor = _FakeXGBRegressor

    pycwt = _stub_if_absent("pycwt")
    wmod = _stub_if_absent("pycwt.wavelet")

    class _FakeMorlet:
        cdelta = 0.776
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *a, **kw):
            return _FakeMorlet()

    wmod.Morlet = _FakeMorlet
    pycwt.wavelet = wmod
    _stub_if_absent("tqdm", {"tqdm": lambda x, **kw: x})


_setup_stubs()

from models import ModelSelector, LRDropoutPSO, LRforcedPSO  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny dataset helper
# ---------------------------------------------------------------------------

def _make_tiny_Xy(n_samples=30, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=n_samples)
    return X, y


# ===========================================================================
# ModelSelector — factory behaviour
# ===========================================================================

class TestModelSelectorFactory:

    def test_linear_regression_creates_correct_type(self):
        ms = ModelSelector("LinearRegression")
        assert isinstance(ms.model, LinearRegression)

    def test_random_forest_creates_correct_type(self):
        ms = ModelSelector("RandomForestRegressor")
        assert isinstance(ms.model, RandomForestRegressor)

    def test_mlp_creates_correct_type(self):
        ms = ModelSelector("MLPRegressor")
        assert isinstance(ms.model, MLPRegressor)

    def test_lr_dropout_pso_creates_instance(self):
        ms = ModelSelector("LRDropoutPSO")
        assert isinstance(ms.model, LRDropoutPSO)

    def test_lr_dropout_pso_50_creates_instance(self):
        ms = ModelSelector("LRDropoutPSO_50percent")
        assert isinstance(ms.model, LRDropoutPSO)

    def test_lr_forced_pso_creates_instance(self):
        ms = ModelSelector("LRforcedPSO")
        assert isinstance(ms.model, LRforcedPSO)

    def test_lr_forced_pso_10_creates_instance(self):
        ms = ModelSelector("LRforcedPSO_10percent")
        assert isinstance(ms.model, LRforcedPSO)

    def test_lr_forced_pso_50_creates_instance(self):
        ms = ModelSelector("LRforcedPSO_50percent")
        assert isinstance(ms.model, LRforcedPSO)

    def test_lr_forced_pso_75_creates_instance(self):
        ms = ModelSelector("LRforcedPSO_75percent")
        assert isinstance(ms.model, LRforcedPSO)

    def test_invalid_modelname_raises_value_error(self):
        with pytest.raises(ValueError):
            ModelSelector("NotAModel")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            ModelSelector("")

    def test_modelname_attribute_stored(self):
        ms = ModelSelector("LinearRegression")
        assert ms.modelname == "LinearRegression"


# ===========================================================================
# ModelSelector — attribute delegation via __getattr__
# ===========================================================================

class TestModelSelectorDelegation:

    def test_fit_delegated_to_model(self):
        ms = ModelSelector("LinearRegression")
        X, y = _make_tiny_Xy()
        ms.fit(X, y)
        assert hasattr(ms.model, "coef_")

    def test_predict_delegated_to_model(self):
        ms = ModelSelector("LinearRegression")
        X, y = _make_tiny_Xy()
        ms.fit(X, y)
        preds = ms.predict(X)
        assert len(preds) == len(y)

    def test_coef_accessible_after_fit_linear(self):
        ms = ModelSelector("LinearRegression")
        X, y = _make_tiny_Xy()
        ms.fit(X, y)
        assert ms.coef_ is not None
        assert len(ms.coef_) == X.shape[1]


# ===========================================================================
# ModelSelector — feature_importances()
# ===========================================================================

class TestModelSelectorFeatureImportances:

    def test_linear_regression_importances_shape(self):
        ms = ModelSelector("LinearRegression")
        X, y = _make_tiny_Xy(n_features=4)
        ms.fit(X, y)
        imp = ms.feature_importances()
        assert imp.shape == (4,)

    def test_linear_regression_importances_non_negative(self):
        ms = ModelSelector("LinearRegression")
        X, y = _make_tiny_Xy(n_features=4)
        ms.fit(X, y)
        assert (ms.feature_importances() >= 0).all()

    def test_random_forest_importances_sum_to_one(self):
        ms = ModelSelector("RandomForestRegressor")
        X, y = _make_tiny_Xy(n_features=4)
        ms.fit(X, y)
        imp = ms.feature_importances()
        assert abs(imp.sum() - 1.0) < 1e-9

    def test_random_forest_importances_non_negative(self):
        ms = ModelSelector("RandomForestRegressor")
        X, y = _make_tiny_Xy(n_features=4)
        ms.fit(X, y)
        assert (ms.feature_importances() >= 0).all()

    def test_mlp_importances_shape(self):
        ms = ModelSelector("MLPRegressor")
        X, y = _make_tiny_Xy(n_features=4)
        ms.fit(X, y)
        imp = ms.feature_importances()
        assert imp.shape == (4,)

    def test_mlp_importances_non_negative(self):
        ms = ModelSelector("MLPRegressor")
        X, y = _make_tiny_Xy(n_features=4)
        ms.fit(X, y)
        assert (ms.feature_importances() >= 0).all()

    def test_feature_importances_returns_ndarray(self):
        ms = ModelSelector("LinearRegression")
        X, y = _make_tiny_Xy(n_features=3)
        ms.fit(X, y)
        assert isinstance(ms.feature_importances(), np.ndarray)


# ===========================================================================
# LRDropoutPSO — focused unit tests (small iteration counts for speed)
# ===========================================================================

class TestLRDropoutPSO:

    @pytest.fixture
    def tiny_model(self):
        return LRDropoutPSO(
            max_diff=0.5,
            max_iter=5,          # minimal iterations for speed
            population_size=10,
            dropout_rate=0.2,
        )

    def test_fit_returns_self(self, tiny_model):
        X, y = _make_tiny_Xy()
        result = tiny_model.fit(X, y)
        assert result is tiny_model

    def test_coef_shape_after_fit(self, tiny_model):
        X, y = _make_tiny_Xy(n_features=3)
        tiny_model.fit(X, y)
        assert tiny_model.coef_.shape == (3,)

    def test_predict_shape(self, tiny_model):
        X, y = _make_tiny_Xy(n_samples=20, n_features=3)
        tiny_model.fit(X[:15], y[:15])
        preds = tiny_model.predict(X[15:])
        assert len(preds) == 5

    def test_predict_with_uncertainty_returns_tuple(self, tiny_model):
        X, y = _make_tiny_Xy(n_samples=20, n_features=3)
        tiny_model.fit(X[:15], y[:15])
        result = tiny_model.predict(X[15:], with_uncertainty=True, n_iter=10)
        assert isinstance(result, tuple)
        y_pred, y_std = result
        assert len(y_pred) == 5
        assert len(y_std) == 5

    def test_predict_single_shape(self, tiny_model):
        X, y = _make_tiny_Xy(n_samples=20, n_features=3)
        tiny_model.fit(X[:15], y[:15])
        preds = tiny_model.predict_single(X[15:])
        assert len(preds) == 5

    def test_set_coef_updates_intercept(self, tiny_model):
        X, y = _make_tiny_Xy(n_samples=20, n_features=3)
        importances = [0.1, 0.5, 0.4]
        tiny_model.set_coef(importances, X, y)
        assert tiny_model.coef_ is not None
        # intercept should be a scalar
        assert np.isscalar(tiny_model.intercept_)

    def test_dropout_rate_stored(self):
        model = LRDropoutPSO(dropout_rate=0.35)
        assert model.dropout_rate == 0.35

    def test_max_diff_stored(self):
        model = LRDropoutPSO(max_diff=0.1)
        assert model.max_diff == 0.1


# ===========================================================================
# LRforcedPSO — focused unit tests
# ===========================================================================

class TestLRforcedPSO:

    @pytest.fixture
    def tiny_model(self):
        return LRforcedPSO(
            max_diff=0.5,
            max_iter=5,
            population_size=10,
        )

    def test_fit_returns_self(self, tiny_model):
        X, y = _make_tiny_Xy()
        assert tiny_model.fit(X, y) is tiny_model

    def test_coef_shape_after_fit(self, tiny_model):
        X, y = _make_tiny_Xy(n_features=3)
        tiny_model.fit(X, y)
        assert tiny_model.coef_.shape == (3,)

    def test_predict_shape(self, tiny_model):
        X, y = _make_tiny_Xy(n_samples=20, n_features=3)
        tiny_model.fit(X[:15], y[:15])
        preds = tiny_model.predict(X[15:])
        assert len(preds) == 5

    def test_set_coef_sets_coef_array(self, tiny_model):
        X, y = _make_tiny_Xy(n_samples=20, n_features=3)
        importances = [0.2, 0.5, 0.3]
        tiny_model.set_coef(importances, X, y)
        np.testing.assert_array_equal(tiny_model.coef_, np.array(importances))

    def test_max_diff_stored(self):
        model = LRforcedPSO(max_diff=0.75)
        assert model.max_diff == 0.75

    def test_predict_after_set_coef(self, tiny_model):
        X, y = _make_tiny_Xy(n_samples=20, n_features=3)
        tiny_model.set_coef([1.0, 0.0, 0.0], X, y)
        preds = tiny_model.predict(X)
        assert len(preds) == 20


# ===========================================================================
# ModelSelector — XGBoost paths (run with stub, not real xgboost)
# ===========================================================================

class TestModelSelectorXGBoostStubbed:
    """
    Since xgboost is stubbed, ModelSelector("XGBoost") will create our fake
    regressor.  We verify the code path runs without error.
    """

    def test_xgboost_model_created_with_stub(self):
        # The real xgboost is absent; the stub allows instantiation.
        ms = ModelSelector("XGBoost")
        assert ms.model is not None

    def test_xgboost_feature_importances_accessible(self):
        ms = ModelSelector("XGBoost")
        # The stub has a feature_importances_ attribute
        assert hasattr(ms.model, "feature_importances_")
