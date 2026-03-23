"""
test_evaluation.py — TDD tests for plotting utilities.

We test the plotting code paths that can be exercised without real dataset
files by directly calling plot_mae_vs_corr_reporting() and the
plot_scatter_plot() helper, using the Agg matplotlib backend so no display
window ever opens.

evaluate_and_plot_model() depends on actual CSV files and wavelet filters, so
we test it via an integration-style test that mocks the file-loading functions
it calls.
"""

import sys
import types
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Stubs for unavailable packages (must be set up before importing project code)
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
    xgb.XGBRegressor = type("XGBRegressor", (), {
        "__init__": lambda self, **kw: None,
        "fit": lambda self, X, y: self,
        "predict": lambda self, X: np.zeros(len(X)),
        "feature_importances_": np.array([]),
    })

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

from plotting import plot_mae_vs_corr_reporting  # noqa: E402

TARGET_COL = "amoSSTmjjaso"
TIME_COL = "Time"


# ===========================================================================
# Helper: build a minimal metrics DataFrame
# ===========================================================================

def _make_metrics_df(n=6):
    records = []
    for i in range(n):
        forecast = 16 + i
        feature_list = [f"{forecast}_lag_{TARGET_COL}", f"{forecast}_lag_nao"]
        importances = [0.6, 0.4]
        selected = (
            ", ".join(feature_list)
            + f"_{10 + i}_{forecast}"
        )
        records.append({
            "model": "LinearRegression",
            "num_irrelevant": i,
            "max_allowed_features": 4,
            "target_feature": TARGET_COL,
            "dataset": "shigh",
            "year_start": 850,
            "year_end": 969,
            "splitsize": 0.7,
            "max_lag": 20 + i,
            "init_lag": 15,
            "max_forecast": forecast,
            "mae_score": 5.0 + i * 0.5,
            "corr_score": 0.9 - i * 0.05,
            "n_total_features": 10 + i,
            "waveletfilter": False,
            "is_amo_guided": False,
            "selected_feature_list": str(feature_list),
            "feature_importances": str(importances),
            "selected_features": selected,
        })
    return pd.DataFrame(records)


# ===========================================================================
# plot_mae_vs_corr_reporting
# ===========================================================================

class TestPlotMaeVsCorrReporting:

    def _make_axes(self):
        fig, ax = plt.subplots()
        return fig, ax

    def test_runs_without_error(self):
        df = _make_metrics_df()
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        plt.close("all")

    def test_axes_has_lines(self):
        df = _make_metrics_df()
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        assert len(ax.lines) >= 1
        plt.close("all")

    def test_axes_x_label_set(self):
        df = _make_metrics_df()
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        assert ax.get_xlabel() == "Selected Features"
        plt.close("all")

    def test_axes_y_label_set(self):
        df = _make_metrics_df()
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        assert "MAE" in ax.get_ylabel()
        plt.close("all")

    def test_title_contains_target_feature(self):
        df = _make_metrics_df()
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        assert TARGET_COL in ax.get_title()
        plt.close("all")

    def test_works_with_single_row(self):
        df = _make_metrics_df(n=1)
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        plt.close("all")

    def test_filters_to_target_feature_only(self):
        """Only rows matching target_feature should influence the plot."""
        df = _make_metrics_df()
        df2 = df.copy()
        df2["target_feature"] = "OTHER"
        combined = pd.concat([df, df2], ignore_index=True)
        fig, ax = self._make_axes()
        # Should not raise even when other features are present
        plot_mae_vs_corr_reporting(combined, TARGET_COL, width=15, ax=ax)
        plt.close("all")

    def test_mae_line_is_blue(self):
        df = _make_metrics_df()
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        mae_line_color = ax.lines[0].get_color()
        assert mae_line_color == "blue"
        plt.close("all")

    def test_grid_is_on(self):
        df = _make_metrics_df()
        fig, ax = self._make_axes()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        assert ax.get_xgridlines()[0].get_visible() or ax.xaxis._gridOnMajor
        plt.close("all")


# ===========================================================================
# plot_scatter_plot (internal helper) — imported directly from evaluation
# ===========================================================================

class TestPlotScatterPlot:
    """
    The function plot_scatter_plot lives inside evaluation.py.  It is not
    exported from the package, but we can test it by importing it directly.
    """

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        # evaluation.py imports libs.py which in turn tries to import xgboost
        # and pycwt — stubs are already in place from module level.
        import importlib
        # evaluation.py uses `from libs import *` which needs the stubs active
        import evaluation as eval_mod
        self.eval_mod = eval_mod

    def _make_axes(self):
        fig, ax = plt.subplots()
        return fig, ax

    def test_scatter_plot_runs_without_error(self):
        fig, ax = self._make_axes()
        y_test = pd.Series(np.linspace(0, 100, 20))
        y_pred = np.linspace(5, 95, 20)
        labels = {"actual": ["Actual", "blue"], "predicted": ["Predicted", "red"]}
        self.eval_mod.plot_scatter_plot(ax, y_test, y_pred, labels)
        plt.close("all")

    def test_scatter_plot_creates_collection(self):
        fig, ax = self._make_axes()
        y_test = pd.Series(np.linspace(0, 100, 20))
        y_pred = np.linspace(5, 95, 20)
        labels = {"actual": ["Actual", "blue"], "predicted": ["Predicted", "red"]}
        self.eval_mod.plot_scatter_plot(ax, y_test, y_pred, labels)
        assert len(ax.collections) >= 1
        plt.close("all")

    def test_scatter_plot_sets_x_label(self):
        fig, ax = self._make_axes()
        y_test = pd.Series(np.linspace(0, 50, 10))
        y_pred = np.linspace(1, 49, 10)
        labels = {"actual": ["Actual", "blue"], "predicted": ["Predicted", "red"]}
        self.eval_mod.plot_scatter_plot(ax, y_test, y_pred, labels)
        assert "Actual" in ax.get_xlabel()
        plt.close("all")

    def test_scatter_plot_sets_y_label(self):
        fig, ax = self._make_axes()
        y_test = pd.Series(np.linspace(0, 50, 10))
        y_pred = np.linspace(1, 49, 10)
        labels = {"actual": ["Actual", "blue"], "predicted": ["Predicted", "red"]}
        self.eval_mod.plot_scatter_plot(ax, y_test, y_pred, labels)
        assert "Predicted" in ax.get_ylabel()
        plt.close("all")


# ===========================================================================
# evaluate_and_plot_model — integration test with mocked data loading
# ===========================================================================

class TestEvaluateAndPlotModelIntegration:
    """
    evaluate_and_plot_model() receives load_data, preprocess_data, and
    generate_lagdata as function arguments — it does NOT look them up from
    its own module namespace.  We pass stub callables directly, so no
    patching of the evaluation module is needed.
    """

    def _build_mock_data(self):
        """Return a synthetic preprocessed DataFrame."""
        rng = np.random.default_rng(7)
        n = 80
        df = pd.DataFrame({
            TIME_COL: np.arange(850, 850 + n),
            TARGET_COL: rng.uniform(0, 100, n),
            "nao": rng.uniform(0, 100, n),
        })
        return df

    def _build_filtered_df_for_test(self, mock_data):
        """
        Build a single-row filtered_df that matches the mock_data columns.
        The selected feature column must actually exist after generate_lagdata.
        """
        from dataloader import generate_lagdata

        lagged = generate_lagdata(1, 3, mock_data.copy(), TARGET_COL, with_mean_feature=False)
        real_col = [c for c in lagged.columns if "_lag_" in c][0]
        selected = f"{real_col}_{len(lagged.columns)}_{1}"

        return pd.DataFrame([{
            "model": "LinearRegression",
            "num_irrelevant": 0,
            "max_allowed_features": 1,
            "target_feature": TARGET_COL,
            "dataset": "shigh",
            "year_start": 850,
            "year_end": 929,
            "splitsize": 0.7,
            "max_lag": 3,
            "init_lag": 1,
            "max_forecast": 1,
            "mae_score": 8.0,
            "corr_score": 0.6,
            "n_total_features": 2,
            "waveletfilter": False,
            "is_amo_guided": False,
            "selected_feature_list": str([real_col]),
            "feature_importances": str([0.9]),
            "selected_features": selected,
        }])

    def test_evaluate_and_plot_runs_without_error(self):
        import evaluation as eval_mod
        from dataloader import generate_lagdata

        mock_data = self._build_mock_data()
        filtered_df = self._build_filtered_df_for_test(mock_data)

        fig, ax = plt.subplots()

        def _mock_load_data(path, del_feats, year_start=None):
            return mock_data.copy()

        def _mock_preprocess(data, target):
            return data.copy()

        info = eval_mod.evaluate_and_plot_model(
            filtered_df=filtered_df,
            target_feature=TARGET_COL,
            delete_features=[],
            generate_lagdata=generate_lagdata,
            load_data=_mock_load_data,
            preprocess_data=_mock_preprocess,
            ModelSelector=__import__("models").ModelSelector,
            ax=ax,
            waveletfilters=[False],
        )
        assert isinstance(info, str)
        assert "MAE" in info
        plt.close("all")

    def test_evaluate_returns_string_with_feature_importances(self):
        import evaluation as eval_mod
        from dataloader import generate_lagdata

        mock_data = self._build_mock_data()
        filtered_df = self._build_filtered_df_for_test(mock_data)

        fig, ax = plt.subplots()

        def _mock_load_data(path, del_feats, year_start=None):
            return mock_data.copy()

        def _mock_preprocess(data, target):
            return data.copy()

        info = eval_mod.evaluate_and_plot_model(
            filtered_df=filtered_df,
            target_feature=TARGET_COL,
            delete_features=[],
            generate_lagdata=generate_lagdata,
            load_data=_mock_load_data,
            preprocess_data=_mock_preprocess,
            ModelSelector=__import__("models").ModelSelector,
            ax=ax,
            waveletfilters=[False],
        )
        assert "Feature Importances" in info
        plt.close("all")
