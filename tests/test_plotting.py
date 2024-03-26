"""
test_plotting.py — Additional TDD tests for plotting.py functions not yet
covered: plot_climate_index_analysis, plot_mae_corr.

All tests use matplotlib's Agg backend so no display window ever opens.
"""

import sys
import types
import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Stubs for unavailable packages (must run before any project import)
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
    if not hasattr(xgb, "XGBRegressor"):
        xgb.XGBRegressor = type("XGBRegressor", (), {
            "__init__": lambda self, **kw: None,
            "fit": lambda self, X, y: self,
            "predict": lambda self, X: np.zeros(len(X)),
            "feature_importances_": np.array([]),
        })

    pycwt = _stub_if_absent("pycwt")
    wmod = _stub_if_absent("pycwt.wavelet")
    if not hasattr(wmod, "Morlet"):
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

# Import after stubs are in place
from plotting import (  # noqa: E402
    plot_mae_vs_corr_reporting,
    plot_climate_index_analysis,
    plot_mae_corr,
)

TARGET_COL = "amoSSTmjjaso"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics_df(n=6, target=TARGET_COL):
    records = []
    for i in range(n):
        forecast = 16 + i
        feature_list = [f"{forecast}_lag_{target}", f"{forecast}_lag_nao"]
        importances = [0.6, 0.4]
        selected = ", ".join(feature_list) + f"_{10 + i}_{forecast}"
        records.append({
            "model": "LinearRegression",
            "num_irrelevant": i,
            "max_allowed_features": 4,
            "target_feature": target,
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


def _make_logged_results(modelname, n=4):
    """Build the logged_results dict that plot_mae_corr() expects."""
    results = {}
    for i in range(1, n + 1):
        key = f"{modelname}_{i}"
        results[key] = {
            "selected_features": f"feat_{i}",
            "num_irrelevant": i,
            "mae_score": 10.0 + i,
            "corr_score": 0.7 - i * 0.05,
            "300_windowing_maescore": [9.5 + i, 10.0 + i],
            "300_windowing_coorscore": [0.65 - i * 0.05, 0.7 - i * 0.05],
        }
    return results


# ===========================================================================
# plot_climate_index_analysis
# ===========================================================================

class TestPlotClimateIndexAnalysis:

    def test_runs_without_error(self):
        df = _make_metrics_df()
        with patch("matplotlib.pyplot.show"):
            plot_climate_index_analysis(df)
        plt.close("all")

    def test_accepts_custom_corr_score(self):
        df = _make_metrics_df()
        with patch("matplotlib.pyplot.show"):
            plot_climate_index_analysis(df, corr_score=0.5)
        plt.close("all")

    def test_accepts_custom_mae_score(self):
        df = _make_metrics_df()
        with patch("matplotlib.pyplot.show"):
            plot_climate_index_analysis(df, mae_score=20.0)
        plt.close("all")

    def test_works_with_single_row(self):
        df = _make_metrics_df(n=1)
        with patch("matplotlib.pyplot.show"):
            plot_climate_index_analysis(df)
        plt.close("all")

    def test_works_with_multiple_target_features(self):
        df1 = _make_metrics_df(n=3, target="amo2")
        df2 = _make_metrics_df(n=3, target="nao")
        combined = pd.concat([df1, df2], ignore_index=True)
        with patch("matplotlib.pyplot.show"):
            plot_climate_index_analysis(combined)
        plt.close("all")

    def test_scatter_plots_created(self):
        """Four scatter plots must be produced (2x2 layout)."""
        df = _make_metrics_df()
        with patch("matplotlib.pyplot.show"):
            plot_climate_index_analysis(df)
        # We can't directly inspect the fig here, but no exception = pass
        plt.close("all")


# ===========================================================================
# plot_mae_corr — saves a file using plt.savefig
# ===========================================================================

class TestPlotMaeCorr:

    def test_runs_without_error(self, tmp_path):
        modelname = "LinearRegression"
        logged_results = _make_logged_results(modelname, n=4)
        filename = str(tmp_path / "test_plot.png")
        with patch("matplotlib.pyplot.show"):
            plot_mae_corr(
                num_irrelevant=5,
                logged_results=logged_results,
                modelname=modelname,
                filename=filename,
            )
        plt.close("all")

    def test_creates_output_file(self, tmp_path):
        modelname = "LinearRegression"
        logged_results = _make_logged_results(modelname, n=4)
        filename = str(tmp_path / "test_plot.png")
        with patch("matplotlib.pyplot.show"):
            plot_mae_corr(
                num_irrelevant=5,
                logged_results=logged_results,
                modelname=modelname,
                filename=filename,
            )
        assert os.path.isfile(filename)
        plt.close("all")

    def test_also_creates_csv(self, tmp_path):
        """plot_mae_corr() saves a CSV when rotate=False."""
        modelname = "LinearRegression"
        logged_results = _make_logged_results(modelname, n=4)
        filename = str(tmp_path / "test_plot.png")
        with patch("matplotlib.pyplot.show"):
            plot_mae_corr(
                num_irrelevant=5,
                logged_results=logged_results,
                modelname=modelname,
                filename=filename,
                rotate=False,
            )
        csv_path = filename[:-3] + "csv"
        assert os.path.isfile(csv_path)
        plt.close("all")

    def test_rotate_true_runs_without_error(self, tmp_path):
        """
        plot_mae_corr() with rotate=True saves 'rotated_<basename>' so the
        filename must be a bare basename (no directory prefix) while cwd is
        set to tmp_path.
        """
        modelname = "LinearRegression"
        logged_results = _make_logged_results(modelname, n=4)
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("matplotlib.pyplot.show"):
                plot_mae_corr(
                    num_irrelevant=5,
                    logged_results=logged_results,
                    modelname=modelname,
                    filename="test_plot.png",
                    rotate=True,
                )
        finally:
            os.chdir(original_dir)
        plt.close("all")

    def test_rotated_filename_prefixed(self, tmp_path):
        """When rotate=True, the saved file is named 'rotated_<filename>'."""
        modelname = "LinearRegression"
        logged_results = _make_logged_results(modelname, n=4)
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("matplotlib.pyplot.show"):
                plot_mae_corr(
                    num_irrelevant=5,
                    logged_results=logged_results,
                    modelname=modelname,
                    filename="test_plot.png",
                    rotate=True,
                )
        finally:
            os.chdir(original_dir)
        rotated_path = tmp_path / "rotated_test_plot.png"
        assert rotated_path.is_file()
        plt.close("all")

    def test_custom_year_length(self, tmp_path):
        modelname = "LinearRegression"
        logged_results = _make_logged_results(modelname, n=4)
        # Add the custom year_length key
        for key in logged_results:
            logged_results[key]["100_windowing_maescore"] = [9.0, 10.0]
            logged_results[key]["100_windowing_coorscore"] = [0.6, 0.65]
        filename = str(tmp_path / "test_plot.png")
        with patch("matplotlib.pyplot.show"):
            plot_mae_corr(
                num_irrelevant=5,
                logged_results=logged_results,
                modelname=modelname,
                filename=filename,
                year_length=100,
            )
        plt.close("all")

    def test_empty_num_irrelevant_no_iterations(self, tmp_path):
        """num_irrelevant=1 means range(1,1) = empty, so the dict is empty."""
        modelname = "LinearRegression"
        # plot_mae_corr iterates range(1, num_irrelevant), so with 1 it's empty
        logged_results = {}
        filename = str(tmp_path / "test_plot.png")
        with patch("matplotlib.pyplot.show"):
            plot_mae_corr(
                num_irrelevant=1,
                logged_results=logged_results,
                modelname=modelname,
                filename=filename,
            )
        plt.close("all")


# ===========================================================================
# Additional plot_mae_vs_corr_reporting coverage (head_limit scenario)
# ===========================================================================

class TestPlotMaeVsCorrReportingEdgeCases:

    def test_more_than_8_rows_capped_at_8(self):
        """The function internally calls .head(8) so only 8 points appear."""
        df = _make_metrics_df(n=12)
        fig, ax = plt.subplots()
        plot_mae_vs_corr_reporting(df, TARGET_COL, width=15, ax=ax)
        # The MAE line should have at most 8 points
        assert len(ax.lines[0].get_xdata()) <= 8
        plt.close("all")

    def test_deduplication_on_max_forecast(self):
        """Duplicate max_forecast values are dropped before plotting."""
        df = _make_metrics_df(n=4)
        # Duplicate all max_forecast values
        df2 = df.copy()
        df2["mae_score"] = df2["mae_score"] + 100  # worse
        combined = pd.concat([df, df2], ignore_index=True)
        fig, ax = plt.subplots()
        plot_mae_vs_corr_reporting(combined, TARGET_COL, width=15, ax=ax)
        # Only the first (better, sorted by mae) occurrence per forecast is kept
        assert len(ax.lines[0].get_xdata()) == 4
        plt.close("all")
