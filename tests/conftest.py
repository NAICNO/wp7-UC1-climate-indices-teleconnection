"""
conftest.py — Shared pytest fixtures for the UC1 climate teleconnection test suite.

All fixtures produce synthetic climate-like data so tests run without any real
dataset files and without unavailable dependencies (xgboost, pycwt, ipywidgets).
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path plumbing — make the source packages importable from within the tests/
# directory without installing the project as a package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts", "lrbased_teleconnection")

for _path in (PROJECT_ROOT, SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# Synthetic climate data constants
# ---------------------------------------------------------------------------
N_ROWS = 120          # ~120 "years" — enough for lag operations without NaN blowup
N_FEATURES = 5        # number of climate predictors (excluding Time and target)
YEAR_START = 850
RANDOM_SEED = 42
TARGET_COL = "amoSSTmjjaso"
TIME_COL = "Time"


# ---------------------------------------------------------------------------
# Core fixture: synthetic DataFrame that mimics a raw CSV after load_data()
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_raw_df():
    """
    A DataFrame with a 'Time' column, one target column, and N_FEATURES
    predictor columns.  Values are seeded random floats with realistic ranges.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    years = np.arange(YEAR_START, YEAR_START + N_ROWS)
    df = pd.DataFrame({TIME_COL: years})
    df[TARGET_COL] = rng.normal(loc=10.0, scale=3.0, size=N_ROWS)
    feature_names = ["traDP", "amo3", "nao", "pdo", "enso"]
    for name in feature_names:
        df[name] = rng.normal(loc=0.0, scale=1.0, size=N_ROWS)
    return df


@pytest.fixture
def synthetic_raw_df_with_gap(synthetic_raw_df):
    """Same as synthetic_raw_df but with a 50-year gap so year_start filtering
    can be exercised."""
    df = synthetic_raw_df.copy()
    # insert a low year that should be filtered out
    old_rows = pd.DataFrame({
        TIME_COL: np.arange(700, 750),
        TARGET_COL: np.zeros(50),
        "traDP": np.zeros(50),
        "amo3": np.zeros(50),
        "nao": np.zeros(50),
        "pdo": np.zeros(50),
        "enso": np.zeros(50),
    })
    return pd.concat([old_rows, df], ignore_index=True)


# ---------------------------------------------------------------------------
# Preprocessed fixture — mirrors the output of preprocess_data()
# ---------------------------------------------------------------------------
@pytest.fixture
def preprocessed_df(synthetic_raw_df):
    """
    DataFrame already passed through preprocess_data():
    - columns reordered as [Time, target, ...features]
    - all non-Time columns normalised to [0, 100]
    - no NaN rows
    """
    from dataloader import preprocess_data
    return preprocess_data(synthetic_raw_df.copy(), TARGET_COL)


# ---------------------------------------------------------------------------
# Lagged data fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def lagged_df(preprocessed_df):
    """
    DataFrame produced by generate_lagdata() with small lag parameters so
    tests are fast.
    """
    from dataloader import generate_lagdata
    return generate_lagdata(
        init_lag=1,
        max_lag=4,
        data=preprocessed_df.copy(),
        target_feature=TARGET_COL,
        with_mean_feature=False,
    )


# ---------------------------------------------------------------------------
# CSV file fixture (uses tmp_path for isolation)
# ---------------------------------------------------------------------------
@pytest.fixture
def csv_file(tmp_path, synthetic_raw_df):
    """Writes synthetic_raw_df to a temp CSV and returns its path."""
    path = tmp_path / "climate_data.csv"
    synthetic_raw_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def csv_file_with_gap(tmp_path, synthetic_raw_df_with_gap):
    """Writes the gap-containing DataFrame to a temp CSV."""
    path = tmp_path / "climate_data_gap.csv"
    synthetic_raw_df_with_gap.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Small X/y arrays for model tests (derived from lagged_df)
# ---------------------------------------------------------------------------
@pytest.fixture
def Xy_train_test(lagged_df):
    """
    Returns (X_train, X_test, y_train, y_test) split 70/30 from lagged_df.
    Columns used are all predictors except 'Time' and the target.
    """
    X = lagged_df.drop([TIME_COL, TARGET_COL], axis=1)
    y = lagged_df[TARGET_COL]
    split = int(len(X) * 0.7)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ---------------------------------------------------------------------------
# Evaluation metrics DataFrame fixture (mirrors logged_results CSV output)
# ---------------------------------------------------------------------------
@pytest.fixture
def metrics_df():
    """
    A minimal DataFrame matching the schema that evaluate_and_plot_model() and
    plot_mae_vs_corr_reporting() expect to receive as 'filtered_df'.
    """
    records = []
    for i, forecast in enumerate([16, 17, 18, 19, 20, 21, 22, 23], start=1):
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
            "year_start": YEAR_START,
            "year_end": YEAR_START + N_ROWS - 1,
            "splitsize": 0.7,
            "max_lag": 20 + i,
            "init_lag": 15,
            "max_forecast": forecast,
            "mae_score": 5.0 + i * 0.3,
            "corr_score": 0.8 - i * 0.02,
            "n_total_features": 10 + i,
            "waveletfilter": False,
            "is_amo_guided": False,
            "selected_feature_list": str(feature_list),
            "feature_importances": str(importances),
            "selected_features": selected,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Matplotlib backend — force non-interactive so plots never pop up
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# IPython stub — matplotlib checks IPython.version_info at canvas creation
# time, and pyplot.install_repl_displayhook() calls get_ipython().
# We inject a fake version >= (8, 24) so matplotlib skips its backend fix.
# This must be done in conftest.py (executed first) so it wins over any
# module-level stubs in individual test files.
# ---------------------------------------------------------------------------
import types as _types
_ipython_stub = _types.ModuleType("IPython")
_ipython_stub.version_info = (8, 24, 0)
_ipython_stub.get_ipython = lambda: None  # matplotlib calls this in pyplot
_display_stub = _types.ModuleType("IPython.display")
_display_stub.clear_output = lambda wait=False: None
_ipython_stub.display = _display_stub
# Use setdefault so a real IPython installation is not displaced
sys.modules.setdefault("IPython", _ipython_stub)
sys.modules.setdefault("IPython.display", _display_stub)

# ---------------------------------------------------------------------------
# ipywidgets stub — used by widgets.py.  Must provide a proper .value
# attribute so test_widgets.py can test the selector logic.  The stub is
# placed here in conftest so it takes effect before any test module's own
# module-level stub.
# ---------------------------------------------------------------------------
def _make_widget_stub():
    stub = _types.ModuleType("ipywidgets")

    class _Widget:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            if not hasattr(self, "value"):
                self.value = None

        def observe(self, callback, names=None):
            pass

    for _cls_name in [
        "Dropdown", "SelectMultiple", "Text", "IntSlider",
        "IntText", "FloatSlider", "Checkbox",
    ]:
        setattr(stub, _cls_name, type(_cls_name, (_Widget,), {}))

    return stub


sys.modules.setdefault("ipywidgets", _make_widget_stub())
