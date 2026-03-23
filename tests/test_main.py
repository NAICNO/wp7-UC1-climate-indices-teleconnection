"""
test_main.py — End-to-end TDD tests for the main() pipeline function.

main() in main.py:
  1. Loads a CSV via load_data()
  2. Preprocesses via preprocess_data()
  3. Iterates over lag steps, generating lagged features
  4. Trains a ModelSelector model, extracts feature importances
  5. Saves CSV results to disk and returns the filepath (when is_jupyter_run=True)

We exercise main() end-to-end with:
  - A tiny synthetic CSV in tmp_path (no real dataset needed)
  - Minimal lag parameters (end_lag = init_lag + 1 = 16 so 1 iteration)
  - n_ensembles = 1 for speed
  - is_jupyter_run = True so the function returns a filepath

XGBoost / pycwt stubs are set up before importing main.
"""

import sys
import os
import types
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


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
    xgb.XGBRegressor = type("XGBRegressor", (), {
        "__init__": lambda self, **kw: None,
        "fit": lambda self, X, y: self,
        "predict": lambda self, X: np.zeros(len(X)),
        "feature_importances_": np.array([0.5, 0.5]),
    })

    pycwt = _stub_if_absent("pycwt")
    wmod = _stub_if_absent("pycwt.wavelet")
    class _FM:
        cdelta = 0.776
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *a, **kw):
            return _FM()

    wmod.Morlet = _FM
    pycwt.wavelet = wmod

    _stub_if_absent("tqdm", {"tqdm": lambda x, **kw: x})

    # IPython.display.clear_output used inside main() when is_jupyter_run=True
    ipython = _stub_if_absent("IPython")
    display_mod = _stub_if_absent("IPython.display")
    display_mod.clear_output = lambda wait=False: None
    ipython.display = display_mod


_setup_stubs()

from main import main  # noqa: E402

TARGET_COL = "amoSSTmjjaso"
TIME_COL = "Time"
N_ROWS = 80
YEAR_START = 850


# ---------------------------------------------------------------------------
# Synthetic CSV fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_csv(tmp_path):
    """
    Write a tiny synthetic CSV inside a 'dataset/' subdirectory that uses the
    naming convention expected by main().

    main.py derives dataset_name via:
        data_file.replace("dataset/noresm-f-p1000_", "").replace("_new_jfm.csv", "")

    So we place the file at dataset/noresm-f-p1000_testdata_new_jfm.csv inside
    tmp_path, and pass the *relative* path so the replace() strips it correctly,
    leaving dataset_name = "testdata".
    """
    rng = np.random.default_rng(42)
    years = np.arange(YEAR_START, YEAR_START + N_ROWS)
    df = pd.DataFrame({
        TIME_COL: years,
        TARGET_COL: rng.uniform(5, 15, N_ROWS),
        "nao": rng.normal(0, 1, N_ROWS),
        "pdo": rng.normal(0, 1, N_ROWS),
        "enso": rng.normal(0, 1, N_ROWS),
    })
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "noresm-f-p1000_testdata_new_jfm.csv"
    df.to_csv(path, index=False)
    # Return the RELATIVE path that main() expects
    return "dataset/noresm-f-p1000_testdata_new_jfm.csv"


# ---------------------------------------------------------------------------
# Helper to run main() with tiny parameters from within tmp_path
# (main() calls os.makedirs and writes CSVs relative to cwd)
# ---------------------------------------------------------------------------

def _run_main(data_file, tmp_path, modelname="LinearRegression", **overrides):
    """
    Runs main() with cwd set to tmp_path so that the relative data_file path
    and the results/ directory created by main() both resolve inside tmp_path.
    """
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        kwargs = dict(
            data_file=data_file,
            target_feature=TARGET_COL,
            delete_features=[],
            modelname=modelname,
            max_allowed_features=2,
            end_lag=16,          # init_lag=15 inside main, so 1 iteration
            n_ensembles=1,
            splitsize=0.6,
            with_mean_feature=False,
            main_year_start=None,
            step_lag=10,
            with_wavelet_filter=False,
            is_jupyter_run=True,
        )
        kwargs.update(overrides)
        result = main(**kwargs)
        # Convert the relative path returned by main() to an absolute path
        # while we are still chdir-ed into tmp_path.
        if result is not None:
            result = os.path.abspath(result)
        return result
    finally:
        os.chdir(original_dir)


# ===========================================================================
# End-to-end smoke tests
# ===========================================================================

class TestMainPipeline:

    def test_main_returns_csv_filepath_when_jupyter_run(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        assert isinstance(result, str)
        assert result.endswith(".csv")

    def test_returned_filepath_exists(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        assert os.path.isfile(result)

    def test_output_csv_is_non_empty(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        assert len(df) >= 1

    def test_output_csv_has_expected_columns(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        required_columns = {
            "model", "target_feature", "mae_score", "corr_score",
            "max_forecast", "selected_feature_list",
        }
        assert required_columns.issubset(set(df.columns))

    def test_mae_score_is_non_negative(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        assert (df["mae_score"] >= 0).all()

    def test_modelname_recorded_in_output(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        assert (df["model"] == "LinearRegression").all()

    def test_target_feature_recorded_in_output(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        assert (df["target_feature"] == TARGET_COL).all()

    def test_splitsize_recorded_in_output(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        assert (df["splitsize"] == 0.6).all()

    def test_results_directory_created(self, tiny_csv, tmp_path):
        _run_main(tiny_csv, tmp_path)
        results_dir = tmp_path / "results"
        assert results_dir.is_dir()

    def test_main_runs_with_random_forest(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path, modelname="RandomForestRegressor")
        assert os.path.isfile(result)

    def test_main_runs_with_mlp(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path, modelname="MLPRegressor")
        assert os.path.isfile(result)

    def test_main_with_year_start_filter(self, tiny_csv, tmp_path):
        """year_start in output reflects the min Time value after filtering."""
        result = _run_main(tiny_csv, tmp_path, main_year_start=860)
        df = pd.read_csv(result)
        # year_start logged by main() is min(Time_data), which is >= 860
        assert (df["year_start"] >= 860).all()

    def test_main_with_mean_feature_enabled(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path, with_mean_feature=True)
        assert os.path.isfile(result)

    def test_feature_importances_recorded(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        assert "feature_importances" in df.columns
        assert df["feature_importances"].iloc[0] != ""

    def test_corr_score_between_neg1_and_1(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path)
        df = pd.read_csv(result)
        assert (df["corr_score"] >= -1.0).all()
        assert (df["corr_score"] <= 1.0).all()


# ===========================================================================
# Edge cases for main()
# ===========================================================================

class TestMainEdgeCases:

    def test_nonexistent_data_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            _run_main("dataset/does_not_exist.csv", tmp_path)

    def test_invalid_modelname_raises_value_error(self, tiny_csv, tmp_path):
        with pytest.raises(ValueError):
            _run_main(tiny_csv, tmp_path, modelname="NotARealModel")

    def test_max_allowed_features_one(self, tiny_csv, tmp_path):
        """With only 1 allowed feature the pipeline should still complete."""
        result = _run_main(tiny_csv, tmp_path, max_allowed_features=1)
        assert os.path.isfile(result)

    def test_splitsize_80_percent(self, tiny_csv, tmp_path):
        result = _run_main(tiny_csv, tmp_path, splitsize=0.8)
        df = pd.read_csv(result)
        assert abs(df["splitsize"].iloc[0] - 0.8) < 1e-9

    def test_multiple_delete_features_prefixes_filename(self, tiny_csv, tmp_path):
        """With >1 delete_features, the CSV filename is prefixed with 'del<N>_'."""
        # The CSV file must exist so we can inspect the parent directory
        result = _run_main(
            tiny_csv,
            tmp_path,
            delete_features=["nao", "pdo"],  # 2 features -> triggers line 124
        )
        assert os.path.isfile(result)
        # Filename should contain the 'del2_' prefix
        assert "del2_" in os.path.basename(result)
