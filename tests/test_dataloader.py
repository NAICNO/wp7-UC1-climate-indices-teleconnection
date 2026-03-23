"""
test_dataloader.py — TDD tests for scripts/lrbased_teleconnection/dataloader.py

Functions under test:
  - load_data(data_file, delete_features, year_start=None)
  - preprocess_data(data, target_feature)
  - generate_lagdata(init_lag, max_lag, data, target_feature, with_mean_feature=False)

RED -> GREEN -> REFACTOR cycle followed strictly.
All tests use synthetic data from conftest fixtures; no real dataset files needed.
"""

import pytest
import numpy as np
import pandas as pd

from dataloader import load_data, preprocess_data, generate_lagdata

TARGET_COL = "amoSSTmjjaso"
TIME_COL = "Time"


# ===========================================================================
# load_data
# ===========================================================================

class TestLoadData:

    def test_returns_dataframe(self, csv_file):
        df = load_data(csv_file, delete_features=[])
        assert isinstance(df, pd.DataFrame)

    def test_all_rows_loaded_when_no_year_start(self, csv_file, synthetic_raw_df):
        df = load_data(csv_file, delete_features=[])
        assert len(df) == len(synthetic_raw_df)

    def test_year_start_filters_rows(self, csv_file_with_gap, synthetic_raw_df):
        """Rows with Time < year_start must be excluded."""
        year_start = 850
        df = load_data(csv_file_with_gap, delete_features=[], year_start=year_start)
        assert (df[TIME_COL] >= year_start).all()

    def test_year_start_none_keeps_all_rows(self, csv_file_with_gap, synthetic_raw_df_with_gap):
        df = load_data(csv_file_with_gap, delete_features=[], year_start=None)
        assert len(df) == len(synthetic_raw_df_with_gap)

    def test_delete_features_removes_columns(self, csv_file):
        df = load_data(csv_file, delete_features=["traDP", "amo3"])
        assert "traDP" not in df.columns
        assert "amo3" not in df.columns

    def test_delete_single_feature(self, csv_file):
        df = load_data(csv_file, delete_features=["nao"])
        assert "nao" not in df.columns

    def test_delete_empty_list_keeps_all_columns(self, csv_file, synthetic_raw_df):
        df = load_data(csv_file, delete_features=[])
        assert set(df.columns) == set(synthetic_raw_df.columns)

    def test_index_reset_after_year_filter(self, csv_file_with_gap):
        df = load_data(csv_file_with_gap, delete_features=[], year_start=850)
        assert df.index[0] == 0
        assert list(df.index) == list(range(len(df)))

    def test_time_column_present(self, csv_file):
        df = load_data(csv_file, delete_features=[])
        assert TIME_COL in df.columns

    def test_target_column_present(self, csv_file):
        df = load_data(csv_file, delete_features=[])
        assert TARGET_COL in df.columns

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_data(str(tmp_path / "does_not_exist.csv"), delete_features=[])

    def test_year_start_exact_boundary_included(self, csv_file_with_gap):
        """A row exactly at year_start must be included (>= not >)."""
        df = load_data(csv_file_with_gap, delete_features=[], year_start=850)
        assert 850 in df[TIME_COL].values

    def test_combined_filter_and_delete(self, csv_file_with_gap):
        df = load_data(
            csv_file_with_gap,
            delete_features=["traDP"],
            year_start=850,
        )
        assert "traDP" not in df.columns
        assert (df[TIME_COL] >= 850).all()


# ===========================================================================
# preprocess_data
# ===========================================================================

class TestPreprocessData:

    def test_returns_dataframe(self, synthetic_raw_df):
        result = preprocess_data(synthetic_raw_df.copy(), TARGET_COL)
        assert isinstance(result, pd.DataFrame)

    def test_target_column_is_second_column(self, synthetic_raw_df):
        result = preprocess_data(synthetic_raw_df.copy(), TARGET_COL)
        assert result.columns[0] == TIME_COL
        assert result.columns[1] == TARGET_COL

    def test_normalization_range_0_to_100(self, synthetic_raw_df):
        result = preprocess_data(synthetic_raw_df.copy(), TARGET_COL)
        numeric_cols = result.columns[1:]  # skip Time
        for col in numeric_cols:
            assert result[col].min() >= 0.0 - 1e-9, f"{col} min below 0"
            assert result[col].max() <= 100.0 + 1e-9, f"{col} max above 100"

    def test_normalization_min_is_zero(self, synthetic_raw_df):
        result = preprocess_data(synthetic_raw_df.copy(), TARGET_COL)
        for col in result.columns[1:]:
            assert abs(result[col].min()) < 1e-9 or result[col].min() == 0.0

    def test_normalization_max_is_hundred(self, synthetic_raw_df):
        result = preprocess_data(synthetic_raw_df.copy(), TARGET_COL)
        for col in result.columns[1:]:
            assert abs(result[col].max() - 100.0) < 1e-9 or result[col].max() == 0.0

    def test_no_nan_rows_in_output(self, synthetic_raw_df):
        result = preprocess_data(synthetic_raw_df.copy(), TARGET_COL)
        assert not result.isnull().any().any()

    def test_constant_column_becomes_zero(self):
        """A column with a single unique value should be set to 0.0."""
        df = pd.DataFrame({
            TIME_COL: [1, 2, 3],
            TARGET_COL: [5.0, 5.0, 5.0],
            "feat": [7.0, 7.0, 7.0],
        })
        result = preprocess_data(df, TARGET_COL)
        assert (result[TARGET_COL] == 0.0).all()
        assert (result["feat"] == 0.0).all()

    def test_numeric_columns_cast_to_float(self, synthetic_raw_df):
        # Introduce an integer column
        df = synthetic_raw_df.copy()
        df["int_feat"] = [1] * len(df)
        result = preprocess_data(df, TARGET_COL)
        assert result["int_feat"].dtype == float

    def test_time_column_not_normalized(self, synthetic_raw_df):
        """Time values must remain as original integer years."""
        result = preprocess_data(synthetic_raw_df.copy(), TARGET_COL)
        assert result[TIME_COL].iloc[0] == synthetic_raw_df[TIME_COL].iloc[0]

    def test_drops_nan_rows_from_input(self):
        df = pd.DataFrame({
            TIME_COL: [1, 2, 3],
            TARGET_COL: [np.nan, 2.0, 3.0],
            "feat": [1.0, np.nan, 3.0],
        })
        result = preprocess_data(df, TARGET_COL)
        # The nan-containing rows are dropped; remaining row is row index 2
        assert len(result) < 3

    def test_column_order_time_target_then_features(self, synthetic_raw_df):
        # Scramble the column order
        cols = list(synthetic_raw_df.columns)
        cols.insert(0, cols.pop(cols.index(TARGET_COL)))
        shuffled = synthetic_raw_df[cols]
        result = preprocess_data(shuffled.copy(), TARGET_COL)
        assert result.columns.tolist()[0] == TIME_COL
        assert result.columns.tolist()[1] == TARGET_COL


# ===========================================================================
# generate_lagdata
# ===========================================================================

class TestGenerateLagdata:

    def test_returns_dataframe(self, preprocessed_df):
        result = generate_lagdata(1, 4, preprocessed_df.copy(), TARGET_COL)
        assert isinstance(result, pd.DataFrame)

    def test_lagged_columns_created(self, preprocessed_df):
        result = generate_lagdata(1, 4, preprocessed_df.copy(), TARGET_COL)
        lag_cols = [c for c in result.columns if "_lag_" in c]
        assert len(lag_cols) > 0

    def test_lag_column_naming_convention(self, preprocessed_df):
        """Columns must follow '{lag}_lag_{feature}' naming."""
        result = generate_lagdata(2, 5, preprocessed_df.copy(), TARGET_COL)
        lag_cols = [c for c in result.columns if "_lag_" in c]
        for col in lag_cols:
            parts = col.split("_lag_")
            assert len(parts) == 2
            assert parts[0].isdigit()

    def test_target_column_retained(self, preprocessed_df):
        result = generate_lagdata(1, 4, preprocessed_df.copy(), TARGET_COL)
        assert TARGET_COL in result.columns

    def test_time_column_retained(self, preprocessed_df):
        result = generate_lagdata(1, 4, preprocessed_df.copy(), TARGET_COL)
        assert TIME_COL in result.columns

    def test_non_target_original_features_dropped(self, preprocessed_df):
        """Original (non-lagged) predictor columns must be removed."""
        result = generate_lagdata(1, 4, preprocessed_df.copy(), TARGET_COL)
        for col in preprocessed_df.columns[2:]:  # skip Time and target
            assert col not in result.columns

    def test_no_nan_rows_in_output(self, preprocessed_df):
        result = generate_lagdata(1, 4, preprocessed_df.copy(), TARGET_COL)
        assert not result.isnull().any().any()

    def test_row_count_reduced_by_max_lag(self, preprocessed_df):
        """NaN rows produced by shifting are dropped so row count is smaller."""
        result = generate_lagdata(1, 5, preprocessed_df.copy(), TARGET_COL)
        assert len(result) < len(preprocessed_df)

    def test_number_of_lag_columns_matches_parameters(self, preprocessed_df):
        """
        With init_lag=1, max_lag=4 the loop runs for lags 1, 2, 3.
        Each lag produces one column per non-Time predictor.
        """
        init_lag, max_lag = 1, 4
        n_predictors = len(preprocessed_df.columns) - 1  # exclude Time
        result = generate_lagdata(init_lag, max_lag, preprocessed_df.copy(), TARGET_COL)
        lag_cols = [c for c in result.columns if "_lag_" in c]
        expected = (max_lag - init_lag) * n_predictors
        assert len(lag_cols) == expected

    def test_with_mean_feature_adds_median_columns(self, preprocessed_df):
        """with_mean_feature=True should add extra median columns."""
        result_no_mean = generate_lagdata(1, 5, preprocessed_df.copy(), TARGET_COL, with_mean_feature=False)
        result_with_mean = generate_lagdata(1, 5, preprocessed_df.copy(), TARGET_COL, with_mean_feature=True)
        assert len(result_with_mean.columns) > len(result_no_mean.columns)

    def test_with_mean_feature_median_columns_named(self, preprocessed_df):
        result = generate_lagdata(1, 5, preprocessed_df.copy(), TARGET_COL, with_mean_feature=True)
        median_cols = [c for c in result.columns if "median" in c]
        assert len(median_cols) > 0

    def test_zero_lag_range_produces_only_original_target(self, preprocessed_df):
        """init_lag == max_lag: loop doesn't execute, so no lag columns."""
        result = generate_lagdata(5, 5, preprocessed_df.copy(), TARGET_COL)
        lag_cols = [c for c in result.columns if "_lag_" in c]
        assert len(lag_cols) == 0

    def test_output_index_is_integer_index(self, preprocessed_df):
        """
        generate_lagdata() calls dropna() but does not reset the index.
        The index is therefore a subset of the original integer positions,
        not necessarily starting from 0.  We verify it contains only integers
        and is monotonically increasing.
        """
        result = generate_lagdata(1, 4, preprocessed_df.copy(), TARGET_COL)
        idx = list(result.index)
        assert all(isinstance(i, (int, np.integer)) for i in idx)
        assert idx == sorted(idx), "Index must be monotonically increasing"

    def test_large_lag_reduces_rows_substantially(self, preprocessed_df):
        """Using a large max_lag should leave fewer rows."""
        result_small = generate_lagdata(1, 3, preprocessed_df.copy(), TARGET_COL)
        result_large = generate_lagdata(1, 10, preprocessed_df.copy(), TARGET_COL)
        assert len(result_large) < len(result_small)
