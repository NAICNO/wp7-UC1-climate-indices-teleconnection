"""
test_widgets.py — TDD tests for widgets.py

ipywidgets is NOT available in this environment.  All tests are written to:
  1. Skip the tests that require a real ipywidgets installation.
  2. Test the pure-Python logic (selector_func branching, get_args_from_widgets
     argument namespace construction) using a lightweight stub.

The stub replicates the minimal widget interface needed by widgets.py:
  - A .value attribute
  - An .observe() method (no-op)
"""

import sys
import types
import argparse
import pytest


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------
try:
    import ipywidgets as _real_ipywidgets
    _real_ipywidgets.Dropdown  # probe a real attribute
    IPYWIDGETS_REAL = True
except (ImportError, AttributeError):
    IPYWIDGETS_REAL = False


# ---------------------------------------------------------------------------
# Build a stub if necessary
# ---------------------------------------------------------------------------

def _install_widget_stubs():
    """
    Register a minimal ipywidgets stub in sys.modules so that widgets.py
    can be imported.  Each widget class stores its kwargs and exposes .value.
    """
    stub = types.ModuleType("ipywidgets")

    class _Widget:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            # Ensure .value is always accessible
            if not hasattr(self, "value"):
                self.value = None

        def observe(self, callback, names=None):
            pass  # no-op

    for cls_name in [
        "Dropdown", "SelectMultiple", "Text", "IntSlider",
        "IntText", "FloatSlider", "Checkbox",
    ]:
        setattr(stub, cls_name, type(cls_name, (_Widget,), {}))

    sys.modules["ipywidgets"] = stub
    return stub


if not IPYWIDGETS_REAL:
    _install_widget_stubs()

import widgets as wmod  # noqa: E402  — must come after stub


# ===========================================================================
# selector_func  — pure branching logic, no display needed
# ===========================================================================

class TestSelectorFunc:

    def test_multi_select_false_returns_dropdown_value_first_item(self):
        options = ["a", "b", "c"]
        widget = wmod.selector_func(False, options, ["b", "c"], "Label:")
        # single-select: value is the first element of the value list
        assert widget.value == "b"

    def test_multi_select_true_returns_select_multiple(self):
        options = ["a", "b", "c"]
        widget = wmod.selector_func(True, options, ["a", "b"], "Label:")
        # multi-select: value is the whole tuple/list
        assert widget.value == ["a", "b"]

    def test_multi_select_false_description_stored(self):
        widget = wmod.selector_func(False, ["x"], ["x"], "MyLabel:")
        assert widget.description == "MyLabel:"

    def test_multi_select_true_description_stored(self):
        widget = wmod.selector_func(True, ["x", "y"], ["x"], "MyLabel:")
        assert widget.description == "MyLabel:"


# ===========================================================================
# build_widgets  — returns a tuple of 12 widget objects
# ===========================================================================

class TestBuildWidgets:

    def test_returns_tuple_of_12(self):
        result = wmod.build_widgets(
            target_features=["amo2", "amo3"],
            data_files=["dataset/noresm-f-p1000_shigh_new_jfm.csv"],
            modelnames=["LinearRegression"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 12

    def test_all_elements_have_value_attribute(self):
        result = wmod.build_widgets(
            target_features=["amo2"],
            data_files=["dataset/noresm-f-p1000_shigh_new_jfm.csv"],
            modelnames=["LinearRegression"],
        )
        for widget in result:
            assert hasattr(widget, "value"), f"{widget!r} has no .value"

    def test_multi_select_false_default(self):
        result = wmod.build_widgets(
            target_features=["amo2", "amo3"],
            data_files=["file1.csv", "file2.csv"],
            modelnames=["LinearRegression", "RandomForestRegressor"],
            multi_select=False,
        )
        assert len(result) == 12

    def test_multi_select_true(self):
        result = wmod.build_widgets(
            target_features=["amo2", "amo3"],
            data_files=["file1.csv"],
            modelnames=["LinearRegression"],
            multi_select=True,
        )
        assert len(result) == 12


# ===========================================================================
# get_args_from_widgets  — namespace construction
# ===========================================================================

class TestGetArgsFromWidgets:

    def _make_stub_widgets(self, modelname="LinearRegression"):
        """Build a tuple of 12 fake widgets with predefined .value attributes."""
        class _W:
            def __init__(self, value):
                self.value = value

        return (
            _W("dataset/noresm-f-p1000_shigh_new_jfm.csv"),  # data_file_widget
            _W("amo2"),                                        # target_feature_widget
            _W("traDP,amo3"),                                  # delete_features_widget
            _W(modelname),                                     # modelname_widget
            _W(6),                                             # max_allowed_features
            _W(100),                                           # end_lag
            _W(50),                                            # n_ensembles
            _W(850),                                           # main_year_start
            _W(0.6),                                           # splitsize
            _W(True),                                          # with_mean_feature
            _W(False),                                         # with_wavelet_filter
            _W(20),                                            # step_lag
        )

    def test_returns_namespace(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert isinstance(args, argparse.Namespace)

    def test_data_file_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.data_file == "dataset/noresm-f-p1000_shigh_new_jfm.csv"

    def test_target_feature_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.target_feature == "amo2"

    def test_delete_features_split_on_comma(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.delete_features == ["traDP", "amo3"]

    def test_modelname_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.modelname == "LinearRegression"

    def test_linear_regression_forces_n_ensembles_to_1(self):
        widgets_tuple = self._make_stub_widgets(modelname="LinearRegression")
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.n_ensembles == 1

    def test_non_lr_model_keeps_n_ensembles(self):
        widgets_tuple = self._make_stub_widgets(modelname="RandomForestRegressor")
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.n_ensembles == 50

    def test_splitsize_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.splitsize == 0.6

    def test_with_mean_feature_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.with_mean_feature is True

    def test_with_wavelet_filter_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.with_wavelet_filter is False

    def test_step_lag_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.step_lag == 20

    def test_max_allowed_features_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.max_allowed_features == 6

    def test_end_lag_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.end_lag == 100

    def test_main_year_start_extracted(self):
        widgets_tuple = self._make_stub_widgets()
        args = wmod.get_args_from_widgets(*widgets_tuple)
        assert args.main_year_start == 850


# ===========================================================================
# Real ipywidgets tests (skipped when not available)
# ===========================================================================

@pytest.mark.skipif(not IPYWIDGETS_REAL, reason="ipywidgets not installed")
class TestRealIpywidgetsWidgets:

    def test_create_dropdown_returns_dropdown(self):
        w = wmod.create_dropdown(["a", "b"], "a", "Test:")
        import ipywidgets as ipw
        assert isinstance(w, ipw.Dropdown)

    def test_create_int_slider_returns_int_slider(self):
        w = wmod.create_int_slider(5, 1, 10, 1, "Slider:")
        import ipywidgets as ipw
        assert isinstance(w, ipw.IntSlider)

    def test_create_checkbox_returns_checkbox(self):
        w = wmod.create_checkbox(True, "Check:")
        import ipywidgets as ipw
        assert isinstance(w, ipw.Checkbox)

    def test_create_float_slider_returns_float_slider(self):
        w = wmod.create_float_slider(0.5, 0.0, 1.0, 0.1, "Float:")
        import ipywidgets as ipw
        assert isinstance(w, ipw.FloatSlider)
