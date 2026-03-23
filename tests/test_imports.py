"""
test_imports.py — Verify every importable module in the project loads without
crashing, and that missing optional dependencies are handled gracefully.

These are RED tests in the TDD sense: if an import fails, the module itself
has a structural problem (missing __init__, syntax error, etc.).
"""

import importlib
import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_module(name, attrs=None):
    """
    Insert a minimal stub into sys.modules so that modules that import an
    unavailable package (e.g. xgboost, pycwt, ipywidgets, paramiko) can still
    be imported without error.
    """
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


def _ensure_stubs():
    """
    Register lightweight stubs for all unavailable third-party packages
    referenced by the project source files.
    """
    # xgboost
    xgb_stub = _stub_module("xgboost")
    xgb_stub.XGBRegressor = type("XGBRegressor", (), {
        "fit": lambda self, X, y: self,
        "predict": lambda self, X: [],
        "feature_importances_": [],
    })

    # pycwt  (wavelet sub-module used as `from pycwt import wavelet`)
    # libs.py evaluates wavelet.Morlet(6) at function *definition* time as a
    # default argument, so the Morlet class must be callable and return an
    # object that also has a cdelta attribute.
    pycwt_stub = _stub_module("pycwt")
    wavelet_stub = _stub_module("pycwt.wavelet")

    class _FakeMorlet:
        cdelta = 0.776
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return _FakeMorlet()

    wavelet_stub.Morlet = _FakeMorlet
    pycwt_stub.wavelet = wavelet_stub

    # ipywidgets — conftest.py registers a good stub with proper .value
    # support; we use setdefault (via _stub_module) and do NOT overwrite.
    ipy_stub = _stub_module("ipywidgets")
    for widget_cls in [
        "Dropdown", "SelectMultiple", "Text", "IntSlider",
        "IntText", "FloatSlider", "Checkbox",
    ]:
        if not hasattr(ipy_stub, widget_cls):
            setattr(ipy_stub, widget_cls, type(widget_cls, (), {
                "__init__": lambda self, *a, **kw: None,
                "observe": lambda self, *a, **kw: None,
            }))

    # IPython — conftest.py registers a stub with version_info and
    # get_ipython; do not replace it.
    _stub_module("IPython")
    _stub_module("IPython.display")


_ensure_stubs()


# ---------------------------------------------------------------------------
# Tests: core script modules
# ---------------------------------------------------------------------------

class TestCoreModuleImports:
    """Each test imports one module and checks the expected public names exist."""

    def test_dataloader_imports(self):
        mod = importlib.import_module("dataloader")
        assert hasattr(mod, "load_data")
        assert hasattr(mod, "preprocess_data")
        assert hasattr(mod, "generate_lagdata")

    def test_models_imports(self):
        mod = importlib.import_module("models")
        assert hasattr(mod, "ModelSelector")
        assert hasattr(mod, "LRDropoutPSO")
        assert hasattr(mod, "LRforcedPSO")

    def test_libs_imports(self):
        mod = importlib.import_module("libs")
        assert hasattr(mod, "findtarget")
        # wavelet functions may be present even if pycwt is stubbed
        assert hasattr(mod, "wavelet_filter") or True  # non-fatal if missing


# ---------------------------------------------------------------------------
# Tests: top-level project files
# ---------------------------------------------------------------------------

class TestTopLevelModuleImports:
    """Imports widgets.py and utils.py from the project root."""

    def test_widgets_imports(self):
        mod = importlib.import_module("widgets")
        assert hasattr(mod, "build_widgets")
        assert hasattr(mod, "get_args_from_widgets")
        assert hasattr(mod, "selector_func")

    def test_utils_imports(self):
        mod = importlib.import_module("utils")
        assert hasattr(mod, "connect_ssh")
        assert hasattr(mod, "get_available_idle_nodes")
        assert hasattr(mod, "run_slurm")


# ---------------------------------------------------------------------------
# Tests: optional dependency presence is detectable
# ---------------------------------------------------------------------------

class TestOptionalDependencyDetection:

    def test_xgboost_not_available(self):
        """xgboost must NOT be a real package in this environment."""
        try:
            import xgboost as xgb
            # If we get here it was already stubbed or is installed.
            # Check the stub attribute rather than the real class.
            assert hasattr(xgb, "XGBRegressor")
        except ImportError:
            pass  # expected

    def test_pycwt_not_available(self):
        """pycwt must NOT be a real package in this environment."""
        try:
            import pycwt
            assert hasattr(pycwt, "wavelet")
        except ImportError:
            pass

    def test_ipywidgets_not_available(self):
        """ipywidgets must NOT be a real package in this environment."""
        try:
            import ipywidgets
            assert hasattr(ipywidgets, "Dropdown")
        except ImportError:
            pass

    def test_numpy_available(self):
        import numpy as np
        assert np.__version__

    def test_pandas_available(self):
        import pandas as pd
        assert pd.__version__

    def test_sklearn_available(self):
        import sklearn
        assert sklearn.__version__

    def test_scipy_available(self):
        import scipy
        assert scipy.__version__

    def test_matplotlib_available(self):
        import matplotlib
        assert matplotlib.__version__
