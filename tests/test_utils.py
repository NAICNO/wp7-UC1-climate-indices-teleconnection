"""
test_utils.py — TDD tests for utils.py (SLURM/SSH utilities)

paramiko IS available in this environment, so we can import utils.py.
However, no real SSH connection is made.  All tests that exercise network
functions use unittest.mock to replace the paramiko client.

Functions under test:
  - connect_ssh()
  - get_available_idle_nodes(ignorelist, enforced_node)
  - run_slurm(node_name, account, ..., args, CPU_PER_TASK)

Module-level side effects in utils.py:
  - A paramiko.SSHClient is instantiated at import time.
  - Environment variables are read at import time.
  We patch these to avoid any real I/O during import or tests.
"""

import sys
import os
import types
import argparse
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# utils.py calls paramiko at module level; we patch it before importing
# ---------------------------------------------------------------------------

def _install_paramiko_stub():
    """
    Build and register a minimal paramiko stub so utils.py can be imported
    without a real SSH server.  If the real paramiko is installed, we still
    mock the client methods to avoid network calls.
    """
    try:
        import paramiko  # real one may be present
        return  # leave it in sys.modules; we'll mock at test time
    except ImportError:
        stub = types.ModuleType("paramiko")
        stub.SSHClient = type("SSHClient", (), {
            "set_missing_host_key_policy": lambda self, *a: None,
            "connect": lambda self, **kw: None,
            "exec_command": lambda self, cmd: (None, MagicMock(), MagicMock()),
            "close": lambda self: None,
        })
        stub.AutoAddPolicy = type("AutoAddPolicy", (), {})
        stub.RSAKey = type("RSAKey", (), {
            "from_private_key_file": staticmethod(lambda path, **kw: None)
        })
        sys.modules["paramiko"] = stub


_install_paramiko_stub()


# ---------------------------------------------------------------------------
# Patch the module-level paramiko client before utils is imported
# ---------------------------------------------------------------------------

import utils as umod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IGNORELIST = ["node-01", "node-02", "node-03", "node-04", "node-gpu-1", "node-gpu-2"]


def _make_sinfo_output(lines):
    """Encode a list of 'node_name cores/idle/total/...' strings to bytes."""
    return "\n".join(lines).encode()


def _mock_exec_command(output_bytes):
    """Return a mock that simulates paramiko's exec_command output."""
    stdout_mock = MagicMock()
    stdout_mock.read.return_value = output_bytes
    return MagicMock(), stdout_mock, MagicMock()


# ===========================================================================
# get_available_idle_nodes — output parsing logic
# ===========================================================================

class TestGetAvailableIdleNodes:

    def test_returns_list(self):
        lines = [
            "node-10 4/8/8/8",
            "node-11 2/4/4/4",
        ]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=[])
        assert isinstance(result, list)

    def test_returns_at_most_4_nodes(self):
        lines = [f"node-{i:02d} {i}/8/8/8" for i in range(10, 20)]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=[])
        assert len(result) <= 4

    def test_nodes_in_ignorelist_excluded(self):
        lines = [
            "node-01 4/8/8/8",   # on ignore list
            "node-10 3/6/6/6",   # should be kept
        ]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=["node-01"])
        assert "node-01" not in result
        assert "node-10" in result

    def test_empty_output_returns_empty_list(self):
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(b"")):
            result = umod.get_available_idle_nodes(ignorelist=[])
        assert result == []

    def test_all_nodes_ignored_returns_empty(self):
        lines = ["node-01 4/8/8/8", "node-02 2/4/4/4"]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=["node-01", "node-02"])
        assert result == []

    def test_node_names_are_strings(self):
        lines = ["node-10 4/8/8/8"]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=[])
        for node in result:
            assert isinstance(node, str)

    def test_default_ignorelist_excludes_gpu_nodes(self):
        lines = [
            "node-gpu-1 8/16/16/16",
            "node-20 4/8/8/8",
        ]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=IGNORELIST)
        assert "node-gpu-1" not in result

    def test_enforced_node_appended_to_command(self):
        """When enforced_node is given, grep is appended to the command string."""
        lines = ["node-15 4/8/8/8"]
        captured_cmds = []

        def _fake_exec(cmd):
            captured_cmds.append(cmd)
            return _mock_exec_command(_make_sinfo_output(lines))

        with patch.object(umod.client, "exec_command", side_effect=_fake_exec):
            umod.get_available_idle_nodes(ignorelist=[], enforced_node="node-15")

        assert len(captured_cmds) == 1
        assert "grep node-15" in captured_cmds[0]

    def test_no_enforced_node_no_grep_in_command(self):
        lines = ["node-10 4/8/8/8"]
        captured_cmds = []

        def _fake_exec(cmd):
            captured_cmds.append(cmd)
            return _mock_exec_command(_make_sinfo_output(lines))

        with patch.object(umod.client, "exec_command", side_effect=_fake_exec):
            umod.get_available_idle_nodes(ignorelist=[])

        assert "grep" not in captured_cmds[0]

    def test_exactly_four_valid_nodes_all_returned(self):
        lines = [f"node-{i:02d} 2/4/4/4" for i in range(10, 14)]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=[])
        assert len(result) == 4

    def test_result_is_shuffled_subset(self):
        """Result is a random subset; running twice may differ (probabilistic)."""
        lines = [f"node-{i:02d} 2/4/4/4" for i in range(10, 20)]
        with patch.object(umod.client, "exec_command",
                          return_value=_mock_exec_command(_make_sinfo_output(lines))):
            result = umod.get_available_idle_nodes(ignorelist=[])
        valid_names = {f"node-{i:02d}" for i in range(10, 20)}
        assert all(n in valid_names for n in result)


# ===========================================================================
# connect_ssh — verifies that paramiko client.connect is called
# ===========================================================================

class TestConnectSsh:

    def test_connect_invokes_client_connect(self):
        fake_key = MagicMock()
        with patch("utils.paramiko.RSAKey.from_private_key_file", return_value=fake_key), \
             patch.object(umod.client, "connect") as mock_connect:
            umod.connect_ssh()
        mock_connect.assert_called_once()

    def test_connect_uses_hostname_from_module(self):
        fake_key = MagicMock()
        with patch("utils.paramiko.RSAKey.from_private_key_file", return_value=fake_key), \
             patch.object(umod.client, "connect") as mock_connect:
            umod.connect_ssh()
        call_kwargs = mock_connect.call_args.kwargs
        assert call_kwargs["hostname"] == umod.hostname


# ===========================================================================
# run_slurm — verifies sbatch command construction
# ===========================================================================

class TestRunSlurm:

    def _make_args(self):
        return argparse.Namespace(
            target_feature="amoSSTmjjaso",
            modelname="LinearRegression",
            splitsize=0.6,
            data_file="dataset/noresm-f-p1000_shigh_new_jfm.csv",
            max_allowed_features=6,
            with_mean_feature=True,
        )

    def test_run_slurm_calls_exec_command(self):
        stdout_mock = MagicMock()
        stdout_mock.read.return_value = b"Submitted batch job 12345"
        with patch.object(umod.client, "exec_command",
                          return_value=(None, stdout_mock, None)) as mock_exec:
            umod.run_slurm(
                node_name="node-10",
                account="nn9999k",
                target_feature="amoSSTmjjaso",
                outputlogs_path="/tmp/logs",
                py_source="/path/to/activate",
                source_path="/path/to/project",
                args=self._make_args(),
                CPU_PER_TASK=2,
            )
        mock_exec.assert_called_once()

    def test_run_slurm_sbatch_in_command(self, capsys):
        stdout_mock = MagicMock()
        stdout_mock.read.return_value = b"Submitted batch job 99"
        with patch.object(umod.client, "exec_command",
                          return_value=(None, stdout_mock, None)):
            umod.run_slurm(
                node_name="node-10",
                account="nn9999k",
                target_feature="amoSSTmjjaso",
                outputlogs_path="/tmp/logs",
                py_source="/path/to/activate",
                source_path="/path/to/project",
                args=self._make_args(),
            )
        captured = capsys.readouterr()
        assert "sbatch" in captured.out

    def test_run_slurm_includes_target_feature_in_command(self, capsys):
        stdout_mock = MagicMock()
        stdout_mock.read.return_value = b""
        with patch.object(umod.client, "exec_command",
                          return_value=(None, stdout_mock, None)):
            umod.run_slurm(
                node_name="node-10",
                account="nn9999k",
                target_feature="amoSSTmjjaso",
                outputlogs_path="/tmp/logs",
                py_source="/path/to/activate",
                source_path="/path/to/project",
                args=self._make_args(),
            )
        captured = capsys.readouterr()
        assert "amoSSTmjjaso" in captured.out

    def test_run_slurm_includes_node_in_command(self, capsys):
        stdout_mock = MagicMock()
        stdout_mock.read.return_value = b""
        with patch.object(umod.client, "exec_command",
                          return_value=(None, stdout_mock, None)):
            umod.run_slurm(
                node_name="node-42",
                account="nn9999k",
                target_feature="amoSSTmjjaso",
                outputlogs_path="/tmp/logs",
                py_source="/path/to/activate",
                source_path="/path/to/project",
                args=self._make_args(),
            )
        captured = capsys.readouterr()
        assert "node-42" in captured.out


# ===========================================================================
# run_optimization — branching coverage
# ===========================================================================

class TestRunOptimization:
    """
    run_optimization() calls connect_ssh() then branches on execution_mode.
    We mock connect_ssh and the client to avoid real SSH.
    """

    def _make_args(self):
        return argparse.Namespace(
            target_feature="amoSSTmjjaso",
            modelname="LinearRegression",
            splitsize=0.6,
            data_file="dataset/noresm-f-p1000_shigh_new_jfm.csv",
            max_allowed_features=6,
            with_mean_feature=True,
            slurmaccount="nn9999k",
            outputlogs_path="/tmp/logs",
            py_source="/path/activate",
            source_path="/path/project",
        )

    def test_cluster_run_with_available_nodes_returns_true(self):
        sinfo_output = b"node-10 4/8/8/8"
        stdout_mock = MagicMock()
        stdout_mock.read.return_value = sinfo_output
        with patch("utils.connect_ssh"), \
             patch.object(umod.client, "exec_command",
                          return_value=(None, stdout_mock, None)), \
             patch.object(umod.client, "close"):
            result = umod.run_optimization(self._make_args(), "Cluster Run", False)
        assert result is True

    def test_cluster_run_no_nodes_returns_false(self, capsys):
        with patch("utils.connect_ssh"), \
             patch.object(umod.client, "exec_command",
                          return_value=(None, MagicMock(read=lambda: b""), None)), \
             patch.object(umod.client, "close"):
            # Make get_available_idle_nodes return empty
            with patch("utils.get_available_idle_nodes", return_value=[]):
                result = umod.run_optimization(self._make_args(), "Cluster Run", False)
        assert result is False

    def test_parallel_run_with_nodes_returns_true(self):
        sinfo_output = b"node-20 2/4/4/4"
        stdout_mock = MagicMock()
        stdout_mock.read.return_value = sinfo_output
        with patch("utils.connect_ssh"), \
             patch.object(umod.client, "exec_command",
                          return_value=(None, stdout_mock, None)), \
             patch.object(umod.client, "close"), \
             patch.dict(os.environ, {"SLURMD_NODENAME": "node-20"}):
            result = umod.run_optimization(self._make_args(), "Parallel Run", False)
        assert result is True
