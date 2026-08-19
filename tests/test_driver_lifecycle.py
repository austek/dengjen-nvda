# coding: utf-8
"""
Tests for aio module lifecycle resilience and re-initialization behavior.
"""

import ast
import asyncio
import gc
import os
import glob
import subprocess
import threading
import time
import types
import warnings
import importlib.util
from types import SimpleNamespace

import pytest

_STRESS_THREADS = 8
_STRESS_ITERATIONS = 40

_TESTS_DIR = os.path.dirname(__file__)
_PKG_DIR = os.path.join(
    _TESTS_DIR, "..", "addon", "synthDrivers", "dengjen_neural_voices"
)
_AIO_PATH = os.path.join(_PKG_DIR, "aio.py")
_GRPC_CLIENT_PATH = os.path.join(_PKG_DIR, "grpc_client", "__init__.py")

_LOOP_THREAD_NAME = "piper4nvda_asyncio"


def _load_module_function(path, name, namespace):
    """Exec a single top-level function against a stubbed namespace.

    grpc_client imports grpc and NVDA globals, so it cannot be imported here.
    """
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    for node in ast.parse(source, filename=path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            exec(ast.get_source_segment(source, node), namespace)
            return namespace[name]
    raise LookupError(f"{name} not found in {path}")


class _FakeAioChannel:
    """Stands in for grpc.aio.Channel, whose close() is a coroutine."""

    def __init__(self):
        self.close_awaited = False
        self.closed_on_loop = None

    async def _close(self):
        self.close_awaited = True
        self.closed_on_loop = asyncio.get_running_loop()

    def close(self):
        return self._close()


def _never_awaited_warnings(caught):
    return [w for w in caught if "never awaited" in str(w.message)]


def _load_real_aio():
    spec = importlib.util.spec_from_file_location(
        "dengjen_neural_voices.aio_real", _AIO_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _package_sources():
    """(path, AST) for every first-party module, skipping vendored libraries."""
    for path in sorted(glob.glob(os.path.join(_PKG_DIR, "**", "*.py"), recursive=True)):
        if f"{os.sep}lib{os.sep}" in path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            yield path, ast.parse(f.read(), filename=path)


def _top_level_names(tree):
    names = set()
    body = list(tree.body)
    for node in tree.body:
        if isinstance(node, ast.With):
            body.extend(node.body)
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            names.update(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


aio = _load_real_aio()


def _settled_loop_thread_count(timeout=2):
    """Loop-thread count once stopped threads have had a chance to exit."""
    deadline = time.monotonic() + timeout
    while True:
        count = len(
            [t for t in threading.enumerate() if t.name == _LOOP_THREAD_NAME]
        )
        if count <= 1 or time.monotonic() > deadline:
            return count
        time.sleep(0.05)


class TestAioLifecycle:

    def setup_method(self):
        aio.initialize()

    def teardown_method(self):
        aio.terminate()

    def test_initialize_is_idempotent(self):
        aio.initialize()
        aio.initialize()
        assert aio.ENGINE.event_loop is not None
        assert aio.ENGINE.event_loop.is_running()
        assert aio.ENGINE.executor is not None

    def test_reinitialization_after_terminate(self):
        # Shutdown loop and executor
        aio.terminate()
        assert aio.ENGINE.executor is None

        # Re-initialize
        aio.initialize()
        assert aio.ENGINE.event_loop is not None
        assert aio.ENGINE.event_loop.is_running()
        assert aio.ENGINE.executor is not None

    def test_asyncio_coroutine_to_concurrent_future_resurrects_stopped_loop(self):
        aio.terminate()

        @aio.asyncio_coroutine_to_concurrent_future
        async def dummy_coro():
            return 42

        fut = dummy_coro()
        assert fut.result(timeout=5) == 42

    def test_run_in_executor_resurrects_stopped_loop(self):
        aio.terminate()

        def sync_fn(val):
            return val * 2

        @aio.asyncio_coroutine_to_concurrent_future
        async def run_test():
            return await aio.run_in_executor(sync_fn, 21)

        fut = run_test()
        assert fut.result(timeout=5) == 42

    def test_task_creation_resolves_the_live_loop_after_reinitialization(self):
        aio.terminate()

        async def spoken():
            return "spoken"

        @aio.asyncio_coroutine_to_concurrent_future
        async def create_task_like_process_speech():
            loop = aio.asyncio.get_running_loop()
            assert loop is aio.ENGINE.event_loop
            return await loop.create_task(spoken())

        assert create_task_like_process_speech().result(timeout=5) == "spoken"

    def test_terminate_closes_and_clears_the_loop(self):
        loop = aio.ENGINE.event_loop
        assert loop is not None

        aio.terminate()

        assert loop.is_closed()
        assert aio.ENGINE.event_loop is None

    def test_repeated_cycles_do_not_accumulate_loop_threads(self):
        for _ in range(10):
            aio.terminate()
            aio.initialize()

        assert _settled_loop_thread_count() == 1

    def test_concurrent_ensure_running_does_not_orphan_loops(self):
        errors = []
        barrier = threading.Barrier(_STRESS_THREADS + 1)

        def worker():
            barrier.wait()
            for _ in range(_STRESS_ITERATIONS):
                try:
                    aio.ensure_running()
                except Exception as exc:
                    errors.append(repr(exc))
                    continue
                if aio.ENGINE.event_loop is None:
                    errors.append("ensure_running() left the event loop unset")

        threads = [threading.Thread(target=worker) for _ in range(_STRESS_THREADS)]
        for thread in threads:
            thread.start()

        barrier.wait()
        for _ in range(15):
            aio.terminate()
            time.sleep(0.01)
        for thread in threads:
            thread.join()

        aio.ensure_running()
        assert errors == []
        assert _settled_loop_thread_count() == 1


class TestAioGlobalsDoNotLeakMutableState:
    """Guards against re-introducing scattered mutable globals: the event
    loop/executor live on aio.ENGINE (a stable singleton whose properties
    are read fresh on every access), never as rebindable module-level
    names a caller could import by value and have go stale."""

    def test_the_old_scattered_globals_are_gone(self):
        assert not hasattr(aio, "ASYNCIO_EVENT_LOOP")
        assert not hasattr(aio, "THREADED_EXECUTOR")
        assert not hasattr(aio, "ASYNCIO_LOOP_THREAD")
        assert not hasattr(aio, "EXECUTOR_IS_SHUTDOWN")

    def test_no_module_imports_a_loop_or_executor_by_value_from_aio(self):
        offenders = []
        for path, tree in _package_sources():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level == 0:
                    continue
                if (node.module or "").split(".")[-1] != "aio":
                    continue
                for alias in node.names:
                    name = alias.name
                    # UPPER_CASE only: aio's helper functions (run_in_executor,
                    # etc.) also contain "loop"/"executor" but are stable
                    # function references, not mutable state.
                    if name.isupper() and ("LOOP" in name or "EXECUTOR" in name):
                        offenders.append(f"{os.path.basename(path)}: {name}")

        assert offenders == [], (
            f"{offenders} import loop/executor state by value from aio; read "
            "aio.ENGINE.event_loop / aio.ENGINE.executor fresh instead, or use "
            "asyncio.get_running_loop()."
        )


class TestGrpcChannelTeardown:

    def setup_method(self):
        aio.initialize()

    def teardown_method(self):
        aio.terminate()

    def _close_channel_with(self, channel):
        namespace = {
            "asyncio": asyncio,
            "aio": aio,
            "log": types.SimpleNamespace(debug=lambda *a, **k: None),
            "CHANNEL": channel,
            "CHANNEL_PORT": 50051,
            "SONATA_GRPC_SERVICE": object(),
            "CHANNEL_CLOSE_TIMEOUT": 5,
        }
        close_channel = _load_module_function(
            _GRPC_CLIENT_PATH, "close_channel", namespace
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            close_channel()
            gc.collect()
        return namespace, caught

    def test_dropping_the_close_coroutine_is_detectable(self):
        """Self-check: the assertions below are only meaningful if this warns."""
        channel = _FakeAioChannel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            channel.close()
            gc.collect()

        assert _never_awaited_warnings(caught)

    def test_close_channel_awaits_close_on_the_owning_loop(self):
        channel = _FakeAioChannel()
        namespace, caught = self._close_channel_with(channel)

        assert channel.close_awaited
        assert channel.closed_on_loop is aio.ENGINE.event_loop
        assert namespace["CHANNEL"] is None
        assert namespace["CHANNEL_PORT"] is None
        assert namespace["SONATA_GRPC_SERVICE"] is None
        assert _never_awaited_warnings(caught) == []

    def test_close_channel_discards_coroutine_when_loop_is_gone(self):
        aio.terminate()
        channel = _FakeAioChannel()
        namespace, caught = self._close_channel_with(channel)

        assert not channel.close_awaited
        assert namespace["CHANNEL"] is None
        assert namespace["CHANNEL_PORT"] is None
        assert namespace["SONATA_GRPC_SERVICE"] is None
        assert _never_awaited_warnings(caught) == []


class TestGrpcChannelInitialization:
    @staticmethod
    def _load_initialize(namespace):
        return _load_module_function(_GRPC_CLIENT_PATH, "initialize", namespace)

    def test_rebuilds_channel_when_replacement_helper_uses_a_new_port(self):
        old_channel = _FakeAioChannel()
        created_channels = []

        async def run_test():
            loop = asyncio.get_running_loop()
            old_channel._loop = loop

            def create_channel(target):
                channel = _FakeAioChannel()
                channel._loop = loop
                channel.target = target
                created_channels.append(channel)
                return channel

            namespace = {
                "aio": SimpleNamespace(
                    asyncio_coroutine_to_concurrent_future=lambda func: func,
                    ENGINE=SimpleNamespace(event_loop=loop),
                ),
                "grpc": SimpleNamespace(
                    aio=SimpleNamespace(insecure_channel=create_channel)
                ),
                "sonata_grpcStub": lambda channel: ("stub", channel),
                "start_grpc_server": lambda: True,
                "CHANNEL": old_channel,
                "CHANNEL_PORT": 50051,
                "SONATA_GRPC_SERVICE": ("old-stub", old_channel),
                "SONATA_GRPC_SERVER_PORT": 50052,
                "log": SimpleNamespace(debug=lambda *args, **kwargs: None),
            }

            await self._load_initialize(namespace)()

            assert old_channel.close_awaited
            assert len(created_channels) == 1
            assert namespace["CHANNEL"] is created_channels[0]
            assert namespace["CHANNEL_PORT"] == 50052
            assert namespace["SONATA_GRPC_SERVICE"] == ("stub", created_channels[0])
            assert created_channels[0].target == "localhost:50052"

        asyncio.run(run_test())

    def test_reuses_channel_only_when_loop_and_port_both_match(self):
        channel = _FakeAioChannel()

        async def run_test():
            loop = asyncio.get_running_loop()
            channel._loop = loop
            namespace = {
                "aio": SimpleNamespace(
                    asyncio_coroutine_to_concurrent_future=lambda func: func,
                    ENGINE=SimpleNamespace(event_loop=loop),
                ),
                "grpc": SimpleNamespace(
                    aio=SimpleNamespace(
                        insecure_channel=lambda target: (_ for _ in ()).throw(
                            AssertionError("matching channel should be reused")
                        )
                    )
                ),
                "sonata_grpcStub": lambda value: value,
                "start_grpc_server": lambda: True,
                "CHANNEL": channel,
                "CHANNEL_PORT": 50051,
                "SONATA_GRPC_SERVICE": object(),
                "SONATA_GRPC_SERVER_PORT": 50051,
                "log": SimpleNamespace(debug=lambda *args, **kwargs: None),
            }

            await self._load_initialize(namespace)()

            assert namespace["CHANNEL"] is channel
            assert not channel.close_awaited

        asyncio.run(run_test())

    def test_failed_server_start_never_builds_a_channel_to_no_port(self):
        namespace = {
            "aio": SimpleNamespace(
                asyncio_coroutine_to_concurrent_future=lambda func: func,
            ),
            "start_grpc_server": lambda: False,
            "CHANNEL": None,
            "CHANNEL_PORT": None,
            "SONATA_GRPC_SERVICE": None,
            "SONATA_GRPC_SERVER_PORT": None,
        }

        with pytest.raises(RuntimeError, match="could not be started"):
            asyncio.run(self._load_initialize(namespace)())


class TestCrossModuleAttributesResolve:
    """__init__.py cannot be imported without NVDA, so check its references statically."""

    def test_grpc_client_attribute_references_are_defined(self):
        sources = dict(_package_sources())
        grpc_client_path = os.path.join(_PKG_DIR, "grpc_client", "__init__.py")
        defined = _top_level_names(sources[grpc_client_path])

        unresolved = []
        for path, tree in sources.items():
            if path == grpc_client_path:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "grpc_client"
                    and node.attr not in defined
                ):
                    unresolved.append(
                        f"{os.path.basename(path)}:{node.lineno} grpc_client.{node.attr}"
                    )

        assert unresolved == [], (
            f"{unresolved} reference names that grpc_client does not define at module "
            "level; these fail at runtime only, since NVDA-only modules are not importable."
        )


class _LogRecorder:
    def __init__(self):
        self.debug_messages = []
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []
        self.exception_messages = []

    def debug(self, message, *args, **kwargs):
        self.debug_messages.append(message)

    def info(self, message, *args, **kwargs):
        self.info_messages.append(message)

    def warning(self, message, *args, **kwargs):
        self.warning_messages.append(message)

    def error(self, message, *args, **kwargs):
        self.error_messages.append(message)

    def exception(self, message, *args, **kwargs):
        self.exception_messages.append(message)


class _FakeProcess:
    def __init__(self, pid, exe, *, parent=None, parent_error=None, survives_terminate=False):
        self.info = {"pid": pid, "name": os.path.basename(exe), "exe": exe}
        self._parent = parent
        self._parent_error = parent_error
        self.survives_terminate = survives_terminate
        self.terminated = False
        self.killed = False

    def parent(self):
        if self._parent_error is not None:
            raise self._parent_error
        return self._parent

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _FakePsutil:
    class AccessDenied(Exception):
        pass

    class NoSuchProcess(Exception):
        pass

    def __init__(self, processes, *, iteration_error=None, wait_error=None):
        self.processes = processes
        self.iteration_error = iteration_error
        self.wait_error = wait_error
        self.wait_calls = []

    def process_iter(self, attrs):
        assert attrs == ["pid", "name", "exe"]
        if self.iteration_error is not None:
            raise self.iteration_error
        return iter(self.processes)

    def wait_procs(self, processes, timeout):
        if self.wait_error is not None:
            raise self.wait_error
        processes = list(processes)
        self.wait_calls.append((processes, timeout))
        alive = [p for p in processes if p.survives_terminate and not p.killed]
        return [p for p in processes if p not in alive], alive


class TestGrpcServerProcessLifecycle:
    def _load_reaper(self, processes, **psutil_kwargs):
        fake_psutil = _FakePsutil(processes, **psutil_kwargs)
        log = _LogRecorder()
        namespace = {
            "os": SimpleNamespace(path=os.path, getpid=lambda: 100),
            "psutil": fake_psutil,
            "log": log,
            "PROCESS_EXIT_TIMEOUT": 3,
        }
        reaper = _load_module_function(
            _GRPC_CLIENT_PATH, "_reap_stale_grpc_servers", namespace
        )
        return reaper, fake_psutil, log

    def test_start_reuses_saved_live_helper_before_running_reaper(self):
        class LiveProcess:
            def poll(self):
                return None

        process = LiveProcess()
        saved_state = SimpleNamespace(
            SONATA_GRPC_SERVER_PORT=50051,
            GRPC_SERVER_PROCESS=process,
        )
        namespace = {
            "globalVars": saved_state,
            "GRPC_SERVER_PROCESS": None,
            "SONATA_GRPC_SERVER_PORT": None,
            "_reap_stale_grpc_servers": lambda path: (_ for _ in ()).throw(
                AssertionError("the live saved helper must be reused before reaping")
            ),
        }
        start_grpc_server = _load_module_function(
            _GRPC_CLIENT_PATH, "start_grpc_server", namespace
        )

        assert start_grpc_server()
        assert namespace["GRPC_SERVER_PROCESS"] is process
        assert namespace["SONATA_GRPC_SERVER_PORT"] == 50051

    @pytest.mark.parametrize("poll_result", [1, OSError("invalid process handle")])
    def test_start_recovers_from_dead_or_invalid_saved_helper(self, poll_result):
        class UnavailableProcess:
            def poll(self):
                if isinstance(poll_result, Exception):
                    raise poll_result
                return poll_result

        process = UnavailableProcess()
        saved_state = SimpleNamespace(
            SONATA_GRPC_SERVER_PORT=50051,
            GRPC_SERVER_PROCESS=process,
        )
        reaped_paths = []
        log = _LogRecorder()
        clear_state = _load_module_function(
            _GRPC_CLIENT_PATH,
            "_clear_saved_server_state",
            {"globalVars": saved_state},
        )
        namespace = {
            "os": os,
            "globalVars": saved_state,
            "GRPC_SERVER_PROCESS": None,
            "SONATA_GRPC_SERVER_PORT": None,
            "BIN_DIRECTORY": "addon-bin",
            "VC_REDIST_URL": "https://example.invalid/vc-redist",
            "_clear_saved_server_state": clear_state,
            "_reap_stale_grpc_servers": reaped_paths.append,
            "_vcruntime_missing": lambda: True,
            "_show_vcruntime_warning": lambda: None,
            "log": log,
        }
        start_grpc_server = _load_module_function(
            _GRPC_CLIENT_PATH, "start_grpc_server", namespace
        )

        assert not start_grpc_server()
        assert reaped_paths == [os.path.join("addon-bin", "sonata-grpc.exe")]
        assert not hasattr(saved_state, "SONATA_GRPC_SERVER_PORT")
        assert not hasattr(saved_state, "GRPC_SERVER_PROCESS")
        if isinstance(poll_result, Exception):
            assert log.debug_messages == [
                "Could not inspect the saved Dengjen GRPC helper"
            ]

    def test_reaper_only_stops_helpers_from_this_addon_copy(self, tmp_path):
        expected = str(tmp_path / "addon" / "sonata-grpc.exe")
        abandoned = _FakeProcess(1, expected)
        stale_from_this_nvda = _FakeProcess(
            2, expected, parent=SimpleNamespace(pid=100)
        )
        another_copy = _FakeProcess(4, str(tmp_path / "portable" / "sonata-grpc.exe"))
        active_in_another_nvda = _FakeProcess(
            3, expected, parent=SimpleNamespace(pid=200)
        )
        reaper, fake_psutil, log = self._load_reaper(
            [abandoned, stale_from_this_nvda, another_copy, active_in_another_nvda]
        )

        reaper(expected)

        assert abandoned.terminated
        assert stale_from_this_nvda.terminated
        assert not another_copy.terminated
        assert not active_in_another_nvda.terminated
        assert len(fake_psutil.wait_calls) == 1
        assert log.info_messages == ["Removed 2 abandoned Dengjen GRPC helper process(es)"]

    def test_reaper_reports_helper_that_does_not_exit_without_retrying(self, tmp_path):
        expected = str(tmp_path / "sonata-grpc.exe")
        stuck = _FakeProcess(1, expected, survives_terminate=True)
        reaper, fake_psutil, log = self._load_reaper([stuck])

        reaper(expected)

        assert stuck.terminated
        assert not stuck.killed
        assert len(fake_psutil.wait_calls) == 1
        assert log.warning_messages == [
            "Could not remove 1 abandoned Dengjen GRPC helper process(es)"
        ]

    def test_reaper_preserves_helper_when_parent_cannot_be_verified(self, tmp_path):
        expected = str(tmp_path / "sonata-grpc.exe")
        uncertain = _FakeProcess(
            1,
            expected,
            parent_error=_FakePsutil.AccessDenied(),
        )
        reaper, fake_psutil, log = self._load_reaper([uncertain])

        reaper(expected)

        assert not uncertain.terminated
        assert fake_psutil.wait_calls == []
        assert log.debug_messages == [
            "Could not inspect or terminate a Dengjen GRPC helper"
        ]

    def test_reaper_failure_never_escapes_or_blocks_synth_startup(self, tmp_path):
        expected = str(tmp_path / "sonata-grpc.exe")
        reaper, _, log = self._load_reaper(
            [], iteration_error=RuntimeError("process enumeration failed")
        )

        reaper(expected)

        assert log.exception_messages == [
            "Failed while checking for abandoned Dengjen GRPC helpers"
        ]

    def test_reaper_wait_failure_never_escapes_or_blocks_synth_startup(self, tmp_path):
        expected = str(tmp_path / "sonata-grpc.exe")
        abandoned = _FakeProcess(1, expected)
        reaper, _, log = self._load_reaper(
            [abandoned], wait_error=RuntimeError("wait failed")
        )

        reaper(expected)

        assert abandoned.terminated
        assert log.exception_messages == [
            "Failed while checking for abandoned Dengjen GRPC helpers"
        ]

    def test_clear_saved_state_removes_both_global_values(self):
        saved_state = SimpleNamespace(
            SONATA_GRPC_SERVER_PORT=12345,
            GRPC_SERVER_PROCESS=object(),
            unrelated="preserved",
        )
        namespace = {"globalVars": saved_state}
        clear_state = _load_module_function(
            _GRPC_CLIENT_PATH, "_clear_saved_server_state", namespace
        )

        clear_state()

        assert not hasattr(saved_state, "SONATA_GRPC_SERVER_PORT")
        assert not hasattr(saved_state, "GRPC_SERVER_PROCESS")
        assert saved_state.unrelated == "preserved"

    def test_terminate_timeout_always_clears_saved_state(self):
        class StuckProcess:
            terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout):
                raise subprocess.TimeoutExpired("sonata-grpc.exe", timeout)

        process = StuckProcess()
        saved_state = SimpleNamespace(
            SONATA_GRPC_SERVER_PORT=50051,
            GRPC_SERVER_PROCESS=process,
        )
        log = _LogRecorder()
        namespace = {
            "atexit": SimpleNamespace(register=lambda func: func),
            "subprocess": subprocess,
            "close_channel": lambda: None,
            "aio": SimpleNamespace(terminate=lambda: None),
            "log": log,
            "globalVars": saved_state,
            "GRPC_SERVER_PROCESS": process,
            "SONATA_GRPC_SERVER_PORT": 50051,
            "PROCESS_EXIT_TIMEOUT": 3,
            "_clear_saved_server_state": _load_module_function(
                _GRPC_CLIENT_PATH,
                "_clear_saved_server_state",
                {"globalVars": saved_state},
            ),
        }
        terminate = _load_module_function(_GRPC_CLIENT_PATH, "terminate", namespace)

        terminate()

        assert process.terminated
        assert namespace["GRPC_SERVER_PROCESS"] is None
        assert namespace["SONATA_GRPC_SERVER_PORT"] is None
        assert not hasattr(saved_state, "SONATA_GRPC_SERVER_PORT")
        assert not hasattr(saved_state, "GRPC_SERVER_PROCESS")
        assert log.warning_messages == [
            "Dengjen GRPC helper did not exit during shutdown"
        ]

    def test_terminate_phase_failures_do_not_block_process_cleanup(self):
        class Process:
            terminated = False
            waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout):
                self.waited = True

        process = Process()
        saved_state = SimpleNamespace(
            SONATA_GRPC_SERVER_PORT=50051,
            GRPC_SERVER_PROCESS=process,
        )
        log = _LogRecorder()

        def fail_channel_close():
            raise RuntimeError("channel close failed")

        def fail_aio_terminate():
            raise RuntimeError("async engine shutdown failed")

        namespace = {
            "atexit": SimpleNamespace(register=lambda func: func),
            "subprocess": subprocess,
            "close_channel": fail_channel_close,
            "aio": SimpleNamespace(terminate=fail_aio_terminate),
            "log": log,
            "globalVars": saved_state,
            "GRPC_SERVER_PROCESS": process,
            "SONATA_GRPC_SERVER_PORT": 50051,
            "PROCESS_EXIT_TIMEOUT": 3,
            "_clear_saved_server_state": _load_module_function(
                _GRPC_CLIENT_PATH,
                "_clear_saved_server_state",
                {"globalVars": saved_state},
            ),
        }
        terminate = _load_module_function(_GRPC_CLIENT_PATH, "terminate", namespace)

        terminate()

        assert process.terminated
        assert process.waited
        assert namespace["GRPC_SERVER_PROCESS"] is None
        assert namespace["SONATA_GRPC_SERVER_PORT"] is None
        assert not hasattr(saved_state, "SONATA_GRPC_SERVER_PORT")
        assert not hasattr(saved_state, "GRPC_SERVER_PROCESS")
        assert log.exception_messages == [
            "Failed to close the Dengjen GRPC channel during shutdown",
            "Failed to stop the Dengjen asynchronous engine during shutdown",
        ]

    def test_terminate_poll_failure_is_logged_and_saved_state_is_cleared(self):
        class InvalidProcess:
            def poll(self):
                raise OSError("invalid process handle")

        process = InvalidProcess()
        saved_state = SimpleNamespace(
            SONATA_GRPC_SERVER_PORT=50051,
            GRPC_SERVER_PROCESS=process,
        )
        log = _LogRecorder()
        namespace = {
            "atexit": SimpleNamespace(register=lambda func: func),
            "subprocess": subprocess,
            "close_channel": lambda: None,
            "aio": SimpleNamespace(terminate=lambda: None),
            "log": log,
            "globalVars": saved_state,
            "GRPC_SERVER_PROCESS": process,
            "SONATA_GRPC_SERVER_PORT": 50051,
            "PROCESS_EXIT_TIMEOUT": 3,
            "_clear_saved_server_state": _load_module_function(
                _GRPC_CLIENT_PATH,
                "_clear_saved_server_state",
                {"globalVars": saved_state},
            ),
        }
        terminate = _load_module_function(_GRPC_CLIENT_PATH, "terminate", namespace)

        terminate()

        assert namespace["GRPC_SERVER_PROCESS"] is None
        assert namespace["SONATA_GRPC_SERVER_PORT"] is None
        assert not hasattr(saved_state, "SONATA_GRPC_SERVER_PORT")
        assert not hasattr(saved_state, "GRPC_SERVER_PROCESS")
        assert log.exception_messages == [
            "Failed to stop the Dengjen GRPC helper during shutdown"
        ]

