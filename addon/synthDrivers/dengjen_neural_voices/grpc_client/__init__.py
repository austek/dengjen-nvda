# coding: utf-8

import asyncio
import atexit
import ctypes
import os
import subprocess
import time
from pathlib import Path

import globalVars
from logHandler import log

VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def _vcruntime_missing():
    """Return True if vcruntime140_1.dll cannot be loaded.

    sonata-grpc.exe is built with MSVC and needs the Visual C++ 2015-2022
    Redistributable (x64). On fresh Windows installs without it, Popen
    succeeds but the child process exits immediately with a missing-DLL
    dialog the user never sees from inside NVDA; the addon then logs the
    misleading 'Connection refused' from the failing gRPC channel.

    Use ctypes.WinDLL to ask Windows directly — it respects the standard
    DLL search path, so this is more reliable than checking a fixed
    System32 location.
    """
    try:
        ctypes.WinDLL("vcruntime140_1.dll")
        return False
    except (OSError, AttributeError):
        return True


def _show_vcruntime_warning():
    """Defer a user-facing wx messageBox about the missing VC++ redistributable.

    Imports of wx and gui are local so this module stays importable from
    contexts (tests, headless tooling) where the NVDA GUI isn't available.
    """
    try:
        import wx
        import gui
        wx.CallAfter(
            gui.messageBox,
            (
                "Dengjen Neural Voices could not start because the "
                "Microsoft Visual C++ 2015-2022 Redistributable (x64) "
                f"is not installed.\n\nDownload and install it from:\n{VC_REDIST_URL}\n\n"
                "Then restart NVDA."
            ),
            "Dengjen: missing dependency",
            style=wx.ICON_ERROR,
            parent=gui.mainFrame,
        )
    except Exception:
        log.exception("Failed to show VC++ redistributable warning dialog", exc_info=True)

from ..const import DENGJEN_VOICES_BASE_DIR
from ..helpers import BIN_DIRECTORY, find_free_port, import_bundled_library


with import_bundled_library():
    import grpc
    import psutil
    from .. import aio
    from .grpc_protos.sonata_grpc_pb2_grpc import sonata_grpcStub
    from .grpc_protos import sonata_grpc_pb2 as msgs


SONATA_GRPC_SERVER_PORT = None
GRPC_SERVER_PROCESS = None
CHANNEL = None
CHANNEL_PORT = None
SONATA_GRPC_SERVICE = None
SERVER_CHECK_TIMEOUT = 15
# Outer bound on the startup futures; must exceed SERVER_CHECK_TIMEOUT so the
# coroutine's own error surfaces instead of a bare future timeout.
STARTUP_TIMEOUT = SERVER_CHECK_TIMEOUT + 5
CALL_TIMEOUT = 10
CHANNEL_CLOSE_TIMEOUT = 5
PROCESS_EXIT_TIMEOUT = 3


def _clear_saved_server_state():
    for name in ("SONATA_GRPC_SERVER_PORT", "GRPC_SERVER_PROCESS"):
        if hasattr(globalVars, name):
            delattr(globalVars, name)


def _reap_stale_grpc_servers(grpc_server_exe):
    """Stop helpers whose parent no longer owns them, including failed local shutdowns."""
    try:
        expected_exe = os.path.normcase(os.path.realpath(grpc_server_exe))
        expected_name = os.path.normcase(os.path.basename(expected_exe))
        stale_processes = []
        for proc in psutil.process_iter(attrs=["pid", "name", "exe"]):
            try:
                process_name = proc.info.get("name") or ""
                if os.path.normcase(process_name) != expected_name:
                    continue
                process_exe = proc.info.get("exe")
                if not process_exe:
                    continue
                if os.path.normcase(os.path.realpath(process_exe)) != expected_exe:
                    continue
                # DETACHED_PROCESS does not sever the Windows parent relationship.
                # start_grpc_server() has already reused this NVDA session's saved
                # live helper before reaching the reaper. A same-path helper whose
                # parent is this NVDA is therefore stale after a failed local
                # shutdown; one owned by another live process must be preserved.
                parent = proc.parent()
                if parent is not None and parent.pid != os.getpid():
                    continue
                proc.terminate()
                stale_processes.append(proc)
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                log.debug("Could not inspect or terminate a Dengjen GRPC helper", exc_info=True)
        if not stale_processes:
            return
        _, alive = psutil.wait_procs(stale_processes, timeout=PROCESS_EXIT_TIMEOUT)
        removed_count = len(stale_processes) - len(alive)
        if removed_count:
            log.info(f"Removed {removed_count} abandoned Dengjen GRPC helper process(es)")
        if alive:
            # On Windows, terminate() and kill() both call TerminateProcess. A
            # second attempt cannot improve the result, so report and continue.
            log.warning(
                f"Could not remove {len(alive)} abandoned Dengjen GRPC helper process(es)"
            )
    except Exception:
        # Reaping is recovery for a previous failed shutdown. It must never make
        # the synthesizer unavailable in an otherwise healthy NVDA session.
        log.exception("Failed while checking for abandoned Dengjen GRPC helpers")


def start_grpc_server():
    global GRPC_SERVER_PROCESS, SONATA_GRPC_SERVER_PORT
    if hasattr(globalVars, "SONATA_GRPC_SERVER_PORT"):
        saved_process = getattr(globalVars, "GRPC_SERVER_PROCESS", None)
        try:
            saved_process_is_alive = (
                saved_process is not None and saved_process.poll() is None
            )
        except OSError:
            saved_process_is_alive = False
            log.debug("Could not inspect the saved Dengjen GRPC helper", exc_info=True)
        if saved_process_is_alive:
            SONATA_GRPC_SERVER_PORT = globalVars.SONATA_GRPC_SERVER_PORT
            GRPC_SERVER_PROCESS = saved_process
            return True
        _clear_saved_server_state()
    grpc_server_exe = os.path.join(BIN_DIRECTORY, "sonata-grpc.exe")
    _reap_stale_grpc_servers(grpc_server_exe)
    if _vcruntime_missing():
        log.error(
            "Dengjen GRPC server cannot start: vcruntime140_1.dll not found. "
            "The Microsoft Visual C++ 2015-2022 Redistributable (x64) is required. "
            f"Download and install it from {VC_REDIST_URL} then restart NVDA."
        )
        _show_vcruntime_warning()
        return False
    SONATA_GRPC_SERVER_PORT = find_free_port()
    nvda_espeak_dir = os.path.join(globalVars.appDir, "synthDrivers")
    env = os.environ.copy()
    env.update({
        "SONATA_GRPC_SERVER_PORT": str(SONATA_GRPC_SERVER_PORT),
        "SONATA_ESPEAKNG_DATA_DIRECTORY": os.fspath(nvda_espeak_dir),
        "SONATA_GRPC": "info",
    })
    creationflags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.REALTIME_PRIORITY_CLASS
    )
    try:
        server_log_file = os.path.join(DENGJEN_VOICES_BASE_DIR, "logs", "sonata-grpc.log")
        Path(server_log_file).parent.mkdir(parents=True, exist_ok=True)
        server_stdout = open(server_log_file, "wb")
    except OSError:
        log.exception("Failed to open server log file for writing", exc_info=True)
        server_stdout = subprocess.DEVNULL
    try:
        GRPC_SERVER_PROCESS = subprocess.Popen(
            args=grpc_server_exe,
            cwd=os.fspath(BIN_DIRECTORY),
            env=env,
            creationflags=creationflags,
            stdout=server_stdout,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log.exception(
            "Failed to start Dengjen GRPC server. The synth will not be available.",
            exc_info=True
        )
        return False
    globalVars.SONATA_GRPC_SERVER_PORT = SONATA_GRPC_SERVER_PORT
    globalVars.GRPC_SERVER_PROCESS = GRPC_SERVER_PROCESS
    return True


@aio.asyncio_coroutine_to_concurrent_future
async def initialize():
    global CHANNEL, CHANNEL_PORT, SONATA_GRPC_SERVICE, SONATA_GRPC_SERVER_PORT
    if not start_grpc_server():
        raise RuntimeError("Dengjen GRPC server could not be started")
    port = SONATA_GRPC_SERVER_PORT
    if CHANNEL is not None:
        try:
            # grpc.aio binds a channel to the loop that created it, so a channel
            # outliving its loop has to be replaced rather than reused. A new
            # helper also receives a new port, which invalidates the old channel.
            channel_loop = getattr(CHANNEL, "_loop", None)
            if (
                channel_loop is aio.ENGINE.event_loop
                and aio.ENGINE.event_loop.is_running()
                and CHANNEL_PORT == port
            ):
                return
        except Exception:
            log.debug("Failed to inspect the existing GRPC channel", exc_info=True)
        try:
            await CHANNEL.close()
        except Exception:
            log.debug("Failed to close the stale GRPC channel", exc_info=True)
        CHANNEL = None
        CHANNEL_PORT = None
        SONATA_GRPC_SERVICE = None
    CHANNEL = grpc.aio.insecure_channel(f"localhost:{port}")
    CHANNEL_PORT = port
    SONATA_GRPC_SERVICE = sonata_grpcStub(CHANNEL)


def close_channel():
    """Close the aio channel on the loop that owns it.

    Channel.close() is a coroutine whose internals walk the running loop's
    task set, so it cannot be driven from another thread or a stopped loop.
    """
    global CHANNEL, CHANNEL_PORT, SONATA_GRPC_SERVICE
    if CHANNEL is None:
        CHANNEL_PORT = None
        SONATA_GRPC_SERVICE = None
        return
    channel, CHANNEL = CHANNEL, None
    CHANNEL_PORT = None
    SONATA_GRPC_SERVICE = None
    loop = aio.ENGINE.event_loop
    if loop is None or not loop.is_running():
        log.debug("Discarding the GRPC channel: its event loop is gone")
        channel.close().close()
        return
    try:
        asyncio.run_coroutine_threadsafe(channel.close(), loop).result(
            timeout=CHANNEL_CLOSE_TIMEOUT
        )
    except Exception:
        log.debug("Failed to close the GRPC channel cleanly", exc_info=True)


@atexit.register
def terminate():
    global GRPC_SERVER_PROCESS, SONATA_GRPC_SERVER_PORT
    SONATA_GRPC_SERVER_PORT = None
    try:
        close_channel()
    except Exception:
        log.exception("Failed to close the Dengjen GRPC channel during shutdown")
    try:
        aio.terminate()
    except Exception:
        log.exception("Failed to stop the Dengjen asynchronous engine during shutdown")
    process, GRPC_SERVER_PROCESS = GRPC_SERVER_PROCESS, None
    try:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=PROCESS_EXIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            # On Windows, Popen.kill() is the same TerminateProcess call as
            # terminate(), so retrying would only add another timeout delay.
            log.warning("Dengjen GRPC helper did not exit during shutdown")
        except OSError:
            log.debug("Dengjen GRPC helper was already unavailable during shutdown", exc_info=True)
    except Exception:
        log.exception("Failed to stop the Dengjen GRPC helper during shutdown")
    finally:
        _clear_saved_server_state()


@aio.asyncio_coroutine_to_concurrent_future
async def check_grpc_server() -> str:
    async with asyncio.timeout(SERVER_CHECK_TIMEOUT):
        return await get_sonata_version()


async def get_sonata_version():
    resp = await SONATA_GRPC_SERVICE.GetSonataVersion(msgs.Empty())
    return resp.version


@aio.asyncio_coroutine_to_concurrent_future
async def load_voice(config_path):
    req = msgs.VoicePath(config_path=config_path)
    return await SONATA_GRPC_SERVICE.LoadVoice(req)


@aio.asyncio_coroutine_to_concurrent_future
async def get_synth_options(voice_id):
    req = msgs.VoiceIdentifier(voice_id=voice_id)
    return await SONATA_GRPC_SERVICE.GetSynthesisOptions(req)


@aio.asyncio_coroutine_to_concurrent_future
async def set_synth_options(
    voice_id, speaker=None, length_scale=None, noise_scale=None, noise_w=None
):
    req = msgs.VoiceSynthesisOptions(
        voice_id=voice_id,
        synthesis_options=msgs.SynthesisOptions(
            speaker=speaker,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
        ),
    )
    return await SONATA_GRPC_SERVICE.SetSynthesisOptions(req)


async def speak(
    voice_id, text, rate=None, volume=None, pitch=None, appended_silence_ms=None, streaming=False
):
    speech_args = None
    if any([rate, volume, pitch, appended_silence_ms]):
        speech_args = msgs.SpeechArgs(
            rate=rate,
            volume=volume,
            pitch=pitch,
            appended_silence_ms=appended_silence_ms,
        )
    utterance = msgs.Utterance(
        voice_id=voice_id,
        text=text,
        speech_args=speech_args,
    )
    if streaming:
        stream = SONATA_GRPC_SERVICE.SynthesizeUtteranceRealtime
    else:
        stream = SONATA_GRPC_SERVICE.SynthesizeUtterance
    async for ret in stream(utterance):
        yield ret


async def bench(n=10000):
    initialize()
    t0 = time.perf_counter()
    for _ in range(n):
        await get_sonata_version()
    return time.perf_counter() - t0
