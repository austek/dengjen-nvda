# coding: utf-8
"""
Tests for SynthDriver._set_voice's failure handling
(addon/synthDrivers/sonata_neural_voices/__init__.py).

Regression coverage for issue #69: a voice that exists on disk but fails to
load (e.g. a corrupted .onnx file) must not leave the driver in a
half-switched state -- the previously active voice should remain current,
and NVDA should report a message instead of letting an unhandled exception
surface as an error chime.
"""

import os

import pytest
import ui
from logHandler import log

from tests.conftest import SYNTH_PKG_DIR, load_module_from_path

_driver_module = load_module_from_path(
    "sonata_neural_voices._init_under_test",
    os.path.join(SYNTH_PKG_DIR, "__init__.py"),
    package="sonata_neural_voices",
)
SynthDriver = _driver_module.SynthDriver


class _FakeVoiceEntry:
    """Stand-in for a _standard_voice_map value (a SonataVoice)."""

    def __init__(self, key, variant="unknown"):
        self.key = key
        self.variant = variant


class _FakeVoiceInfo:
    """Stand-in for synthDriverHandler.VoiceInfo -- only .displayName is used."""

    def __init__(self, display_name):
        self.displayName = display_name


class _FakeTTSRaising:
    """Stand-in for SonataTextToSpeechSystem whose voice setter always fails,
    as happens when the underlying .onnx model is corrupted/incomplete."""

    voice = None

    def __setattr__(self, name, value):
        if name == "voice":
            raise RuntimeError("Protobuf parsing failed")
        super().__setattr__(name, value)


class _FakeTTSAccepting:
    """Stand-in for SonataTextToSpeechSystem whose voice setter succeeds."""

    def __init__(self):
        self.voice = None


def _make_driver(voice_map, available_voices, tts, initial_voice=None):
    driver = SynthDriver.__new__(SynthDriver)
    driver._standard_voice_map = voice_map
    driver.availableVoices = available_voices
    driver._voice_map = {}
    driver.noise_scale = 50
    driver.length_scale = 50
    driver.noise_w = 50
    driver._SynthDriver__voice = initial_voice
    driver.tts = tts
    return driver


@pytest.fixture(autouse=True)
def _reset_mocks():
    ui.message.reset_mock()
    log.exception.reset_mock()
    yield


class TestSetVoiceFailure:
    def test_failed_load_keeps_the_previous_voice(self):
        driver = _make_driver(
            voice_map={
                "alex": _FakeVoiceEntry("en_US-alex-medium"),
                "bryce": _FakeVoiceEntry("en_US-bryce-medium"),
            },
            available_voices={
                "alex": _FakeVoiceInfo("Alex (en-US)"),
                "bryce": _FakeVoiceInfo("Bryce (en-US)"),
            },
            tts=_FakeTTSRaising(),
            initial_voice="alex",
        )

        driver._set_voice("bryce")

        # No half-switched state: the driver still reports the last voice
        # that actually loaded, not the one that failed.
        assert driver._SynthDriver__voice == "alex"

    def test_failed_load_reports_a_message_naming_the_voice(self):
        driver = _make_driver(
            voice_map={"bryce": _FakeVoiceEntry("en_US-bryce-medium")},
            available_voices={"bryce": _FakeVoiceInfo("Bryce (en-US)")},
            tts=_FakeTTSRaising(),
            initial_voice=None,
        )

        driver._set_voice("bryce")

        ui.message.assert_called_once()
        (message,), _kwargs = ui.message.call_args
        assert "Bryce (en-US)" in message

    def test_failed_load_does_not_raise(self):
        driver = _make_driver(
            voice_map={"bryce": _FakeVoiceEntry("en_US-bryce-medium")},
            available_voices={"bryce": _FakeVoiceInfo("Bryce (en-US)")},
            tts=_FakeTTSRaising(),
            initial_voice=None,
        )

        driver._set_voice("bryce")  # must not raise

    def test_failed_load_logs_the_exception(self):
        driver = _make_driver(
            voice_map={"bryce": _FakeVoiceEntry("en_US-bryce-medium")},
            available_voices={"bryce": _FakeVoiceInfo("Bryce (en-US)")},
            tts=_FakeTTSRaising(),
            initial_voice=None,
        )

        driver._set_voice("bryce")

        log.exception.assert_called_once()


class TestSetVoiceSuccess:
    def test_successful_load_switches_the_current_voice(self):
        driver = _make_driver(
            voice_map={
                "alex": _FakeVoiceEntry("en_US-alex-medium"),
                "danny": _FakeVoiceEntry("en_US-danny-low"),
            },
            available_voices={
                "alex": _FakeVoiceInfo("Alex (en-US)"),
                "danny": _FakeVoiceInfo("Danny (en-US)"),
            },
            tts=_FakeTTSAccepting(),
            initial_voice="alex",
        )

        driver._set_voice("danny")

        assert driver._SynthDriver__voice == "danny"
        assert driver.tts.voice == "en_US-danny-low"
        ui.message.assert_not_called()

    def test_falls_back_to_the_first_available_voice_when_value_unknown(self):
        driver = _make_driver(
            voice_map={"alex": _FakeVoiceEntry("en_US-alex-medium")},
            available_voices={"alex": _FakeVoiceInfo("Alex (en-US)")},
            tts=_FakeTTSAccepting(),
            initial_voice=None,
        )

        driver._set_voice("does-not-exist")

        assert driver._SynthDriver__voice == "alex"
        assert driver.tts.voice == "en_US-alex-medium"
