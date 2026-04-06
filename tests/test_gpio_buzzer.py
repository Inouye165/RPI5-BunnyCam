"""Regression tests for the GPIO buzzer module and alarm API routes."""

# pylint: disable=protected-access

from __future__ import annotations

import threading
import time

import pytest

from gpio_buzzer import GpioBuzzer


# ---------------------------------------------------------------------------
# GpioBuzzer unit tests (no real GPIO — always runs in disabled/unavail mode)
# ---------------------------------------------------------------------------

class TestGpioBuzzerNoHardware:
    """Tests that run on any machine — gpiozero is not available."""

    def test_buzzer_not_available_without_gpiozero(self):
        buzzer = GpioBuzzer(gpio_pin=17, enabled=True)
        # On machines without gpiozero, it should gracefully degrade.
        # available may be True on a real Pi, False on laptop/CI.
        # We just verify it doesn't crash.
        assert isinstance(buzzer.available, bool)

    def test_buzzer_disabled_by_config(self):
        buzzer = GpioBuzzer(gpio_pin=17, enabled=False)
        assert buzzer.available is False

    def test_beep_is_noop_when_unavailable(self):
        buzzer = GpioBuzzer(gpio_pin=17, enabled=False)
        # Should not raise
        buzzer.beep(on_time=0.01, off_time=0.01, count=1)
        buzzer.quick_buzz()
        buzzer.siren()
        buzzer.off()

    def test_get_status_returns_expected_fields(self):
        buzzer = GpioBuzzer(gpio_pin=17, enabled=False)
        status = buzzer.get_status()
        assert status["gpio_pin"] == 17
        assert status["enabled"] is False
        assert status["available"] is False

    def test_cleanup_is_safe_when_disabled(self):
        buzzer = GpioBuzzer(gpio_pin=17, enabled=False)
        buzzer.cleanup()
        assert buzzer.available is False

    def test_default_singleton_does_not_crash(self):
        from gpio_buzzer import get_buzzer
        buzzer = get_buzzer()
        assert isinstance(buzzer.available, bool)
        status = buzzer.get_status()
        assert "gpio_pin" in status


# ---------------------------------------------------------------------------
# Mock-based tests (simulates gpiozero being available)
# ---------------------------------------------------------------------------

class FakeBuzzerPin:
    """Minimal mock that records on/off calls."""
    def __init__(self):
        self.calls: list[str] = []
        self._active = False

    def on(self):
        self.calls.append("on")
        self._active = True

    def off(self):
        self.calls.append("off")
        self._active = False

    @property
    def is_active(self):
        return self._active

    def close(self):
        self.calls.append("close")


class TestGpioBuzzerWithMock:
    """Tests using a mock buzzer object to verify beep patterns."""

    def _make_buzzer_with_mock(self) -> tuple[GpioBuzzer, FakeBuzzerPin]:
        buzzer = GpioBuzzer(gpio_pin=17, enabled=False)  # start disabled
        mock = FakeBuzzerPin()
        buzzer._buzzer = mock
        buzzer._available = True
        buzzer._enabled = True
        return buzzer, mock

    def test_quick_buzz_fires_on_then_off(self):
        buzzer, mock = self._make_buzzer_with_mock()
        buzzer._beep_worker(0.01, 0.0, 1)
        assert "on" in mock.calls
        assert "off" in mock.calls

    def test_siren_fires_multiple_cycles(self):
        buzzer, mock = self._make_buzzer_with_mock()
        buzzer._beep_worker(0.01, 0.01, 3)
        on_count = mock.calls.count("on")
        assert on_count == 3

    def test_beep_is_non_blocking(self):
        buzzer, mock = self._make_buzzer_with_mock()
        start = time.monotonic()
        buzzer.beep(on_time=0.5, off_time=0.1, count=1)
        elapsed = time.monotonic() - start
        # beep() should return immediately (runs in background thread)
        assert elapsed < 0.2

    def test_cleanup_calls_close(self):
        buzzer, mock = self._make_buzzer_with_mock()
        buzzer.cleanup()
        assert "close" in mock.calls
        assert buzzer.available is False
