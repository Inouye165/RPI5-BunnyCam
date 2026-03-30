"""GPIO buzzer driver for Raspberry Pi physical alarm.

Wraps gpiozero.Buzzer so the buzzer fires in sync with browser audio
alerts.  Degrades gracefully on non-Pi hardware (laptop backend) where
GPIO is unavailable — all public methods become silent no-ops.

Hardware wiring (2-wire active buzzer via NPN transistor):
  - Buzzer + → Pi Pin 2 (5 V)
  - Buzzer − → Transistor Collector
  - Transistor Emitter → Pi Pin 6 (GND)
  - 1 kΩ resistor from Transistor Base → Pi GPIO 17 (Pin 11)

Environment variable overrides:
  BUNNYCAM_BUZZER_GPIO    – GPIO pin number (default: 17)
  BUNNYCAM_BUZZER_ENABLED – set to "0" or "false" to disable (default: on)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_GPIO = 17


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


BUZZER_GPIO: int = _env_int("BUNNYCAM_BUZZER_GPIO", _DEFAULT_GPIO)
BUZZER_ENABLED: bool = _env_bool("BUNNYCAM_BUZZER_ENABLED", True)


class GpioBuzzer:
    """Safe wrapper around a GPIO-driven active buzzer.

    Falls back to a no-op stub when gpiozero is not available or the
    GPIO pin cannot be claimed (e.g. running on a laptop).
    """

    def __init__(self, gpio_pin: int | None = None, enabled: bool | None = None) -> None:
        self._gpio_pin = gpio_pin if gpio_pin is not None else BUZZER_GPIO
        self._enabled = enabled if enabled is not None else BUZZER_ENABLED
        self._buzzer: Any = None
        self._lock = threading.Lock()
        self._available = False

        if not self._enabled:
            logger.info("gpio_buzzer: disabled by configuration")
            return

        try:
            from gpiozero import Buzzer  # type: ignore
            self._buzzer = Buzzer(self._gpio_pin)
            self._available = True
            logger.info("gpio_buzzer: ready on GPIO %d", self._gpio_pin)
        except (ImportError, Exception) as exc:
            logger.info("gpio_buzzer: unavailable — %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def beep(self, on_time: float = 0.2, off_time: float = 0.1, count: int = 1) -> None:
        """Fire the buzzer with the given pattern in a background thread.

        Non-blocking.  Safe to call from request handlers.
        """
        if not self._available:
            return
        threading.Thread(
            target=self._beep_worker,
            args=(on_time, off_time, count),
            daemon=True,
            name="gpio-beep",
        ).start()

    def _beep_worker(self, on_time: float, off_time: float, count: int) -> None:
        with self._lock:
            try:
                for i in range(count):
                    self._buzzer.on()
                    time.sleep(on_time)
                    self._buzzer.off()
                    if i < count - 1:
                        time.sleep(off_time)
            except Exception as exc:
                logger.debug("gpio_buzzer: beep error — %s", exc)
            finally:
                try:
                    self._buzzer.off()
                except Exception:
                    pass

    def siren(self, cycles: int = 4, on_time: float = 0.3, off_time: float = 0.15) -> None:
        """Rapid pulsing siren pattern for urgent alerts (e.g. dog detected)."""
        self.beep(on_time=on_time, off_time=off_time, count=cycles)

    def quick_buzz(self) -> None:
        """Single short buzz for motion alerts."""
        self.beep(on_time=0.18, off_time=0.0, count=1)

    def off(self) -> None:
        """Ensure buzzer is off."""
        if not self._available:
            return
        try:
            self._buzzer.off()
        except Exception:
            pass

    def get_status(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "enabled": self._enabled,
            "gpio_pin": self._gpio_pin,
        }

    def cleanup(self) -> None:
        """Release the GPIO pin."""
        if self._buzzer is not None:
            try:
                self._buzzer.off()
                self._buzzer.close()
            except Exception:
                pass
            self._buzzer = None
            self._available = False


# Module-level singleton — created on import, safe on any platform.
_buzzer = GpioBuzzer()


def get_buzzer() -> GpioBuzzer:
    return _buzzer
