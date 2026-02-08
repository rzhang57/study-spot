import threading
import time

from pynput import keyboard

DISENGAGEMENT_THRESHOLD = 20


class KeyboardService:
    def __init__(self):
        self._last_activity = time.time()
        self._lock = threading.Lock()
        self._listener = None

    def _on_press(self, key):
        with self._lock:
            self._last_activity = time.time()

    @property
    def is_disengaged(self):
        with self._lock:
            return time.time() - self._last_activity > DISENGAGEMENT_THRESHOLD

    def start(self):
        if self._listener is not None:
            return
        with self._lock:
            self._last_activity = time.time()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener is None:
            return
        self._listener.stop()
        self._listener.join(timeout=2)
        self._listener = None
