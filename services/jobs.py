"""Tiny in-process background job runner so a Streamlit rerun (or a double
click) can never start the same paid request twice."""
from __future__ import annotations

import threading
from typing import Callable


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._errors: dict[str, BaseException] = {}

    def start(self, key: str, fn: Callable[[], None]) -> bool:
        """Start `fn` in a thread. Returns False if `key` is already running."""
        with self._lock:
            t = self._threads.get(key)
            if t is not None and t.is_alive():
                return False
            self._errors.pop(key, None)

            def runner() -> None:
                try:
                    fn()
                except BaseException as exc:  # noqa: BLE001
                    with self._lock:
                        self._errors[key] = exc

            thread = threading.Thread(target=runner, name=f"job-{key}", daemon=True)
            self._threads[key] = thread
            thread.start()
            return True

    def is_running(self, key: str) -> bool:
        with self._lock:
            t = self._threads.get(key)
            return t is not None and t.is_alive()

    def pop_error(self, key: str) -> BaseException | None:
        with self._lock:
            return self._errors.pop(key, None)

    def wait(self, key: str, timeout: float | None = None) -> None:
        with self._lock:
            t = self._threads.get(key)
        if t is not None:
            t.join(timeout)


JOBS = JobManager()
