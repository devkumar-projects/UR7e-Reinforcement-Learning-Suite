"""Non-blocking Gazebo pose visualisation helper.

The RL / ROS executor thread must never wait for ``gz service``.  This helper
keeps at most one short-lived subprocess in flight and reaps or kills it on a
later call.  A failed visual update is therefore cosmetic only: it cannot stall
camera callbacks, robot control or the environment watchdog.
"""
from __future__ import annotations

import subprocess
import time
from typing import Callable, Optional


class AsyncGzPoseUpdater:
    """Best-effort, non-blocking updater for one Gazebo entity pose."""

    def __init__(
        self,
        world_name: str,
        entity_name: str,
        *,
        min_period_s: float = 0.08,
        max_process_age_s: float = 0.45,
        service_timeout_ms: int = 120,
        popen_factory: Callable = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.world_name = str(world_name)
        self.entity_name = str(entity_name)
        self.min_period_s = max(0.02, float(min_period_s))
        self.max_process_age_s = max(0.10, float(max_process_age_s))
        self.service_timeout_ms = max(20, int(service_timeout_ms))
        self._popen = popen_factory
        self._clock = clock
        self._proc = None
        self._proc_started = 0.0
        self._last_launch = -1e9
        self.launch_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_error = ''

    @property
    def busy(self) -> bool:
        self._reap(self._clock())
        return self._proc is not None

    def _terminate(self, proc) -> None:
        try:
            proc.kill()
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=0.05)
        except Exception:
            pass

    def _reap(self, now: float) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            rc = proc.poll()
        except Exception as exc:
            self.failure_count += 1
            self.last_error = f'poll failed: {exc}'
            self._terminate(proc)
            self._proc = None
            return
        if rc is None:
            if now - self._proc_started > self.max_process_age_s:
                self.failure_count += 1
                self.last_error = (
                    f'gz service exceeded {self.max_process_age_s:.2f}s and was killed'
                )
                self._terminate(proc)
                self._proc = None
            return
        if rc == 0:
            self.success_count += 1
            self.last_error = ''
        else:
            self.failure_count += 1
            self.last_error = f'gz service return code {rc}'
        self._proc = None

    def update(self, x: float, y: float, z: float) -> bool:
        """Schedule a pose update and return immediately.

        Returns ``True`` only when a new subprocess was launched.  ``False``
        means throttled, one request is already in flight, or launching failed.
        No subprocess is ever waited on in this method.
        """
        now = self._clock()
        self._reap(now)
        if self._proc is not None:
            return False
        if now - self._last_launch < self.min_period_s:
            return False

        req = (
            f'name: "{self.entity_name}" '
            f'position {{ x: {float(x):.4f} y: {float(y):.4f} z: {float(z):.4f} }} '
            'orientation { x: 0.0 y: 0.0 z: 0.0 w: 1.0 }'
        )
        args = [
            'gz', 'service', '-s', f'/world/{self.world_name}/set_pose',
            '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
            '--timeout', str(self.service_timeout_ms), '--req', req,
        ]
        try:
            proc = self._popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except (OSError, ValueError) as exc:
            self.failure_count += 1
            self.last_error = f'launch failed: {exc}'
            self._last_launch = now
            return False

        self._proc = proc
        self._proc_started = now
        self._last_launch = now
        self.launch_count += 1
        return True

    def close(self) -> None:
        proc = self._proc
        if proc is not None:
            self._terminate(proc)
            self._proc = None
