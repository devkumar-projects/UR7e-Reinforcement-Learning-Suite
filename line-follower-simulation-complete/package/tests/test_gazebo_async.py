from ur7e_line_follower.gazebo_async import AsyncGzPoseUpdater


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


class FakeProc:
    def __init__(self, rc=None):
        self.rc = rc
        self.killed = False
    def poll(self): return self.rc
    def kill(self): self.killed = True; self.rc = -9
    def terminate(self): self.kill()
    def wait(self, timeout=None): return self.rc


def test_update_is_non_blocking_and_single_flight():
    clock = Clock(); procs=[]
    def factory(*args, **kwargs):
        p=FakeProc(None); procs.append(p); return p
    u=AsyncGzPoseUpdater('w','dot',popen_factory=factory,clock=clock)
    assert u.update(1,2,3) is True
    assert u.update(1,2,3) is False
    assert len(procs) == 1


def test_completed_request_is_reaped_and_relaunched():
    clock=Clock(); procs=[]
    def factory(*args, **kwargs):
        p=FakeProc(None); procs.append(p); return p
    u=AsyncGzPoseUpdater('w','dot',min_period_s=.08,popen_factory=factory,clock=clock)
    assert u.update(1,2,3)
    procs[0].rc=0; clock.t=.10
    assert u.update(1,2,3)
    assert u.success_count == 1 and len(procs) == 2


def test_hung_request_is_killed_without_exception():
    clock=Clock(); procs=[]
    def factory(*args, **kwargs):
        p=FakeProc(None); procs.append(p); return p
    u=AsyncGzPoseUpdater('w','dot',max_process_age_s=.2,popen_factory=factory,clock=clock)
    assert u.update(1,2,3)
    clock.t=.25
    # Reap the hung process and launch a replacement in the same non-blocking call.
    assert u.update(1,2,3)
    assert procs[0].killed is True
    assert u.failure_count == 1


def test_launch_error_is_cosmetic():
    clock=Clock()
    def factory(*args, **kwargs): raise OSError('synthetic')
    u=AsyncGzPoseUpdater('w','dot',popen_factory=factory,clock=clock)
    assert u.update(1,2,3) is False
    assert u.failure_count == 1
    assert 'synthetic' in u.last_error
