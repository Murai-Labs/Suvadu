"""Progress-reporter tests (TASK P1.004).

The clock is injected so ETA arithmetic is checked deterministically rather than by sleeping.
"""

import io

import pytest

from suvadu.train.progress import ProgressReporter, _fmt_duration


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make(total=1000, log_every=100):
    clock = FakeClock()
    stream = io.StringIO()
    r = ProgressReporter(total_steps=total, log_every=log_every, stream=stream, clock=clock)
    r.start()
    return r, clock, stream


def test_rejects_log_interval_over_100():
    """AGENTS.md 4 caps the interval at 100 steps; a config bug should fail, not be clamped."""
    with pytest.raises(ValueError, match="AGENTS.md 4"):
        ProgressReporter(total_steps=10, log_every=500)


def test_rejects_nonpositive_total():
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ProgressReporter(total_steps=0)


def test_elapsed_before_start_is_an_error():
    r = ProgressReporter(total_steps=10)
    with pytest.raises(RuntimeError, match="start\\(\\) was never called"):
        r.elapsed()


def test_logs_first_step():
    """The most valuable line is the one proving the loop entered at all."""
    r, _, stream = make()
    assert r.log(1, loss=2.0) is True
    assert "step 1/1000" in stream.getvalue()


def test_logs_on_interval_and_not_between():
    r, _, _ = make()
    assert r.should_log(100)
    assert r.should_log(200)
    assert not r.should_log(150)


def test_always_logs_final_step():
    r, _, _ = make(total=1050)
    assert r.should_log(1050)


def test_eta_is_linear_in_throughput():
    r, clock, _ = make(total=100)
    clock.advance(10.0)          # 10 steps in 10s -> 1 step/s
    assert r.eta_seconds(10) == pytest.approx(90.0)


def test_eta_zero_at_completion():
    r, clock, _ = make(total=100)
    clock.advance(100.0)
    assert r.eta_seconds(100) == 0.0


def test_line_carries_every_required_field():
    """step/total, elapsed and ETA are contractual; loss/throughput/memory are the useful extras."""
    r, clock, _ = make(total=1000)
    clock.advance(60.0)
    line = r.format_line(100, loss=1.2345, tokens_per_s=4321.0, peak_mem_gib=78.9)
    assert "step 100/1000" in line
    assert "elapsed 1m00s" in line
    assert "eta " in line
    assert "loss 1.2345" in line
    assert "tok/s 4,321" in line
    assert "peak 78.9GiB" in line


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (45, "45s"), (60, "1m00s"), (3600, "1h00m"), (7325, "2h02m")],
)
def test_duration_formatting(seconds, expected):
    assert _fmt_duration(seconds) == expected
