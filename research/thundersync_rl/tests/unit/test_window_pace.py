"""Drafting-window pacing (thundersync_rl.window_pace): CPU, fake clock."""

from types import SimpleNamespace

import pytest

from thundersync_rl.window_pace import WindowPacer, share_window_pacer


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def pacer(clock, **kw):
    return WindowPacer(clock=clock, sleep=clock.sleep, poll_s=0.01, **kw)


def test_no_fit_no_wait():
    c = Clock(10.0)
    p = pacer(c)
    st = p.hold(t_start=4.0, unfinished=lambda: 5)
    assert st["pace/wait_s"] == 0.0 and c.t == 10.0


def test_fit_from_cut_short_cohort():
    # ev3-style: 128 drafts at equal progress, cut at 885/1024 after 5.8 s.
    p = pacer(Clock(), margin=0.0)
    p.observe(5.8, [885] * 128, cap=1024, all_done=False)
    assert p.full_window_s == pytest.approx(5.8 * 1024 / 885)


def test_fit_second_wave():
    # A queued pool (0 tokens at the cut) needs a whole cap after the cut.
    p = pacer(Clock(), margin=0.0)
    p.observe(4.0, [800, 0], cap=1000, all_done=False)
    assert p.full_window_s == pytest.approx(4.0 + 1000 * 4.0 / 800)


def test_hold_waits_to_deadline():
    c = Clock(6.0)
    p = pacer(c, margin=0.0, max_wait_frac=0.3)
    p.full_window_s = 7.0
    st = p.hold(t_start=0.5, unfinished=lambda: 3)  # deadline 7.5, 1.5 s away
    assert st["pace/deadline"] == 1.0 and st["pace/released"] == 0.0
    assert c.t == pytest.approx(7.5) and st["pace/wait_s"] == pytest.approx(1.5)


def test_hold_released_when_all_finished():
    c = Clock(6.0)
    p = pacer(c, margin=0.0)
    p.full_window_s = 7.0
    left = iter([4, 2, 0])
    st = p.hold(t_start=0.5, unfinished=lambda: next(left))
    assert st["pace/released"] == 1.0 and c.t == pytest.approx(6.02)


def test_hold_skips_far_deadline():
    c = Clock(3.0)
    p = pacer(c, margin=0.0, max_wait_frac=0.3)
    p.full_window_s = 7.0
    st = p.hold(t_start=0.0, unfinished=lambda: 9)  # 4 s away > 2.1 s
    assert st["pace/skipped"] == 1.0 and c.t == 3.0


def test_hold_past_deadline_cuts_now():
    c = Clock(9.0)
    p = pacer(c)
    p.full_window_s = 7.0
    st = p.hold(t_start=0.0, unfinished=lambda: 9)
    assert st["pace/wait_s"] == 0.0 and c.t == 9.0


def test_estimate_up_at_once_down_slowly():
    p = pacer(Clock(), margin=0.0, decay=0.5)
    p.observe(6.0, [600], cap=1000, all_done=False)  # 10 s
    p.observe(4.0, [500], cap=1000, all_done=False)  # 8 s -> blend 9
    assert p.full_window_s == pytest.approx(9.0)
    p.observe(6.0, [500], cap=1000, all_done=False)  # 12 s -> at once
    assert p.full_window_s == pytest.approx(12.0)
    p.observe(13.0, [], cap=1000, all_done=True)  # upper bound above: no change
    assert p.full_window_s == pytest.approx(12.0)
    p.observe(10.0, [], cap=1000, all_done=True)  # finished by 10 s: blend
    assert p.full_window_s == pytest.approx(11.0)


def test_damps_alternation():
    """Toy loop: step = max(T_fast, s + window) when the cohort was fully
    drafted, else T_fast + penalty; window of the next cohort = this step - s.
    Without pacing it alternates; with pacing it settles at s + D."""
    s, D, T_fast, penalty = 0.8, 7.2, 6.6, 4.0

    def run(paced, steps=12):
        c = Clock()
        p = pacer(c, margin=0.02)
        full, out = True, []
        for _ in range(steps):
            t0 = c.t
            t_start = t0 + s
            c.t = t0 + (T_fast if full else T_fast + penalty)  # training ends
            if paced:
                done_at = t_start + D
                p.hold(t_start, lambda: 0 if c.t >= done_at else 1)
            w = c.t - t_start
            full = w >= D
            # all streams at equal progress: length = cap * min(1, w / D)
            p.observe(min(w, D) if full else w, [] if full else [int(1024 * w / D)], 1024, full)
            out.append(c.t - t0)
        return out

    plain = run(False)
    paced = run(True)
    assert max(plain[2:]) - min(plain[2:]) > 3.0  # alternation
    assert max(paced[3:]) - min(paced[3:]) < 0.05
    assert sum(paced[3:]) / len(paced[3:]) < sum(plain[3:]) / len(plain[3:])
    assert paced[-1] == pytest.approx(s + D, abs=0.05)


def test_lanes_share_one_pacer():
    a = SimpleNamespace(_pacer=WindowPacer())
    b = SimpleNamespace(_pacer=WindowPacer())
    off = SimpleNamespace(_pacer=None)
    share_window_pacer([a, b, off])
    assert a._pacer is b._pacer and off._pacer is None


def test_bad_args():
    with pytest.raises(ValueError):
        WindowPacer(decay=0.0)


def test_init_prior_paces_from_first_cut():
    c = Clock(6.0)
    p = pacer(c, margin=0.0, max_wait_frac=0.5, init_s=7.0)
    st = p.hold(t_start=0.0, unfinished=lambda: 0 if c.t >= 6.5 else 4)
    assert st["pace/released"] == 1.0 and c.t == pytest.approx(6.5, abs=0.02)
    with pytest.raises(ValueError):
        WindowPacer(init_s=0.0)
