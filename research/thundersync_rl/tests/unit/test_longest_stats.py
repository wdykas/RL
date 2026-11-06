"""spec/longest_*: earliest rejection among the longest verified drafts."""

from types import SimpleNamespace

from thundersync_rl.speculative import SpeculativeGeneration


def _stats(pairs):
    obj = SimpleNamespace(stats={})
    SpeculativeGeneration._longest_stats(obj, pairs)
    return obj.stats


def test_top_decile_earliest_rejection():
    # 20 drafts: the top decile (2 longest, lengths 1000 and 900) decides.
    pairs = [(100 + i, 100 + i) for i in range(18)] + [(1000, 400), (900, 900)]
    s = _stats(pairs)
    assert s["spec/longest_len_cut"] == 900
    assert s["spec/longest_drafts"] == 2
    assert s["spec/longest_rejections"] == 1
    assert s["spec/longest_min_reject_pos"] == 400
    assert s["spec/longest_min_reject_frac"] == 0.4
    assert s["spec/longest_max_remaining"] == 600
    assert s["spec/longest_accepted_frac"] == 1300 / 1900


def test_no_rejection_and_empty():
    s = _stats([(10, 10), (20, 20)])
    assert s["spec/longest_rejections"] == 0
    assert s["spec/longest_min_reject_pos"] == -1
    assert s["spec/longest_min_reject_frac"] == 1.0
    assert _stats([]) == {}
