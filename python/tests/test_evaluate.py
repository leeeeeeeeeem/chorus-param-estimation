import csv

import pytest

import evaluate


def truth_row(id, split="test", source="vocalset", **kw):
    p = dict(rate_hz=1.0, depth_ms=2.0, delay_ms=15.0, mix=0.5) | kw
    return {"id": id, "split": split, "source": source, **{k: str(v) for k, v in p.items()}}


def pred(row, **kw):
    return {"id": row["id"], **{k: row[k] for k in ("rate_hz", "depth_ms", "delay_ms", "mix")}} | kw


def test_perfect_predictions_have_zero_error():
    truth = [truth_row("a"), truth_row("b", source="vctk", split="heldout")]
    rows = evaluate.summarise([pred(r) for r in truth], truth)
    assert {(r["split"], r["group"]) for r in rows} == {
        ("test", "all"), ("test", "source vocalset"), ("heldout", "all"), ("heldout", "source vctk")}
    assert all(r[e] == 0 for r in rows for e in evaluate.ERRORS)


def test_errors_are_in_real_units():
    t = truth_row("a")
    e = evaluate.errors(pred(t, rate_hz=2.0, depth_ms=2.5, delay_ms=14.0, mix=0.6), t)
    assert e["rate_oct"] == pytest.approx(1.0)
    assert e["depth_ms"] == pytest.approx(0.5)
    assert e["delay_ms"] == pytest.approx(1.0)
    assert e["mix"] == pytest.approx(0.1)
    # scaled: rate 1 octave of log2(50) octaves, depth 0.5/5, delay 1/18, mix 0.1
    import numpy as np
    expected = np.mean([1 / np.log2(50), 0.1, 1 / 18, 0.1])
    assert e["scaled"] == pytest.approx(expected)


def test_predictions_are_clipped_to_the_ranges():
    t = truth_row("a", rate_hz=0.1, depth_ms=0.0, delay_ms=7.0, mix=0.0)
    e = evaluate.errors(pred(t, rate_hz=-3, depth_ms=-1, delay_ms=0, mix=-0.5), t)
    assert all(v == 0 for v in e.values())


def test_conditions_and_unknown_ids():
    truth = [truth_row("a", mix=0.05), truth_row("b", depth_ms=0.2), truth_row("c", rate_hz=0.2),
             truth_row("d", mix=0.95)]
    groups = {r["group"]: r["n"] for r in evaluate.summarise([pred(r) for r in truth], truth)}
    assert groups["mix < 0.1"] == 1 and groups["depth < 0.5 ms"] == 1 and groups["rate < 0.3 Hz"] == 1
    assert groups["mix > 0.9"] == 1
    with pytest.raises(ValueError, match="not in params.csv"):
        evaluate.summarise([pred(truth_row("zzz"))], truth)


def test_main_writes_a_summary(tmp_path, capsys):
    truth = [truth_row("a")]
    (tmp_path / "ds").mkdir()
    for path, rows in [(tmp_path / "ds" / "params.csv", truth), (tmp_path / "p.csv", [pred(truth[0])])]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    evaluate.main([str(tmp_path / "p.csv"), "--dataset", str(tmp_path / "ds")])
    assert (tmp_path / "p_summary.csv").exists() and "vocalset" in capsys.readouterr().out
