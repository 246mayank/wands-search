import numpy as np
import polars as pl
import pytest

from wands.eval.bootstrap import paired_bootstrap
from wands.eval.qrels import build_qrels
from wands.eval.report import compare_table, render_report, score_run
from wands.eval.runfile import read_run, write_run
from wands.eval.split import make_split

QUERIES = [(i, "big" if i < 300 else f"class{i % 90}") for i in range(480)]


def test_split_sizes_and_disjoint():
    split = make_split(QUERIES)
    assert len(split["dev"]) == 320 and len(split["test"]) == 160
    assert not set(split["dev"]) & set(split["test"])
    assert set(split["dev"]) | set(split["test"]) == {qid for qid, _ in QUERIES}


def test_split_same_seed_same_split():
    assert make_split(QUERIES) == make_split(list(reversed(QUERIES)))


def test_split_different_seed_differs():
    assert make_split(QUERIES, seed=1) != make_split(QUERIES, seed=2)


def test_split_stratifies_big_classes():
    split = make_split(QUERIES)
    in_dev = sum(qid < 300 for qid in split["dev"])
    assert abs(in_dev - 200) <= 1


def test_bootstrap_identical_runs_give_zero():
    a = np.array([0.1, 0.5, 0.9, 0.3])
    d = paired_bootstrap(a, a.copy())
    assert (d.delta, d.low, d.high) == (0.0, 0.0, 0.0) and not d.significant


def test_bootstrap_finds_a_real_shift():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.4, 0.6, 200)
    d = paired_bootstrap(a, a + 0.05)
    assert d.delta == pytest.approx(0.05) and d.significant


def test_bootstrap_noise_is_not_significant():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, 100)
    b = a + rng.normal(0, 0.2, 100) - 0.0001
    assert not paired_bootstrap(a, b).significant


def test_bootstrap_drops_queries_undefined_on_either_side():
    a = np.array([0.5, np.nan, 0.5])
    b = np.array([0.7, 0.9, np.nan])
    d = paired_bootstrap(a, b)
    assert d.n == 1 and d.delta == pytest.approx(0.2)


def test_bootstrap_rejects_misaligned():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4))


def test_qrels_keep_highest_gain_of_a_duplicate_pair():
    labels = pl.DataFrame(
        {"query_id": [1, 1, 1], "product_id": [7, 7, 8], "label": ["Partial", "Exact", "Irrelevant"]}
    )
    assert build_qrels(labels) == {1: {7: 2, 8: 0}}


def test_run_file_round_trip(tmp_path):
    path = tmp_path / "run.tsv"
    write_run({2: [(9, 1.5), (8, 0.5)], 1: [(3, 2.0)]}, path)
    assert read_run(path) == {1: [3], 2: [9, 8]}


def test_query_missing_from_run_scores_zero_not_skipped():
    scored = score_run({}, {1: {5: 2}}, [1], "empty", "dev")
    assert scored.mean("ndcg@10") == 0.0


def test_report_and_table_render():
    qrels = {1: {5: 2, 6: 1}, 2: {7: 1}}
    good = score_run({1: [5, 6], 2: [7]}, qrels, [1, 2], "good", "dev")
    bad = score_run({1: [6, 5], 2: [9]}, qrels, [1, 2], "bad", "dev")
    text = render_report(good, "abc1234", "configs/x.yaml", {1: "a|b", 2: "c"})
    assert "commit `abc1234`" in text and "a\\|b" in text
    table = compare_table([bad, good], bad, ["ndcg@10"])
    assert "`good`" in table and "base" in table
