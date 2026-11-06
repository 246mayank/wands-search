import math

import numpy as np
import pytest
from ranx import Qrels as RanxQrels
from ranx import Run as RanxRun
from ranx import evaluate as ranx_evaluate

from wands.eval.metrics import (
    METRICS,
    capped_recall_at,
    dcg,
    judged_at,
    mrr_at,
    ndcg_at,
    recall_at,
    score_query,
)

JUDGED = {1: 2, 2: 1, 3: 0, 4: 2, 5: 1}


def test_dcg_by_hand():
    # 2/log2(2) + 1/log2(3) + 0 = 2 + 0.63093
    assert dcg([2, 1, 0]) == pytest.approx(2.63093, abs=1e-5)


def test_ndcg_by_hand():
    # ranked gains 2,0,1 give 2 + 0 + 0.5 = 2.5; ideal 2,2,1 gives 2 + 1.26186 + 0.5 = 3.76186
    assert ndcg_at([1, 3, 2], JUDGED, k=3) == pytest.approx(2.5 / 3.76186, abs=1e-5)


def test_ideal_ranking_scores_one():
    assert ndcg_at([1, 4, 2, 5, 3], JUDGED) == pytest.approx(1.0)


def test_all_irrelevant_scores_zero():
    assert ndcg_at([3], JUDGED) == 0.0


def test_unjudged_counts_as_zero_gain():
    assert ndcg_at([99, 98], JUDGED) == 0.0


def test_reverse_order_scores_below_right_order():
    right = [1, 4, 2, 5, 3]
    assert ndcg_at(right[::-1], JUDGED, k=3) < ndcg_at(right, JUDGED, k=3)


def test_ideal_uses_all_labels_not_only_retrieved():
    # Retrieving one Partial and nothing else must not look perfect.
    assert ndcg_at([2], JUDGED) < 1.0


def test_exact_only_ignores_partial():
    assert ndcg_at([2, 5], JUDGED, exact_only=True) == 0.0
    assert ndcg_at([1, 4], JUDGED, exact_only=True) == pytest.approx(1.0)


def test_query_with_nothing_relevant_is_nan_not_zero():
    assert math.isnan(ndcg_at([1], {1: 0, 2: 0}))
    assert math.isnan(recall_at([1], {1: 0}))


def test_mrr_finds_first_exact_within_k():
    assert mrr_at([3, 2, 1], JUDGED) == pytest.approx(1 / 3)
    assert mrr_at([3, 2, 5], JUDGED) == 0.0
    assert mrr_at([1], JUDGED, k=1) == 1.0
    assert math.isnan(mrr_at([1], {1: 1}))


def test_recall_counts_exact_and_partial():
    assert recall_at([1, 2, 3], JUDGED) == pytest.approx(2 / 4)


def test_capped_recall_reaches_one_when_more_relevant_than_k():
    judged = {pid: 1 for pid in range(50)}
    assert recall_at(list(range(10)), judged, k=10) == pytest.approx(0.2)
    assert capped_recall_at(list(range(10)), judged, k=10) == pytest.approx(1.0)


def test_judged_share():
    assert judged_at([1, 99, 3, 98], JUDGED, k=4) == pytest.approx(0.5)
    assert math.isnan(judged_at([], JUDGED))


def test_ndcg_matches_ranx_on_fixed_runs():
    """ranx's `ndcg` is Jarvelin's linear gain, which is ours; `ndcg_burges` (2^rel - 1) is not."""
    rng = np.random.default_rng(7)
    qrels, run, ours = {}, {}, {}
    for qid in range(40):
        pids = rng.choice(500, size=60, replace=False)
        gains = rng.choice([0, 1, 2], size=60, p=[0.3, 0.5, 0.2])
        judged = {int(p): int(g) for p, g in zip(pids, gains, strict=True)}
        ranked = [int(p) for p in rng.permutation(500)[:40]]
        key = f"q{qid}"
        qrels[key] = {f"d{p}": g for p, g in judged.items()}
        run[key] = {f"d{p}": 40.0 - i for i, p in enumerate(ranked)}
        ours[key] = ndcg_at(ranked, judged, 10)
    ranx_qrels, ranx_run = RanxQrels(qrels), RanxRun(run)
    ranx_evaluate(ranx_qrels, ranx_run, "ndcg@10")
    theirs = ranx_run.scores["ndcg@10"]
    assert sum(score > 0 for score in theirs.values()) > 20
    for key, score in theirs.items():
        assert ours[key] == pytest.approx(score, abs=1e-9)


def test_mrr_and_recall_match_ranx():
    qrels = {"q": {"d1": 1, "d2": 1, "d3": 1}}
    run = {"q": {"d9": 3.0, "d2": 2.0, "d3": 1.0}}
    judged = {1: 2, 2: 2, 3: 2}
    assert mrr_at([9, 2, 3], judged) == pytest.approx(ranx_evaluate(RanxQrels(qrels), RanxRun(run), "mrr@10"))
    assert recall_at([9, 2, 3], judged, 100) == pytest.approx(
        ranx_evaluate(RanxQrels(qrels), RanxRun(run), "recall@100")
    )


def test_score_query_has_every_metric():
    assert list(score_query([1], JUDGED)) == METRICS
