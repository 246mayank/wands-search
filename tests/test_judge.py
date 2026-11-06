import json

import numpy as np
import pytest

from wands.judge import (
    JudgeCache,
    Pair,
    confusion,
    fill_ndcg,
    judge,
    sample_labelled,
    weighted_kappa,
)


def test_confusion_counts_rows_as_human():
    matrix = confusion([0, 2, 2], [0, 1, 2])
    assert matrix[2, 1] == 1 and matrix[2, 2] == 1 and matrix.sum() == 3


def test_kappa_is_one_on_perfect_agreement():
    assert weighted_kappa(np.diag([10, 20, 30])) == pytest.approx(1.0)


def test_kappa_is_zero_when_answers_are_independent():
    assert weighted_kappa(np.array([[10, 10, 10]] * 3)) == pytest.approx(0.0)


def test_kappa_by_hand():
    # n = 4; weights 0, .5, 1. Observed disagreement: one neighbour miss = 0.5.
    # Margins: rows (2, 0, 2), columns (1, 1, 2). Expected counts = outer / 4:
    #   row 0: .5 .5 1  row 2: .5 .5 1 -> weighted 0*.5 + .5*.5 + 1*1 + 1*.5 + .5*.5 + 0*1 = 2.0
    matrix = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 2]])
    assert weighted_kappa(matrix) == pytest.approx(1 - 0.5 / 2.0)


def test_kappa_punishes_far_misses_more():
    near = np.array([[8, 2, 0], [0, 10, 0], [0, 0, 10]])
    far = np.array([[8, 0, 2], [0, 10, 0], [0, 0, 10]])
    assert weighted_kappa(far) < weighted_kappa(near)


def test_sample_is_balanced_and_repeatable():
    qrels = {q: {p + q * 100: p % 3 for p in range(30)} for q in range(10)}
    first = sample_labelled(qrels, list(range(10)), per_label=20)
    assert first == sample_labelled(qrels, list(range(10)), per_label=20)
    assert [sum(g == gain for _, _, g in first) for gain in (0, 1, 2)] == [20, 20, 20]


def test_sample_only_uses_the_queries_given():
    qrels = {1: {1: 0, 2: 1, 3: 2}, 2: {4: 0, 5: 1, 6: 2}}
    assert {q for q, _, _ in sample_labelled(qrels, [1], per_label=1)} == {1}


class Scripted:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, prompt):
        return self.replies.pop(0)


@pytest.fixture
def prompts(tmp_path):
    (tmp_path / "judge_v1.md").write_text("Pairs:\n{{items}}\n", encoding="utf-8")
    return tmp_path


PAIRS = [Pair("q", 1, "a"), Pair("q", 2, "b")]


def test_judge_caches_labels_and_skips_known_pairs(prompts, tmp_path):
    reply = json.dumps({"results": [{"id": 0, "label": "Exact"}, {"id": 1, "label": "Partial"}]})
    cache = JudgeCache(tmp_path / "j.jsonl")
    stats = judge(Scripted([reply]), "v1", PAIRS, cache, prompts_dir=prompts)
    assert stats["Exact"] == 1 and stats["Partial"] == 1
    assert judge(Scripted([]), "v1", PAIRS, cache, prompts_dir=prompts) == {"asked": 0}


def test_judge_retries_a_bad_label_then_records_failure(prompts, tmp_path):
    first = json.dumps({"results": [{"id": 0, "label": "Exact"}, {"id": 1, "label": "Maybe"}]})
    second = json.dumps({"results": [{"id": 0, "label": "Maybe"}]})
    cache = JudgeCache(tmp_path / "j.jsonl")
    stats = judge(Scripted([first, second]), "v1", PAIRS, cache, prompts_dir=prompts)
    assert stats["failed"] == 1 and stats["retried"] == 1


def test_fill_ndcg_never_overrides_a_human_label():
    ranked = {1: [5, 6]}
    qrels = {1: {5: 0, 7: 2}}
    llm = {(1, 5): 2, (1, 6): 2}
    result = fill_ndcg(ranked, qrels, llm, [1])
    assert result["judged_before"] == 0.5 and result["judged_after"] == 1.0
    assert result["ndcg_before"] == 0.0
    # 6 adds a gain of 2 at rank 2; 5 stays at its human gain of 0.
    assert 0 < result["ndcg_after"] < 1
