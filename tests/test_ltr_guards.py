from types import SimpleNamespace

import numpy as np
import pytest
import typer

from wands import cli, ltr
from wands.qu.rewrite import RewriteRetriever, eligible
from wands.types import Query


def world(qrels):
    return SimpleNamespace(qrels=qrels)


DATA = {1: ([10, 11, 12], np.ones((3, len(ltr.FEATURES)))), 2: ([20, 21], np.ones((2, len(ltr.FEATURES))))}
QRELS = {1: {10: 2, 11: 0}, 2: {20: 1}}


def test_folds_cover_every_query_once_and_repeat():
    query_ids = list(range(100))
    folds = ltr.folds_of(query_ids)
    assert len(folds) == ltr.FOLDS
    assert sorted(q for fold in folds for q in fold) == query_ids
    assert folds == ltr.folds_of(list(reversed(query_ids)))


def test_training_set_drops_unjudged_by_default():
    x, y, groups = ltr.training_set(DATA, world(QRELS), [1, 2], unjudged_as_zero=False)
    assert groups == [2]  # query 2 has one judged row, too few to rank
    assert y.tolist() == [2, 0] and x.shape[0] == 2


def test_training_set_can_keep_unjudged_as_zero():
    _, y, groups = ltr.training_set(DATA, world(QRELS), [1, 2], unjudged_as_zero=True)
    assert groups == [3, 2]
    assert y.tolist() == [2, 0, 0, 1, 0]


def test_model_gains_match_the_metric():
    assert ltr.make_ranker().get_params()["label_gain"] == [0, 1, 2]


def test_ranker_learns_a_planted_signal():
    rng = np.random.default_rng(0)
    data, qrels = {}, {}
    signal = ltr.FEATURES.index("bm25_name")
    for qid in range(40):
        rows = rng.normal(size=(30, len(ltr.FEATURES)))
        gains = np.digitize(rows[:, signal], [-0.3, 0.6])
        ids = list(range(qid * 100, qid * 100 + 30))
        data[qid] = (ids, rows)
        qrels[qid] = {pid: int(g) for pid, g in zip(ids, gains, strict=True)}
    model = ltr.fit(data, world(qrels), list(range(40)), unjudged_as_zero=False)
    ids, rows = data[0]
    top = [pid for pid, _ in ltr.ordered(model, ids, rows)[:5]]
    assert sum(qrels[0][pid] for pid in top) >= 8


def test_test_split_is_refused_without_the_freeze_tag(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="some-other-tag\n", returncode=0)
    )
    with pytest.raises(typer.BadParameter):
        cli.require_freeze()


def test_test_split_is_allowed_at_the_freeze_tag(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=f"{cli.FREEZE_TAG}\n", returncode=0)
    )
    cli.require_freeze()


def test_test_report_runs_only_once(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "require_freeze", lambda: None)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "test_abc1234.md").write_text("done", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="already been run"):
        cli.test_report()


class Echo:
    name = "echo"

    def __init__(self):
        self.seen = []

    def search(self, q, k):
        self.seen.append(q.text)
        return []


def test_rewrite_changes_only_what_retrieval_sees():
    inner = Echo()
    retriever = RewriteRetriever(inner, {"ligth bulb": "light bulb"})
    retriever.search(Query(text="ligth bulb"), 5)
    retriever.search(Query(text="sofa"), 5)
    assert inner.seen == ["light bulb", "sofa"]


def test_rewrite_eligibility_comes_from_qu_alone():
    class Table:
        answers = {
            "sure": (SimpleNamespace(confidence="high", product_classes=["Beds"]), "OK"),
            "unsure": (SimpleNamespace(confidence="low", product_classes=["Beds"]), "OK"),
            "empty": (SimpleNamespace(confidence="high", product_classes=[]), "OK"),
            "failed": (None, "TIMEOUT"),
        }

        def lookup(self, text):
            return self.answers[text]

    assert [eligible(Table(), t) for t in Table.answers] == [False, True, True, True]
