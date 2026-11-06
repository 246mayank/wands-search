import subprocess

import pytest

from wands import readme
from wands.eval import provenance

TABLE = "| run | ndcg@10 | delta |\n|---|---|---|\n| `base` | 0.7000 | base |\n| `final` | 0.8000 | +0.1 |\n"


def test_tables_finds_each_table_in_a_report(tmp_path):
    path = tmp_path / "r.md"
    path.write_text(f"# t\n\n{TABLE}\nprose\n\n| a |\n|---|\n", encoding="utf-8")
    assert len(readme.tables(path)) == 2


def test_cell_reads_by_row_and_header():
    assert readme.cell(TABLE, "final", "ndcg@10") == "0.8000"
    with pytest.raises(KeyError):
        readme.cell(TABLE, "missing", "ndcg@10")


def test_render_fills_an_empty_block_and_is_repeatable(monkeypatch):
    monkeypatch.setattr(readme, "BLOCKS", {"headline": lambda: "filled"})
    text = "x\n<!-- headline:start -->\n<!-- headline:end -->\ny\n"
    once = readme.render(text)
    assert once == "x\n<!-- headline:start -->\nfilled\n<!-- headline:end -->\ny\n"
    assert readme.render(once) == once


def test_render_refuses_a_missing_block(monkeypatch):
    monkeypatch.setattr(readme, "BLOCKS", {"headline": lambda: "filled"})
    with pytest.raises(ValueError):
        readme.render("no markers here")


def test_regenerated_reports_do_not_make_a_commit_dirty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True
        )

    git("init", "-q")
    (tmp_path / "results").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "results" / "a.md").write_text("one", encoding="utf-8")
    (tmp_path / "configs" / "a.yaml").write_text("one", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "first")
    (tmp_path / "results" / "a.md").write_text("two", encoding="utf-8")
    assert not provenance.git_sha().endswith("-dirty")
    (tmp_path / "configs" / "a.yaml").write_text("two", encoding="utf-8")
    assert provenance.git_sha().endswith("-dirty")


def test_appending_answers_stays_clean_but_rewriting_one_does_not(tmp_path, monkeypatch):
    # The test run appends answers at the tagged commit, and its report must still name that commit.
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return done.stdout.strip()

    answers = tmp_path / "cache" / "qu_cache.jsonl"
    answers.parent.mkdir()
    answers.write_bytes(b"one\ntwo\n")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "first")
    sha = git("rev-parse", "--short=7", "HEAD")
    assert provenance.git_sha() == sha
    answers.write_bytes(b"one\ntwo\nthree\n")
    assert provenance.git_sha() == sha
    answers.write_bytes(b"one\nchanged\nthree\n")
    assert provenance.git_sha() == f"{sha}-dirty"
