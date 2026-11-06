"""`wands` command line."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Fetch and inspect the WANDS dataset.")
app.add_typer(data_app, name="data")
qu_app = typer.Typer(no_args_is_help=True, help="Offline LLM query understanding.")
app.add_typer(qu_app, name="qu")
ltr_app = typer.Typer(no_args_is_help=True, help="Learning to rank on dev labels.")
app.add_typer(ltr_app, name="ltr")
judge_app = typer.Typer(no_args_is_help=True, help="LLM relevance judge, calibrated against human labels.")
app.add_typer(judge_app, name="judge")

FREEZE_TAG = "freeze-v1"
RUNS_DIR = Path("runs")


@data_app.command("fetch")
def data_fetch() -> None:
    """Download WANDS at the pinned commit and verify its checksums."""
    from wands import data

    for name, digest in data.fetch().items():
        typer.echo(f"ok {name} sha256={digest}")


@data_app.command("profile")
def data_profile() -> None:
    """Regenerate docs/DATA_NOTES.md."""
    from wands.profile import write_notes

    typer.echo(f"wrote {write_notes()}")


@data_app.command("split")
def data_split() -> None:
    """Create splits.json. Refuses to change an existing split."""
    from wands import data
    from wands.eval.split import SPLITS_PATH, load_split, make_split, write_split

    queries = data.load().queries.select("query_id", "query_class").iter_rows()
    split = make_split(queries)
    if SPLITS_PATH.exists():
        if load_split("dev") != split["dev"] or load_split("test") != split["test"]:
            raise typer.BadParameter("splits.json differs from what this code generates; it is frozen")
        typer.echo("splits.json is unchanged")
        return
    write_split(split)
    typer.echo(f"wrote {SPLITS_PATH}: {len(split['dev'])} dev, {len(split['test'])} test")


@data_app.command("random-run")
def data_random_run(out: Path = Path("runs/random_run.tsv"), seed: int = 7) -> None:
    """Write a run that ranks 200 random products per query: the floor for every metric."""
    from wands import data
    from wands.eval.runfile import write_run

    wands = data.load()
    ids = wands.products["product_id"].to_list()
    rng = random.Random(seed)
    rows = {
        qid: [(pid, 200.0 - rank) for rank, pid in enumerate(rng.sample(ids, 200))]
        for qid in wands.queries["query_id"].to_list()
    }
    write_run(rows, out)
    typer.echo(f"wrote {out}")


def require_freeze() -> None:
    tags = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.split()
    if FREEZE_TAG not in tags:
        raise typer.BadParameter(f"the test split runs once, at tag {FREEZE_TAG}; HEAD is not tagged")


@app.command("eval")
def evaluate(
    config: Path = typer.Option(None, help="Pipeline YAML to run."),
    run: Path = typer.Option(None, help="Existing run file to score instead."),
    split: str = typer.Option("dev", help="dev or test."),
    depth: int = typer.Option(200, help="Results kept per query."),
) -> None:
    """Score a pipeline or a run file on a split and write a report."""
    from wands.eval.provenance import git_sha
    from wands.eval.report import render_report, score_run, write_report
    from wands.eval.runfile import read_run, write_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config, run_pipeline

    if (config is None) == (run is None):
        raise typer.BadParameter("give exactly one of --config and --run")
    if split == "test":
        require_freeze()

    world = World.load()
    query_ids = load_split(split)
    sha = git_sha()
    if config is not None:
        cfg = load_config(config)
        name = cfg["name"]
        hits = run_pipeline(build(cfg["pipeline"], world, name), world, query_ids, depth)
        rows = {qid: [(h.product_id, h.score) for h in found] for qid, found in hits.items()}
        run_path = RUNS_DIR / f"{name}_{split}.tsv"
        write_run(rows, run_path)
        ranked = {qid: [pid for pid, _ in found] for qid, found in rows.items()}
        source = str(config).replace("\\", "/")
    else:
        name = run.stem
        ranked = read_run(run)
        source = str(run).replace("\\", "/")

    scored = score_run(ranked, world.qrels, query_ids, name, split)
    text = render_report(scored, sha, source, world.queries)
    # A dirty tree can't be reproduced from a commit, so its report stays out of results/.
    path = write_report(scored, text, sha, RUNS_DIR if sha.endswith("-dirty") or run else Path("results"))
    typer.echo(text.split("## Per query")[0].rstrip())
    typer.echo(f"\nreport: {path}")


@qu_app.command("candidates")
def qu_candidates(split: str = typer.Option("dev", help="Split to report recall on.")) -> None:
    """Build every query's candidate class list, and report candidate recall on a split."""
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config
    from wands.qu import candidates as cand
    from wands.qu.evaluate import candidate_recall

    if split == "test":
        require_freeze()
    world = World.load()
    all_classes = sorted({c for classes in world.catalog.classes for c in classes})
    if not cand.CLASS_VECTORS_PATH.exists():
        cand.write_class_vectors(all_classes)
    names, vectors = cand.load_class_vectors()
    bm25 = build(load_config(Path("configs/bm25f.yaml"))["pipeline"], world)
    by_product = {int(p): c for p, c in zip(world.catalog.product_ids, world.catalog.classes, strict=True)}
    table = {
        text: cand.candidates_for(text, bm25, world.encoder, names, vectors, by_product)
        for text in world.queries.values()
    }
    cand.write_candidates(table)
    recall = candidate_recall(table, world.queries, world.query_classes, load_split(split))
    typer.echo({k: v for k, v in recall.items() if k != "missed"})
    for query, gold in recall["missed"][:25]:
        typer.echo(f"  missed: {query!r} -> {gold}")


def qu_items(world, query_ids: list[int]) -> list:
    from wands.qu.candidates import load_candidates
    from wands.qu.understand import Item

    table = load_candidates()
    return [Item(qid, world.queries[qid], table[world.queries[qid]]) for qid in query_ids]


@qu_app.command("run")
def qu_run(
    model: str = "gemini-3.5-flash-lite",
    prompt: str = "v1",
    split: str = "dev",
    limit: int = typer.Option(0, help="Only the first N queries, for a trial."),
    batch_size: int = 20,
) -> None:
    """Ask the LLM about a split's queries and append the answers to cache/qu_cache.jsonl."""
    from wands.eval.split import load_split
    from wands.pipeline import World
    from wands.qu.cache import QUCache
    from wands.qu.client import make_client
    from wands.qu.understand import understand

    if split == "test":
        require_freeze()
    world = World.load()
    ids = load_split(split)
    items = qu_items(world, ids[:limit] if limit else ids)
    typer.echo(understand(make_client(model), prompt, items, QUCache(), batch_size=batch_size))


@qu_app.command("eval")
def qu_eval(model: str = "gemini-3.5-flash-lite", prompt: str = "v1", split: str = "dev") -> None:
    """Score a model + prompt against labelled query classes and write results/qu_*.md."""
    from wands.eval.provenance import git_sha
    from wands.eval.split import load_split
    from wands.pipeline import World
    from wands.qu.cache import QUCache, QUTable
    from wands.qu.candidates import load_candidates
    from wands.qu.evaluate import candidate_recall, class_scores, render_scores

    if split == "test":
        require_freeze()
    world = World.load()
    ids = load_split(split)
    table = QUTable.from_cache(QUCache(), model, prompt)
    scores = class_scores(table, world.queries, world.query_classes, ids)
    recall = candidate_recall(load_candidates(), world.queries, world.query_classes, ids)
    fallbacks = table.fallback_counts([world.queries[q] for q in ids])
    text = render_scores(scores, model, prompt, split, git_sha(), fallbacks, recall)
    path = Path("results") / f"qu_{model.replace(':', '-')}_{prompt}_{split}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text.split("## Misses")[0].rstrip())
    typer.echo(f"\nreport: {path}")


@qu_app.command("breakdown")
def qu_breakdown(base: Path, boosted: Path, out: Path, split: str = "dev") -> None:
    """Where does query understanding help or hurt? By predicted intent, and by labelled class."""
    from wands.eval.provenance import git_sha
    from wands.eval.report import score_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config, run_pipeline
    from wands.qu.cache import QUCache, QUTable
    from wands.qu.evaluate import breakdown, render_breakdown

    if split == "test":
        require_freeze()
    world = World.load()
    ids = load_split(split)
    scored = []
    for config in (base, boosted):
        cfg = load_config(config)
        hits = run_pipeline(build(cfg["pipeline"], world, cfg["name"]), world, ids, 10)
        ranked = {qid: [h.product_id for h in found] for qid, found in hits.items()}
        scored.append(score_run(ranked, world.qrels, ids, cfg["name"], split))
    spec = load_config(boosted)["pipeline"]
    spec = spec.get("inner", spec) if spec["type"] == "rerank" else spec
    table = QUTable.from_cache(QUCache(), spec["model"], spec["prompt"])
    intents, classes, attrs = {}, {}, {}
    for qid in ids:
        qu, reason = table.lookup(world.queries[qid])
        intents[qid] = qu.intent if qu else reason
        classes[qid] = world.query_classes[qid] or "(none)"
        attrs[qid] = "has attributes" if qu and qu.attributes else "no attributes"
    text = "\n".join(
        [
            f"# Where query understanding helps: `{boosted.stem}` vs `{base.stem}`\n",
            f"Split `{split}`, commit `{git_sha()}`. Generated by `wands qu breakdown`.\n",
            render_breakdown(breakdown(scored[0], scored[1], intents), "By predicted intent"),
            render_breakdown(breakdown(scored[0], scored[1], attrs), "By whether attributes were predicted"),
            render_breakdown(
                breakdown(scored[0], scored[1], classes, min_size=4), "By labelled class, 4+ queries"
            ),
        ]
    )
    out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)


@qu_app.command("rewrite")
def qu_rewrite(
    model: str = "gemini-3.5-flash-lite",
    prompt: str = "v1",
    split: str = "dev",
    qu_prompt: str = "v1",
    every_query: bool = typer.Option(False, help="Rewrite all queries, into --out, as a comparison."),
    out: Path = Path("cache/rewrites.jsonl"),
) -> None:
    """Rewrite the queries QU was unsure about, and append them to cache/rewrites.jsonl."""
    from wands.eval.split import load_split
    from wands.pipeline import World
    from wands.qu.cache import QUCache, QUTable
    from wands.qu.client import make_client
    from wands.qu.rewrite import eligible, load_rewrites, rewrite_all, save_rewrites

    if split == "test":
        require_freeze()
    world = World.load()
    table = QUTable.from_cache(QUCache(), model, qu_prompt)
    known = load_rewrites(out)
    texts = [world.queries[q] for q in load_split(split)]
    todo = [t for t in texts if (every_query or eligible(table, t)) and t not in known]
    rewrites = rewrite_all(make_client(model), prompt, todo)
    save_rewrites(rewrites, out)
    changed = sum(q != r for q, r in rewrites.items())
    typer.echo(f"{len(todo)} eligible, {len(rewrites)} answered, {changed} changed")


@qu_app.command("rewrite-report")
def qu_rewrite_report(
    base: Path,
    rewritten: Path,
    out: Path,
    split: str = "dev",
    rewrites_file: Path = Path("cache/rewrites.jsonl"),
) -> None:
    """Compare with and without rewriting, on the queries actually changed and on the whole split."""
    from wands.eval.bootstrap import paired_bootstrap
    from wands.eval.metrics import METRICS
    from wands.eval.provenance import git_sha
    from wands.eval.report import compare_table, fmt_delta, score_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config, run_pipeline
    from wands.qu.rewrite import load_rewrites

    if split == "test":
        require_freeze()
    world = World.load()
    ids = load_split(split)
    scored = []
    for config in (base, rewritten):
        cfg = load_config(config)
        hits = run_pipeline(build(cfg["pipeline"], world, cfg["name"]), world, ids, 200)
        ranked = {qid: [h.product_id for h in found] for qid, found in hits.items()}
        scored.append(score_run(ranked, world.qrels, ids, cfg["name"], split))
    rewrites = load_rewrites(rewrites_file)
    changed = [
        i for i, q in enumerate(ids) if rewrites.get(world.queries[q], world.queries[q]) != world.queries[q]
    ]
    a, b = scored[0].scores["ndcg@10"][changed], scored[1].scores["ndcg@10"][changed]
    delta = paired_bootstrap(a, b)
    lines = [
        f"# LLM query rewriting, {split}",
        "",
        f"Commit `{git_sha()}`. Generated by `wands qu rewrite-report`, rewrites from `{rewrites_file.name}`. "
        f"{len(changed)} of {len(ids)} queries changed. Which queries to rewrite was chosen without labels.",
        "",
        compare_table(scored, scored[0], METRICS),
        f"On the {delta.n} changed queries with a score: ndcg@10 {np.nanmean(a):.4f} to {np.nanmean(b):.4f}, "
        f"delta {fmt_delta(delta)}.",
        "",
        "| query | rewrite | before | after |",
        "|---|---|---|---|",
    ]
    for i, position in enumerate(changed):
        text = world.queries[ids[position]]
        lines.append(f"| {text} | {rewrites[text]} | {a[i]:.3f} | {b[i]:.3f} |")
    report = "\n".join(lines) + "\n"
    out.write_text(report, encoding="utf-8", newline="\n")
    typer.echo(report)


@app.command("index")
def index(force: bool = typer.Option(False, help="Re-encode even if the files exist.")) -> None:
    """Encode every product to data/embeddings.npy and every query to cache/query_embeddings.npz."""
    import time

    import numpy as np

    from wands import data
    from wands.catalog import Catalog
    from wands.retrievers import dense

    wands = data.load()
    if force or not dense.EMBEDDINGS_PATH.exists():
        texts = Catalog.from_products(wands.products).embedding_text()
        start = time.perf_counter()
        vectors = dense.encode_products(texts)
        np.save(dense.EMBEDDINGS_PATH, vectors)
        typer.echo(f"encoded {vectors.shape} in {time.perf_counter() - start:.0f}s")
    if force or not dense.QUERY_CACHE_PATH.exists():
        dense.write_query_cache(wands.queries["query"].to_list())
        typer.echo(f"wrote {dense.QUERY_CACHE_PATH}")


@app.command("tune-bm25")
def tune_bm25_cmd() -> None:
    """Grid-search field weights on dev, log every config, and write configs/bm25f.yaml."""
    from wands import tune
    from wands.eval.split import load_split
    from wands.pipeline import World

    rows = tune.tune_bm25(World.load(), load_split("dev"))
    tune.write_grid(rows, Path("results/bm25_grid_dev.csv"))
    tune.write_bm25_config(rows[0], Path("configs/bm25f.yaml"))
    for row in rows[:5]:
        typer.echo(row)
    typer.echo(f"... worst {rows[-1]}")


@app.command("tune-fusion")
def tune_fusion_cmd(config: Path = Path("configs/hybrid_rrf.yaml")) -> None:
    """Grid-search alpha on dev, log it, and write configs/hybrid_weighted.yaml."""
    import yaml

    from wands import tune
    from wands.eval.split import load_split
    from wands.pipeline import World, load_config

    rows = tune.tune_fusion(World.load(), load_split("dev"), config)
    tune.write_grid(rows, Path("results/fusion_grid_dev.csv"))
    weighted = load_config(config)
    weighted["name"] = "hybrid_weighted"
    weighted["pipeline"] = {**weighted["pipeline"], "method": "weighted", "alpha": rows[0]["alpha"]}
    Path("configs/hybrid_weighted.yaml").write_text(
        yaml.safe_dump(weighted, sort_keys=False), encoding="utf-8"
    )
    for row in rows:
        typer.echo(row)


@app.command("tune-qu")
def tune_qu_cmd(config: Path, out: Path) -> None:
    """Grid-search boost weights on dev for a soft-boost config, log it, and update that config."""
    import yaml

    from wands import tune
    from wands.eval.split import load_split
    from wands.pipeline import World, load_config

    rows = tune.tune_qu(World.load(), load_split("dev"), config)
    tune.write_grid(rows, out)
    cfg = load_config(config)
    cfg["pipeline"].update(w_class=rows[0]["w_class"], w_attr=rows[0]["w_attr"])
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    for row in rows[:6]:
        typer.echo(row)
    typer.echo(f"... worst {rows[-1]}")


@app.command("compare")
def compare(
    configs: list[Path] = typer.Argument(..., help="Pipeline YAMLs; the first is the base row."),
    out: Path = typer.Option(..., help="Markdown table to write under results/."),
    split: str = "dev",
    title: str = "Comparison",
    depth: int = 200,
) -> None:
    """Run several configs and write a table with a bootstrap CI against the first."""
    from wands.eval.metrics import METRICS
    from wands.eval.provenance import git_sha
    from wands.eval.report import compare_table, score_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config, run_pipeline

    if split == "test":
        require_freeze()
    world = World.load()
    query_ids = load_split(split)
    rows = []
    for config in configs:
        cfg = load_config(config)
        if split == "dev" and cfg["pipeline"]["type"] == "ltr":
            raise typer.BadParameter("the LTR model was trained on dev; use `wands ltr cv` for dev scores")
        hits = run_pipeline(build(cfg["pipeline"], world, cfg["name"]), world, query_ids, depth)
        ranked = {qid: [h.product_id for h in found] for qid, found in hits.items()}
        rows.append(score_run(ranked, world.qrels, query_ids, cfg["name"], split))
        typer.echo(f"{cfg['name']}: ndcg@10 {rows[-1].mean('ndcg@10'):.4f}")
    table = compare_table(rows, rows[0], METRICS)
    sources = ", ".join(f"`{str(c).replace(chr(92), '/')}`" for c in configs)
    text = (
        f"# {title}\n\nSplit `{split}`, {len(query_ids)} queries, commit `{git_sha()}`, configs {sources}. "
        "Generated by `wands compare`. `*` marks a delta whose 95% paired-bootstrap CI excludes 0.\n\n"
        + table
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)


@ltr_app.command("cv")
def ltr_cv(config: Path = Path("configs/hybrid_qu.yaml")) -> None:
    """5-fold CV inside dev: out-of-fold scores against the same queries, and feature importance."""
    from wands import ltr
    from wands.eval.metrics import METRICS
    from wands.eval.provenance import git_sha
    from wands.eval.report import compare_table, score_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, load_config, run_pipeline

    world = World.load()
    query_ids = load_split("dev")
    pipeline = load_config(config)["pipeline"]
    features = ltr.builder_for(pipeline, world)
    data = {qid: features.rows(world.queries[qid]) for qid in query_ids}
    judged_only, _ = ltr.cross_validate(data, world, query_ids, unjudged_as_zero=False)
    ranked, importance = ltr.cross_validate(data, world, query_ids, unjudged_as_zero=True)
    base_hits = run_pipeline(build(pipeline, world), world, query_ids, 200)
    base_ranked = {qid: [h.product_id for h in found] for qid, found in base_hits.items()}
    base = score_run(base_ranked, world.qrels, query_ids, config.stem, "dev")
    rows = [
        base,
        score_run(judged_only, world.qrels, query_ids, "ltr_judged_only", "dev"),
        score_run(ranked, world.qrels, query_ids, "ltr_unjudged_as_zero", "dev"),
    ]
    lines = [
        "# Learning to rank, 5-fold CV inside dev",
        "",
        f"{len(query_ids)} dev queries, commit `{git_sha()}`. Generated by `wands ltr cv`. Every query is "
        "scored by a model trained on the other four folds. `*` marks a delta whose 95% paired-bootstrap CI "
        "excludes 0. The base row's weights were tuned on all of dev, which flatters it slightly.",
        "",
        compare_table(rows, base, METRICS),
        "## Feature importance of `ltr_unjudged_as_zero` (share of total gain)",
        "",
        "| feature | share |",
        "|---|---|",
    ]
    for j in importance.argsort()[::-1]:
        lines.append(f"| {ltr.FEATURES[j]} | {importance[j]:.3f} |")
    text = "\n".join(lines) + "\n"
    Path("results/ltr_dev.md").write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)


@ltr_app.command("train")
def ltr_train(config: Path = Path("configs/hybrid_qu.yaml")) -> None:
    """Train on all of dev and save the model for the single test run."""
    from wands import ltr
    from wands.eval.split import load_split
    from wands.pipeline import World, load_config

    world = World.load()
    query_ids = load_split("dev")
    features = ltr.builder_for(load_config(config)["pipeline"], world)
    data = {qid: features.rows(world.queries[qid]) for qid in query_ids}
    ltr.fit(data, world, query_ids, unjudged_as_zero=True).booster_.save_model(str(ltr.MODEL_PATH))
    typer.echo(f"wrote {ltr.MODEL_PATH}")


def judge_pairs(world, config: Path, query_ids: list[int]):
    """The calibration pairs, and the final pipeline's unjudged top-10 pairs, as (qid, pid, gain, Pair)."""
    from wands.judge import Pair, product_card, sample_labelled
    from wands.pipeline import build, load_config, run_pipeline

    positions = {int(pid): i for i, pid in enumerate(world.catalog.product_ids)}

    def pair(qid: int, pid: int) -> Pair:
        return Pair(world.queries[qid], pid, product_card(world.catalog, positions[pid]))

    labelled = [(q, p, g, pair(q, p)) for q, p, g in sample_labelled(world.qrels, query_ids)]
    cfg = load_config(config)
    hits = run_pipeline(build(cfg["pipeline"], world, cfg["name"]), world, query_ids, 10)
    ranked = {qid: [h.product_id for h in found] for qid, found in hits.items()}
    unjudged = [
        (qid, pid, -1, pair(qid, pid))
        for qid, found in ranked.items()
        for pid in found
        if pid not in world.qrels.get(qid, {})
    ]
    return labelled, unjudged, ranked


@judge_app.command("run")
def judge_run(
    model: str = "gemini-3.5-flash-lite", prompt: str = "v1", config: Path = Path("configs/final.yaml")
) -> None:
    """Ask the LLM to judge the calibration pairs and the unjudged top-10 pairs on dev."""
    from wands.eval.split import load_split
    from wands.judge import JudgeCache, judge
    from wands.pipeline import World
    from wands.qu.client import make_client

    world = World.load()
    labelled, unjudged, _ = judge_pairs(world, config, load_split("dev"))
    pairs = [row[3] for row in labelled + unjudged]
    typer.echo(f"{len(labelled)} labelled, {len(unjudged)} unjudged")
    typer.echo(judge(make_client(model), prompt, pairs, JudgeCache()))


@judge_app.command("report")
def judge_report(
    model: str = "gemini-3.5-flash-lite", prompt: str = "v1", config: Path = Path("configs/final.yaml")
) -> None:
    """Agreement with human labels, and the estimated effect of unjudged results."""
    from wands import judge as jd
    from wands.eval.bootstrap import paired_bootstrap
    from wands.eval.provenance import git_sha
    from wands.eval.report import fmt_delta
    from wands.eval.split import load_split
    from wands.pipeline import World

    world = World.load()
    ids = load_split("dev")
    labelled, unjudged, ranked = judge_pairs(world, config, ids)
    template = (jd.PROMPTS_DIR / f"judge_{prompt}.md").read_text(encoding="utf-8")
    cache = jd.JudgeCache()

    def label_of(pair: jd.Pair) -> str | None:
        entry = cache.entries.get(jd.pair_key(model, template, pair))
        return entry["label"] if entry else None

    scored = [(g, jd.LABELS.index(label_of(p)), p) for _, _, g, p in labelled if label_of(p)]
    matrix = jd.confusion([t for t, _, _ in scored], [g for _, g, _ in scored])
    accuracy = float(np.trace(matrix) / matrix.sum())
    kappa = jd.weighted_kappa(matrix)
    llm = {(q, pid): jd.LABELS.index(label_of(p)) for q, pid, _, p in unjudged if label_of(p)}
    filled = jd.fill_ndcg(ranked, world.qrels, llm, ids)
    delta = paired_bootstrap(filled["scores_before"], filled["scores_after"])
    names = jd.LABELS
    lines = [
        f"# LLM relevance judge: `{model}`, prompt `{prompt}`, dev",
        "",
        f"Commit `{git_sha()}`. Generated by `wands judge report`. Agreement with human labels on "
        f"{matrix.sum()} labelled dev pairs, sampled evenly across the three labels "
        f"({len(labelled) - matrix.sum()} unanswered). The LLM sees text only; annotators also saw images.",
        "",
        "| measure | value |",
        "|---|---|",
        f"| accuracy | {accuracy:.3f} |",
        f"| weighted Cohen's kappa (linear) | {kappa:.3f} |",
        "",
        "## Confusion matrix (rows: human, columns: LLM)",
        "",
        "| human \\ LLM | " + " | ".join(names) + " |",
        "|---|---|---|---|",
    ]
    lines += [f"| {names[i]} | " + " | ".join(str(v) for v in matrix[i]) + " |" for i in range(3)]
    lines += [
        "",
        "## Far misses: one says Exact, the other Irrelevant (first 15)",
        "",
        "| query | human | LLM | product |",
        "|---|---|---|---|",
    ]
    far = [(t, g, p) for t, g, p in scored if abs(t - g) == 2]
    for t, g, p in far[:15]:
        lines.append(
            f"| {p.query} | {names[t]} | {names[g]} | {p.product.split(' | ')[0].removeprefix('name: ')} |"
        )
    lines += [
        "",
        "## Effect of unjudged results on the final pipeline",
        "",
        f"The LLM labelled {len(llm)} of {len(unjudged)} unjudged top-10 products on dev. Human labels are "
        "never changed. This estimates the bias from scoring unjudged as 0; it is not a measurement.",
        "",
        "| measure | as scored | with LLM labels |",
        "|---|---|---|",
        f"| judged@10 | {filled['judged_before']:.4f} | {filled['judged_after']:.4f} |",
        f"| ndcg@10 | {filled['ndcg_before']:.4f} | {filled['ndcg_after']:.4f} |",
        "",
        f"Delta ndcg@10 [95% CI]: {fmt_delta(delta)}",
    ]
    text = "\n".join(lines) + "\n"
    path = Path(f"results/judge_{model.replace(':', '-')}_{prompt}_dev.md")
    path.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)


@app.command("gate")
def gate(
    config: Path = Path("configs/final.yaml"),
    write_baseline: bool = typer.Option(False, help="Record this run as the new baseline."),
) -> None:
    """Run a config on dev and fail if it regresses against results/baseline.json."""
    from wands.eval import gate as gate_rule
    from wands.eval.provenance import git_sha
    from wands.eval.report import score_run
    from wands.eval.split import load_split
    from wands.pipeline import World, build, find_booster, load_config, run_pipeline

    world = World.load()
    query_ids = load_split("dev")
    cfg = load_config(config)
    retriever = build(cfg["pipeline"], world, cfg["name"])
    hits = run_pipeline(retriever, world, query_ids, 200)
    ranked = {qid: [h.product_id for h in found] for qid, found in hits.items()}
    scored = score_run(ranked, world.qrels, query_ids, cfg["name"], "dev")
    booster = find_booster(retriever)
    texts = [world.queries[q] for q in query_ids]
    fallbacks = booster.table.fallback_counts(texts) if booster else {}
    source = str(config).replace("\\", "/")
    report = gate_rule.make_report(source, "dev", git_sha(), scored.means(), fallbacks)
    if write_baseline:
        gate_rule.write_baseline(report)
        typer.echo(f"wrote {gate_rule.BASELINE_PATH}: {report['metrics']}")
        return
    ok, lines = gate_rule.check(report, gate_rule.load_baseline())
    for line in lines:
        typer.echo(line)
    typer.echo("gate passed" if ok else "gate FAILED")
    raise typer.Exit(0 if ok else 1)


@app.command("latency")
def latency(config: Path = Path("configs/final_qu.yaml"), queries: int = 100, depth: int = 50) -> None:
    """Time each online stage live on dev queries and write results/latency.md."""
    from sentence_transformers import CrossEncoder

    from wands.eval.latency import measure, render_latency
    from wands.eval.provenance import git_sha
    from wands.eval.split import load_split
    from wands.pipeline import World, build, find_booster, load_config
    from wands.rerank import CE_MODEL, CrossEncoderScorer

    world = World.load()
    cfg = load_config(config)
    booster = find_booster(build(cfg["pipeline"], world, cfg["name"]))
    fusion = booster.inner
    bm25, dense = fusion.retrievers
    scorer = CrossEncoderScorer(world.catalog, cache_path=None)
    scorer.model = CrossEncoder(CE_MODEL, device="cpu")
    texts = [world.queries[q] for q in load_split("dev")[:queries]]
    rows = measure(world, bm25, dense, fusion, booster, scorer, texts, depth)
    text = render_latency(rows, git_sha(), len(texts), "Intel i7-12700H, no GPU")
    Path("results/latency.md").write_text(text, encoding="utf-8", newline="\n")
    typer.echo(text)


@app.command("serve")
def serve(config: Path = Path("configs/final.yaml"), port: int = 8000) -> None:
    """Run the demo API and page on localhost."""
    import uvicorn

    from wands.api import create_app

    uvicorn.run(create_app(config), host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
