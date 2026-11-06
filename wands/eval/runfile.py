"""Run files: one TSV row per retrieved product (query_id, product_id, rank, score)."""

from __future__ import annotations

from pathlib import Path

Run = dict[int, list[int]]  # query_id -> product ids, best first


def write_run(rows: dict[int, list[tuple[int, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("query_id\tproduct_id\trank\tscore\n")
        for query_id in sorted(rows):
            for rank, (product_id, score) in enumerate(rows[query_id], start=1):
                f.write(f"{query_id}\t{product_id}\t{rank}\t{score:.6f}\n")


def read_run(path: Path) -> Run:
    ranked: dict[int, list[tuple[int, int]]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            query_id, product_id, rank, _ = line.rstrip("\n").split("\t")
            ranked.setdefault(int(query_id), []).append((int(rank), int(product_id)))
    return {qid: [pid for _, pid in sorted(rows)] for qid, rows in ranked.items()}
