"""The demo: `GET /search` over a pipeline config, and the static page in web/.

No LLM is called here. Query understanding comes from the cache, so a query outside it gets no boost.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query as Param
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from wands import data
from wands.pipeline import World, build, find_booster, load_config
from wands.qu.cache import QUTable
from wands.types import Hit, Query

WEB_DIR = Path("web")


def understood(table: QUTable | None, text: str) -> dict:
    if table is None:
        return {"status": "not used"}
    qu, reason = table.lookup(text)
    if qu is None:
        return {"status": reason}
    return {"status": reason, **qu.model_dump()}


def describe(hit: Hit, rank: int, row: dict) -> dict:
    return {
        "rank": rank,
        "product_id": hit.product_id,
        "name": row["product_name"],
        "class": row["product_class"],
        "rating": row["average_rating"],
        "rating_count": row["rating_count"],
        "score": round(hit.score, 4),
        "why": hit.debug,
    }


def create_app(config: Path = Path("configs/final.yaml")) -> FastAPI:
    app = FastAPI(title="wands-search")
    world = World.load()
    cfg = load_config(config)
    retriever = build(cfg["pipeline"], world, cfg["name"])
    booster = find_booster(retriever)
    table = booster.table if booster else None
    rows = {row["product_id"]: row for row in data.load().products.iter_rows(named=True)}

    @app.get("/search")
    def search(q: str = Param(..., min_length=1, max_length=200), k: int = Param(24, ge=1, le=100)) -> dict:
        hits = retriever.search(Query(text=q), k)
        return {
            "query": q,
            "pipeline": cfg["name"],
            "understanding": understood(table, q),
            "results": [describe(h, i, rows[h.product_id]) for i, h in enumerate(hits, start=1)],
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
    return app
