#!/usr/bin/env bash
# Regenerate every dev table from a clean commit, so they all name the same SHA.
# LLM answers come from cache/. Needs data/ and data/embeddings.npy (`wands data fetch`, `wands index`).
set -euo pipefail

gemini=gemini-3.5-flash-lite
qwen=qwen2.5:3b

wands data profile
wands compare configs/bm25_name.yaml configs/bm25_name_nostem.yaml configs/bm25_all.yaml configs/bm25f.yaml \
  configs/bm25f_nostem.yaml configs/bm25f_rawfeatures.yaml \
  --out results/bm25_dev.md --title "BM25 baselines against name-only"
wands compare configs/bm25f.yaml configs/bm25f_nostem.yaml configs/bm25f_rawfeatures.yaml configs/bm25_all.yaml \
  --out results/bm25_ablations_dev.md --title "BM25 ablations against the tuned weights"
wands compare configs/bm25f.yaml configs/dense.yaml configs/hybrid_rrf.yaml configs/hybrid_weighted.yaml \
  --out results/fusion_dev.md --title "Dense retrieval and fusion"
wands qu eval --model "$gemini" --prompt v1
wands qu eval --model "$gemini" --prompt v2
wands qu eval --model "$qwen" --prompt v1
wands compare configs/hybrid_rrf.yaml configs/hybrid_qu.yaml configs/hybrid_qu_v2.yaml configs/hybrid_qu_filter.yaml \
  configs/hybrid_qu_qwen.yaml configs/hybrid_qu_qwen_filter.yaml \
  --out results/qu_dev.md --title "Query understanding in ranking"
wands qu breakdown configs/hybrid_rrf.yaml configs/hybrid_qu.yaml results/qu_breakdown_dev.md
wands compare configs/hybrid_qu.yaml configs/rerank_d20.yaml configs/rerank_d50.yaml configs/rerank_d100.yaml \
  configs/rerank_rrf_d20.yaml configs/rerank_rrf_d50.yaml \
  --out results/rerank_dev.md --title "Cross-encoder reranking on top of hybrid + QU"
wands compare configs/hybrid_qu.yaml configs/pop_0.02.yaml configs/pop_0.05.yaml configs/pop_0.1.yaml configs/pop_0.2.yaml \
  --out results/popularity_dev.md --title "Popularity prior on top of hybrid + QU"
wands ltr cv
wands judge report
wands qu rewrite-report configs/hybrid_qu.yaml configs/hybrid_rewrite.yaml results/rewrite_dev.md
wands qu rewrite-report configs/hybrid_qu.yaml configs/hybrid_rewrite_all.yaml results/rewrite_all_dev.md \
  --rewrites-file cache/rewrites_all.jsonl
wands gate --write-baseline
