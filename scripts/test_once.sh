#!/usr/bin/env bash
# The single test run. Every command refuses unless HEAD carries the freeze-v1 tag, and `wands test-report`
# refuses if a test report already exists. Recording Qwen's answers takes about an hour on a laptop CPU.
set -euo pipefail

wands qu run --model gemini-3.5-flash-lite --prompt v1 --split test
wands qu run --model qwen2.5:3b --prompt v1 --split test --batch-size 10
wands qu eval --model gemini-3.5-flash-lite --prompt v1 --split test
wands qu eval --model qwen2.5:3b --prompt v1 --split test
wands test-report
wands qu breakdown configs/hybrid_rrf.yaml configs/final.yaml results/qu_breakdown_test.md --split test
