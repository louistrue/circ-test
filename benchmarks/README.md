# Benchmarks

- **benchmark_multimodel.py** — 56-test KBOB benchmark (7 IFC files, ground truth in file)
- **benchmark_oekobaudat.py** — 56-test ÖKOBAUDAT benchmark (keyword extraction + multi-search)
- **benchmark_full.py** — 5-way model comparison (legacy, MiniLM, BGE-M3, +Claude, +GPT)
- **benchmark_real.py** — Single IFC file (M-BIN-TW-51) vs KBOB

Run from project root:

```bash
python benchmarks/benchmark_multimodel.py
python benchmarks/benchmark_oekobaudat.py
```

Requires `LCADATA_API_KEY` in `.env` for KBOB benchmarks.

**Reproducibility:** Benchmark scripts write `benchmark_*_results.json` to the project root. Commit these files so others can verify results without re-running (which requires API keys and ~10–20 min).
