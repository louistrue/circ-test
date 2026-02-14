# Testing Guide

This guide explains how to test the IFC product matching codebase.

## Quick Start: Get Embedding Matching Running

The project requires **Python 3.11+**.

```bash
cd /Users/louistrue/Development/circ-test

# Create virtual environment with Python 3.12 (Homebrew)
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install and run
pip install -e ".[dev]"
python main.py
```

Or run the CLI directly:

```bash
ifc-match match "concrete" --candidates cement brick "rigid insulation" --top-k 3
```

## Run Unit Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Alternatively, without installing (lightweight tests only):

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

## Test Coverage

| Test File | What It Tests |
|-----------|---------------|
| `test_circularity.py` | BCI (Building Circularity Indicator), MCI, DI, ECI formulas |
| `test_database.py` | TOTEM/OKOBAUDAT/KBOB databases, `ConnectionType`, archetypes |
| `test_embedding.py` | Tokenization, cosine similarity, embedding matching (with mocks) |
| `test_hybrid.py` | Hybrid retrieve-rerank pipeline (with mocks) |
| `test_proximity.py` | Bounding boxes, geometric proximity analysis |
| `test_reranker.py` | LLM response parsing, prompt building |
| `test_improvements.py` | Quick 8-case before/after preprocessing validation |

## Running the Full Demo

The `main.py` demo and CLI require `sentence-transformers` (and optionally OpenAI for upgraded embeddings). The project officially requires **Python 3.11+**.

### Option 1: Conda (recommended)

```bash
conda env create -f 00_condaEnvs/ifcProductMatching.yml
conda activate IfcProductMatching
pip install -e .
python main.py
```

### Option 2: pip with Python 3.11+

```bash
# Ensure you have Python 3.11 or newer
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
python main.py
```

### CLI Usage

```bash
# Embedding-only matching (requires sentence-transformers)
ifc-match match "concrete" --candidates cement brick "rigid insulation" --top-k 3

# From a JSON database file
ifc-match match "gypsum board" --database-file path/to/materials.json --top-k 5

# Geometric proximity on an IFC file (requires ifcopenshell)
ifc-match proximity path/to/model.ifc
```

## Run the Benchmark (Embedding vs Hybrid)

Compares the original embedding-only approach against the hybrid retrieve-rerank
pipeline. See **README.md** for detailed results and analysis.

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY=sk-ant-...   # or source .env
python benchmarks/benchmark_multimodel.py
```

Quick preprocessing validation (8 cases, no LLM):

```bash
python tests/test_improvements.py
```

## Test Categories

- **Pure unit tests** (no external services): circularity, database, proximity, reranker parsing
- **Integration-style tests** (use mocks): embedding, hybrid
- **Requires API keys**: hybrid mode with Claude/OpenAI reranker, `bci` command with Neo4j
