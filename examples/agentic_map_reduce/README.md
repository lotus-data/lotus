# Agentic map-reduce examples

Demos of LOTUS agentic map-reduce: the user provides a `Corpus` + a `task`; a planner
derives the map/reduce instructions and sharding; each shard is processed in parallel by
a tool-using agent (with a sandboxed Python REPL); the per-shard results are reduced into
one answer. Tool usage is handled transparently — the `task` never has to mention tools.

## Examples

| File | Corpus | What it shows |
|------|--------|---------------|
| `expense_reports.py` | `Corpus.from_documents` | Exact numeric aggregation — per-report totals (map) → grand total + top category (reduce), all computed via the REPL. |
| `codebase_sweep.py` | `Corpus.from_files` | Codebase analysis (Devin-style) — per-file summary (map) → architecture overview (reduce). Sweeps LOTUS's own `lotus/agentic/*.py` by default; pass a glob to sweep something else. |

## Running

Requires `OPENAI_API_KEY`. Run from the repo root with it on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD" python examples/agentic_map_reduce/expense_reports.py
PYTHONPATH="$PWD" python examples/agentic_map_reduce/codebase_sweep.py
PYTHONPATH="$PWD" python examples/agentic_map_reduce/codebase_sweep.py "lotus/sem_ops/*.py"
```
