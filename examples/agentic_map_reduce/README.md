# Agentic operator examples

Demos of LOTUS agentic operators: the user provides a `Corpus` + a `task` + an ordered
list of `ops` (`map` / `filter` / `reduce`); a planner derives each op's instruction,
sharding, and strategy; each unit is processed in parallel by a tool-using agent (with a
sandboxed Python REPL). Tool usage is handled transparently — the `task` never has to
mention tools. `map`/`filter` return a corpus; `reduce` returns one answer.

## Examples

| File | Ops | What it shows |
|------|-----|---------------|
| `expense_reports.py` | `map` → `reduce` | Exact numeric aggregation — per-report totals (map) → grand total + top category (reduce), all computed via the REPL. |
| `codebase_sweep.py` | `map` → `reduce` | Codebase analysis — per-file summary (map) → architecture overview (reduce). Sweeps LOTUS's own `lotus/agentic/*.py` by default; pass a glob to sweep something else. |
| `buggy_filter.py` | `filter` | Agentic filter that *needs* tools — each agent runs a function in the REPL and keeps only the buggy ones (something a single LLM call can't reliably do). |

## Running

Requires `OPENAI_API_KEY`. Run from the repo root with it on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD" python examples/agentic_map_reduce/expense_reports.py
PYTHONPATH="$PWD" python examples/agentic_map_reduce/codebase_sweep.py
PYTHONPATH="$PWD" python examples/agentic_map_reduce/codebase_sweep.py "lotus/sem_ops/*.py"
PYTHONPATH="$PWD" python examples/agentic_map_reduce/buggy_filter.py
```
