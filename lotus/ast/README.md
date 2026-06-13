# LOTUS AST — Architecture and Internals

This document covers the internal architecture of the `lotus.ast` module: the
node type system, execution engine, caching strategy, optimizer framework, and
reference resolution mechanics. For usage-focused documentation see the
[Sphinx docs](../../docs/ast.rst).

---

## Module Layout

```
lotus/ast/
├── __init__.py              # Public re-exports: LazyFrame, LazyFrameRun, all node types
├── lazyframe.py             # LazyFrame builder — immutable pipeline construction
├── nodes.py                 # Pydantic AST node types with __call__ execution
├── run.py                   # LazyFrameRun — sequential execution engine with caching
├── cache.py                 # Content-addressable cache utilities (hashing, keys)
└── optimizer/
    ├── __init__.py           # Exports all optimizers
    ├── base.py               # BaseOptimizer ABC
    ├── predicate_pushdown.py # Moves pandas filters before sem_filters
    ├── gepa_optimizer.py     # GEPA-based prompt/instruction optimization
    └── cascade.py  # Pre-warms cascade thresholds
```

---

## Design Principles

1. **Immutability** — Every LazyFrame chain method returns a new LazyFrame
   instance with the node appended. The only exception is `lf['col'] = val`
   which mutates in-place for ergonomics (use `.assign()` for immutable
   assignment).

2. **Pydantic Nodes** — All nodes are `pydantic.BaseModel` subclasses. This
   gives us validated construction, `model_copy()` for immutable updates,
   `model_dump()` for serialization, and a structured field system that
   optimizers can introspect.

3. **Resolver Pattern** — Nodes don't know how to execute sub-pipelines. Each
   node's `__call__` receives a `resolver` callable that recursively
   materializes `LazyFrame` and `SourceNode` references into DataFrames.
   This keeps nodes decoupled from the execution engine.

4. **Content-Addressable Cache** — Cache keys are computed from
   `NodeType:hash(node_config):hash(input_data)`. Shared sub-pipelines
   (e.g. the same LazyFrame used as both a join right-side and an assign
   value) are executed once and cached.

---

## Node Type Hierarchy

```
BaseNode (Pydantic BaseModel)
│
├── SourceNode              — Input data source; holds optional bound df + schema
│
├── Semantic Operators
│   ├── SemFilterNode       — Filter by natural language predicate
│   ├── SemMapNode          — Per-row transformation via language instruction
│   ├── SemExtractNode      — Extract structured attributes into new columns
│   ├── SemAggNode          — Aggregate/summarize rows
│   ├── SemTopKNode         — Rank and return top K rows
│   ├── SemJoinNode         — Join on natural language predicate (via _JoinMixin)
│   ├── SemSimJoinNode      — Similarity-based join (via _JoinMixin)
│   ├── SemSearchNode       — Semantic similarity search
│   ├── SemIndexNode        — Build semantic vector index
│   ├── LoadSemIndexNode    — Load existing semantic index
│   ├── SemClusterByNode    — Semantic clustering
│   ├── SemDedupNode        — Semantic deduplication
│   └── SemPartitionByNode  — Partition by callable
│
├── Pandas Operators
│   ├── PandasFilterNode    — Boolean predicate filter
│   └── PandasOpNode        — Generic pandas method/attr/subscript/assign
│
├── Eval Operators
│   ├── LLMAsJudgeNode      — Multi-trial LLM judge evaluation
│   └── PairwiseJudgeNode   — Pairwise comparison judge
│
└── Function Nodes
    └── ApplyFnNode         — Arbitrary callable over resolved inputs
```

### BaseNode Interface

Every node implements three hooks:

```python
class BaseNode(BaseModel):
    def __call__(self, df, resolver=_no_resolver, **context):
        """Execute the node. resolver materializes LazyFrame/SourceNode refs."""

    def signature(self) -> str:
        """One-line summary for tree display (used by LazyFrame.show())."""

    def child_lfs(self) -> list[tuple[str, Any]]:
        """Return (label, LazyFrame) pairs for nested pipeline display."""
```

### Optimization Hooks

BaseNode provides a generic system for addressing, reading, and writing
parameters at arbitrary nesting depth. This is used by the GEPA optimizer
to discover and tune parameters without node-specific logic.

```python
node.supports_optimizable_param("cascade_args.helper_filter_instruction")
node.resolve_optimizable_param_value("user_instruction")
node.apply_optimizable_param_value("user_instruction", "new value")  # returns new node
```

Parameter paths support dot-notation for Pydantic model fields and
bracket-notation for list/dict access (e.g. `cascade_args.thresholds[0]`).

### PandasOpNode Dispatch

`PandasOpNode` handles four patterns uniformly:

| Pattern | Example | Internal representation |
|---------|---------|------------------------|
| Method call | `lf.sort_values("col")` | `op_name="sort_values", args=("col",)` |
| Attribute | `lf.columns` | `op_name="columns", is_attr=True` |
| Subscript | `lf[["a","b"]]` | `op_name="__getitem__", args=(["a","b"],)` |
| Assignment | `lf.assign(x=val)` | `op_name="assign", kwargs={"x": val}` |

When arguments contain LazyFrame references, they are separated into
`lf_args`/`lf_kwargs` dicts and resolved at execution time via the resolver.

### ApplyFnNode

Used by `LazyFrame.from_fn()` and `LazyFrame.concat()`. Arguments (including
nested lists/tuples/dicts) are recursively resolved before the function is
called. This is the only node type that doesn't require a preceding
DataFrame input.

---

## Execution Engine (LazyFrameRun)

`LazyFrameRun` walks the node list sequentially:

```
for each node in lazyframe._nodes:
    1. Compute cache key = NodeType:hash(node):hash(current_input)
    2. Cache hit? → use cached result, skip execution
    3. SourceNode? → resolve from inputs dict or bound df
    4. Other node? → node(current_df, resolver=self._resolve_ref, **configs)
    5. Cache the result
    6. Update current_hash for next node
```

### Reference Resolution

The runner passes `_resolve_ref` as the `resolver` argument to every node.
This method handles:

| Reference type | Resolution |
|----------------|------------|
| `LazyFrame` | Creates a sub-`LazyFrameRun` sharing the same cache and inputs |
| `SourceNode` | Looks up `node.lazyframe_ref` in the inputs dict |
| `pd.DataFrame` | Passes through unchanged |
| `list` / `tuple` / `dict` | Recursively resolves all elements |
| Anything else | Passes through unchanged |

Sub-`LazyFrameRun` instances share the content cache. This means if two
join nodes reference the same LazyFrame, it is executed once and the result
is reused from cache.

### Input Resolution for SourceNode

When `execute()` is called with `dict[LazyFrame, DataFrame]`:

1. The runner looks up `source_node.lazyframe_ref` in the inputs dict.
2. If not found and there is exactly one input, it uses that (single-source
   convenience).
3. Schema validation runs if `expected_schema` is set.

---

## Caching Strategy

### Cache Key Computation

```python
cache_key = f"{type(node).__name__}:{hash_node(node)}:{input_hash}"
```

- `hash_node(node)` — Deterministic hash of all Pydantic fields. DataFrames
  hash by content (pickle → MD5). Callables and LazyFrame refs hash by
  `id()` (identity-based for session-local dedup).
- `input_hash` — For SourceNode: hash of the concrete input DataFrame. For
  other nodes: hash of the previous step's output.

### hash_dataframe

Uses `pickle.dumps(df)` → MD5 truncated to 16 hex chars. Captures values,
dtypes, index/column labels, names, ordering, and attrs.

### hash_result

For DataFrames: uses `hash_dataframe`. For other objects with `__len__`:
uses `pd.util.hash_pandas_object`. Fallback: `hash(str(result))`.

### Cache Sharing

When a `LazyFrameRun` creates sub-runs for nested LazyFrames (joins,
assigns with LazyFrame values, `from_fn`), they share the same `Cache`
instance and the same `cache_stats` dict. This ensures:
- Common prefixes are computed once
- Cache statistics reflect the full execution tree

---

## Optimizer Framework

### BaseOptimizer

```python
class BaseOptimizer(ABC):
    requires_train_data: bool = False

    @abstractmethod
    def optimize(self, nodes: list[BaseNode], train_data=None) -> list[BaseNode]:
        """Return an optimized node list."""
```

`LazyFrame.optimize()` applies optimizers sequentially, passing the output
of one to the input of the next.

### PredicatePushdownOptimizer

**Goal:** Reduce the number of rows processed by expensive LLM-based
semantic operators.

**Algorithm:** For each `PandasFilterNode` in the node list, bubble it
backward past consecutive `SemFilterNode`s by swapping adjacent nodes.

**Safety invariant:** `sem_filter` only removes rows — it never adds or
renames columns. Therefore a pandas filter that depends on existing columns
can safely run before `sem_filter`.

**Requires train data:** No.

```
Before: Source → SemFilter → PandasFilter
After:  Source → PandasFilter → SemFilter
```

### GEPAOptimizer

**Goal:** Automatically tune natural language instructions using LLM-guided
evolutionary search.

**Default optimizable parameters:**

| Node Type | Parameters |
|-----------|-----------|
| `SemFilterNode` | `user_instruction`, `cascade_args.helper_filter_instruction` |
| `PairwiseJudgeNode` | `judge_instruction`, `cascade_args.helper_filter_instruction` |
| `SemMapNode` | `user_instruction` |
| `SemAggNode` | `user_instruction` |
| `SemTopKNode` | `user_instruction` |
| `SemJoinNode` | `join_instruction` |
| `SemSearchNode` | `query` |

**Pipeline:**

1. **Target collection** — `_walk()` traverses the node tree (including
   nested LazyFrames in joins, assigns, and `from_fn`) collecting
   `_OptTarget` instances. Each target records `(node_idx, param_name,
   current_value, step_idx, path)`.

2. **Seed candidate** — Current parameter values become the initial GEPA
   candidate dict with keys like `step0_user_instruction`.

3. **Evaluator** — For each `(candidate, example)` pair, the evaluator:
   - Applies candidate values to a deep copy of the node list
   - Executes the patched pipeline on the example
   - Calls the user's `eval_fn(output_df, example)` for scoring

4. **Evolutionary search** — GEPA runs mutation/reflection cycles to
   discover better parameter values.

5. **Candidate application** — `_apply_candidate()` deep-copies the original
   nodes, groups targets by path (deepest first), and patches values
   recursively. Source refs are restored after deep copy so input dict
   lookups still work.

**PathEntry** — Addresses nested LazyFrames via
`(node_idx, field_name, sub_path)`:

```python
PathEntry(2, "right_lf")           # join's right-hand LazyFrame
PathEntry(0, "lf_args", ("key",))  # PandasOpNode LazyFrame argument
PathEntry(0, "args", (0, 1))       # ApplyFnNode nested argument
```

**Requires train data:** Yes.

### CascadeOptimizer

**Goal:** Pre-warm cascade thresholds so subsequent runs skip threshold
learning.

**Algorithm:** Executes the pipeline once on training data with
`update_cascade_args=True` in the runtime context. `SemFilterNode` and
`SemJoinNode` nodes with `cascade_args` update their thresholds in-place
during execution.

**Requires train data:** Yes.

---

## LazyFrame Construction Internals

### Deep Copy and Source Ref Restoration

`LazyFrame.copy()` performs a `deepcopy` of the node list, then walks the
original and copied trees in parallel to restore `SourceNode.lazyframe_ref`
identity. This is necessary because `deepcopy` creates new LazyFrame
identity objects, which would break `dict[LazyFrame, DataFrame]` input
resolution.

---

## Persistence

`save()` pickles `{"nodes": self._nodes, "source": self._source}`.
`load()` reconstructs a LazyFrame from the pickled data.

Pipelines containing custom callables (lambdas, closures, locally defined
functions) are not portable across different Python environments because
pickle captures bytecode references.

---

## Key Patterns for Contributors

### Adding a New Semantic Operator

1. Create a new node class in `nodes.py` inheriting from `BaseNode`.
2. Implement `__call__`, `signature()`, and optionally `child_lfs()`.
3. Add the corresponding method to `LazyFrame` in `lazyframe.py`.
4. Export from `__init__.py`.
5. If the node has optimizable langex parameters, add it to
   `DEFAULT_OPTIMIZABLE_PARAMS` in `gepa_optimizer.py`.

### Adding a New Optimizer

1. Subclass `BaseOptimizer` in `optimizer/`.
2. Set `requires_train_data` appropriately.
3. Implement `optimize(nodes, train_data)` → `list[BaseNode]`.
4. Export from `optimizer/__init__.py`.

### Testing

- `tests/test_ast.py` — Core LazyFrame mechanics (copy, load_sem_index,
  multi-input concat).
- `tests/test_gepa_optimizer.py` — GEPA target collection, predicate
  pushdown integration.
- Mock external LLM calls; assert on pipeline structure rather than
  exact LLM outputs.
