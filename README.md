<div align="center">

<img src="docs/logo_with_text.png" width="360" alt="LOTUS logo"/>

# LOTUS: Optimized Agentic and LLM Bulk Processing

**Bulk process your datasets with agents and LLMs at scale, with higher accuracy and lower cost.**

*From Stanford University and UC Berkeley*

<!--- BADGES: START --->
[![PyPI](https://img.shields.io/pypi/v/lotus-ai)][#pypi-package]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lotus-ai)][#pypi-package]
[![Arxiv](https://img.shields.io/badge/arXiv-2407.11418-B31B1B.svg)][#arxiv-paper-package]
[![Documentation Status](https://readthedocs.org/projects/lotus-ai/badge/?version=latest)](https://lotus-ai.readthedocs.io/en/latest/?badge=latest)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)][#discord]
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mP65YHHdD6mnZmC5-Uqm2uCXJ4-Kbkhu?usp=sharing)
<!--- BADGES: END --->

[**What is LOTUS?**](#what-is-lotus) • [**Install**](#installation) • [**Quickstart**](#quickstart) • [**Semantic Operators**](#what-are-semantic-operators) • [**Community**](#community) • [**Docs**](https://lotus-ai.readthedocs.io/en/latest/) • [**Cite**](#references)

</div>

---

## What is LOTUS?

LOTUS makes **agentic and LLM bulk processing** fast, easy, and robust. It introduces and optimizes
[**semantic operators**](https://arxiv.org/abs/2407.11418) (e.g., LLM-based `map`,
`reduce`, `filter` primitives) to let you process your large datasets with LLMs and natural language instructions. LOTUS **optimizes** these operations to help you get **higher accuracy and lower cost**.

**What you can build:**

- **Agentic code processing** — run a tool-using agent (with a sandboxed Python REPL) over
  every file, document, or record, then reduce to one answer (codebase analysis, security
  sweeps, migrations).
- **Deep research & synthesis** — fan out over a corpus, extract, and synthesize.
- **Agent-trace failure analysis** — mine large volumes of agent logs for failure modes.
- **Document extraction & unstructured analysis** — structured fields and insights from text.
- **LLM-judge evals & RAG** — declarative pipelines that the engine optimizes for you.

## Installation

```bash
pip install lotus-ai
```

Or with [uv](https://docs.astral.sh/uv/): `uv add lotus-ai`. For the latest features, install
from source: `pip install git+https://github.com/lotus-data/lotus.git@main`.

> On macOS with pip, install FAISS via conda: `conda install -c pytorch faiss-cpu=1.8.0`
> (uv handles this automatically). See the [docs](https://lotus-ai.readthedocs.io/en/latest/installation.html)
> for GPU and troubleshooting details.

## Quickstart

Give LOTUS a **corpus** and a **task**. It allows you to run parallel agent and LLM calls. Below we show an example of an agentic map reduce, allowing you to bulk process code files by automatically sharding the corpus, spawning an agent per shard **in parallel** (each with a sandboxed Python REPL), and reducing the
results into one answer.

```python
import lotus
from lotus.models import LM
from lotus.tools import PythonREPLTool

# Configure the LM — export your API key first (e.g. OPENAI_API_KEY)
lotus.settings.configure(lm=LM(model="gpt-4o-mini"))

# A corpus can be files, documents, DataFrame rows, or one large text
corpus = lotus.Corpus.from_files("myproject/**/*.py")

# One task. LOTUS derives the map + reduce, runs agents in parallel, and aggregates.
result = corpus.agentic_map_reduce(
    task="For each file, find security-sensitive code and summarize the risks with "
         "file:line. Then produce one prioritized report across the codebase.",
    tools=[PythonREPLTool()],
)

print(result.output)     # the reduced report
print(result.findings)   # per-file findings
print(result.usage)      # token usage
```

See the [Agentic Map-Reduce docs](https://lotus-ai.readthedocs.io/en/latest/agentic_map_reduce.html)
and [`examples/agentic_map_reduce/`](examples/agentic_map_reduce) for more.

## How it works

You express *what* you want over a **corpus** using high-level **semantic operators** (i.e., LLM-based map, reduce, filter); LOTUS' **optimizer** decides *how* to run it — batching calls, applying model
cascades and proxies, and lazily planning the whole pipeline — for higher accuracy at lower
cost.

<div align="center">

<img src="assets/how_it_works.svg" width="1000" alt="LOTUS pipeline: Corpus → Declarative Programming → LOTUS Optimizer → Results"/>

</div>

**The Results**: Across diverse tasks, LOTUS' optimized pipelines match or exceed the accuracy of
high-quality baselines while running substantially faster and cheaper:

<div align="center">

<img src="assets/results_summary.png" width="720" alt="Results of LOTUS optimized pipelines"/>

</div>

## What are Semantic Operators

LOTUS introduced and optimizes [semantic operators](https://arxiv.org/abs/2407.11418).
Each operator implements an LLM-based transformation over your dataset, which you specify
with a natural language instruction, and the operations can be transparently optimized.
Here are a few examples:

<div align="center">

<img src="assets/semantic_operators.svg" width="960" alt="Semantic operators: sem_map, sem_filter, sem_agg (reduce), and sem_join, each showing docs flowing through an LM to an output"/>

</div>

See the
[documentation](https://lotus-ai.readthedocs.io/en/latest/) and the
[intro Colab tutorial](https://colab.research.google.com/drive/1mP65YHHdD6mnZmC5-Uqm2uCXJ4-Kbkhu?usp=sharing) for more on semantic operators that LOTUS serves.

## Docs

Full documentation at [lotus-ai.readthedocs.io](https://lotus-ai.readthedocs.io/en/latest/).
Key sections:

- [Installation](https://lotus-ai.readthedocs.io/en/latest/installation.html) & [Core Concepts](https://lotus-ai.readthedocs.io/en/latest/core_concepts.html)
- [Agentic Map-Reduce](https://lotus-ai.readthedocs.io/en/latest/agentic_map_reduce.html) — corpus, tools/REPL, and worked [examples](https://lotus-ai.readthedocs.io/en/latest/agentic_examples.html)
- [Semantic Operators](https://lotus-ai.readthedocs.io/en/latest/sem_map.html) — `sem_map`, `sem_filter`, `sem_agg`, `sem_join`, and more
- [Lazy Execution & Optimizations](https://lotus-ai.readthedocs.io/en/latest/lazyframe.html) — the query engine, cascades, and lazy execution
- [Supported Models](https://lotus-ai.readthedocs.io/en/latest/llm.html) — LMs, retrievers, rerankers (any [LiteLLM](https://litellm.vercel.app) provider)

## Community
Join us on [Discord][#discord] to ask questions and share what you're building.

Check out these awesome projects that are building with LOTUS:

- **[MAP: Measuring Agents in Production](https://arxiv.org/abs/2512.04123)** — a large-scale
  empirical study of deployed LLM agent systems across many domains (Stanford, Berkeley, and
  collaborators).
- **[VibeCheck](https://github.com/lisadunlap/VibeCheck)** — discovers and quantifies
  qualitative differences between LLMs (UC Berkeley; ICLR 2025).
- **[DeepScholar](https://deep-scholar.vercel.app/)** — generative research synthesis over
  the scientific literature (Stanford & Berkeley).

Using LOTUS in your project? Reach out to @semantic_operators on discord if you'd like it featured.

## Contributing

Contributions welcome! Read the [Contributing Guide](CONTRIBUTING.md) and check existing
issues before opening a PR. For trouble-shooting or feature requests, open an issue and
we'll get to it promptly.

## References

Follow [@lianapatel_](https://x.com/lianapatel_) on X for updates. If you find LOTUS or
semantic operators useful, please cite:

```bibtex
@article{patel2025semanticoptimization,
    title = {Semantic Operators and Their Optimization: Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS},
    author = {Patel, Liana and Jha, Siddharth and Pan, Melissa and Gupta, Harshit and Asawa, Parth and Guestrin, Carlos and Zaharia, Matei},
    year = {2025},
    journal = {Proc. VLDB Endow.},
    url = {https://doi.org/10.14778/3749646.3749685},
}
@article{patel2024semanticoperators,
    title={Semantic Operators: A Declarative Model for Rich, AI-based Analytics Over Text Data},
    author={Liana Patel and Siddharth Jha and Parth Asawa and Melissa Pan and Carlos Guestrin and Matei Zaharia},
    year={2024},
    eprint={2407.11418},
    url={https://arxiv.org/abs/2407.11418},
}
@article{patel2026ainative,
    title = {Towards AI-Native Data Systems with the Semantic Operator Model and LOTUS},
    author = {Patel, Liana and Guestrin, Carlos and Zaharia, Matei},
    year = {2026},
    journal = {IEEE Data Engineering Bulletin},
    url = {http://sites.computer.org/debull/A26mar/A26MAR-CD.pdf#page=61},
}
```

[#arxiv-paper-package]: https://arxiv.org/abs/2407.11418
[#pypi-package]: https://pypi.org/project/lotus-ai/
[#discord]: https://discord.gg/ZWQBurm5bt
