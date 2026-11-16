import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_import_lotus_does_not_load_torch():
    r = _run("import lotus, sys; sys.exit(0 if 'torch' not in sys.modules else 1)")
    assert r.returncode == 0, f"torch imported eagerly:\n{r.stderr}"


def test_import_lotus_does_not_load_sentence_transformers():
    r = _run("import lotus, sys; sys.exit(0 if 'sentence_transformers' not in sys.modules else 1)")
    assert r.returncode == 0, f"sentence_transformers imported eagerly:\n{r.stderr}"


def test_import_lotus_does_not_load_faiss():
    r = _run("import lotus, sys; sys.exit(0 if 'faiss' not in sys.modules else 1)")
    assert r.returncode == 0, f"faiss imported eagerly:\n{r.stderr}"


def test_lazy_imports_still_resolve():
    r = _run(
        "import lotus; "
        "from lotus.models import LM, RM, Reranker, SentenceTransformersRM, "
        "CrossEncoderReranker, LiteLLMRM, ColBERTv2RM; "
        "from lotus.vector_store import VS, FaissVS, WeaviateVS, QdrantVS; "
        "print('ok')"
    )
    assert r.returncode == 0, r.stderr


def test_public_api_unchanged():
    r = _run(
        "import lotus; "
        "assert set(lotus.models.__all__) == "
        "{'CrossEncoderReranker','LM','RM','Reranker','LiteLLMRM',"
        "'SentenceTransformersRM','ColBERTv2RM'}; "
        "assert set(lotus.vector_store.__all__) == {'VS','FaissVS','WeaviateVS','QdrantVS'}; "
        "print('ok')"
    )
    assert r.returncode == 0, r.stderr


def test_lazy_attribute_cached_after_first_access():
    r = _run(
        "import lotus.vector_store as m; "
        "cls = m.FaissVS; "  # triggers __getattr__ and caches in globals
        "assert 'FaissVS' in m.__dict__, 'cache not populated'; "
        "assert m.__dict__['FaissVS'] is cls; "
        "print('ok')"
    )
    assert r.returncode == 0, r.stderr


def test_unknown_attribute_raises_attributeerror():
    r = _run(
        "import lotus.models, lotus.vector_store\n"
        "for mod in (lotus.models, lotus.vector_store):\n"
        "    try:\n"
        "        mod.DoesNotExist\n"
        "        raise SystemExit('FAIL: no AttributeError')\n"
        "    except AttributeError:\n"
        "        pass\n"
        "print('ok')"
    )
    assert r.returncode == 0, r.stderr


def test_import_lotus_does_not_load_litellm():
    r = _run("import lotus, sys; sys.exit(0 if 'litellm' not in sys.modules else 1)")
    assert r.returncode == 0, f"litellm imported eagerly:\n{r.stderr}"
