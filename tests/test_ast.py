import pandas as pd

from lotus.ast import LazyFrame
from lotus.ast.nodes import LoadSemIndexNode, PandasOpNode, SourceNode


def test_lazyframe_load_sem_index_appends_node():
    lf = LazyFrame()
    updated = lf.load_sem_index("text", "my_index")

    assert len(updated) == len(lf) + 1
    assert isinstance(updated._nodes[-1], LoadSemIndexNode)
    assert "load_sem_index('text', 'my_index')" in updated.show()


def test_lazyframe_load_sem_index_executes_accessor():
    df = pd.DataFrame({"text": ["alpha", "beta"]})

    result = LazyFrame().load_sem_index("text", "my_index").execute(df)

    assert result.attrs["index_dirs"]["text"] == "my_index"


def test_lazyframe_copy_preserves_source_binding_with_multi_input_dict():
    left_ref = object()
    right_ref = object()
    other_ref = object()

    left_nodes = [
        SourceNode(lazyframe_ref=left_ref),
        PandasOpNode(op_name="__getitem__", args=("a",)),
    ]
    right_nodes = [
        SourceNode(lazyframe_ref=right_ref),
        PandasOpNode(op_name="__getitem__", args=("b",)),
    ]
    left_lf = LazyFrame(_nodes=left_nodes, _source=left_nodes[0])
    right_lf = LazyFrame(_nodes=right_nodes, _source=right_nodes[0])
    combined = LazyFrame.concat([left_lf, right_lf], axis=1)

    copied = combined.copy()
    result = copied.execute(
        {
            left_ref: pd.DataFrame({"a": [1, 2]}),
            right_ref: pd.DataFrame({"b": [3, 4]}),
            other_ref: pd.DataFrame({"z": [99]}),
        }
    )
    assert list(result["a"]) == [1, 2]
    assert list(result["b"]) == [3, 4]
