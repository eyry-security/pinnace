"""Session store round-trips through JSONL."""

from langchain_core.messages import HumanMessage, SystemMessage

from pinnace.session import SessionStore


def test_save_load_roundtrip(tmp_path):
    store = SessionStore(tmp_path)
    msgs = [SystemMessage(content="sys"), HumanMessage(content="hello")]
    path = store.save("s1", msgs)
    assert path.exists()
    back = store.load("s1")
    assert [type(m).__name__ for m in back] == ["SystemMessage", "HumanMessage"]
    assert back[1].content == "hello"


def test_load_missing_returns_none(tmp_path):
    assert SessionStore(tmp_path).load("nope") is None


def test_list_and_delete(tmp_path):
    store = SessionStore(tmp_path)
    store.save("a", [HumanMessage(content="x")])
    store.save("b", [HumanMessage(content="y")])
    assert store.list() == ["a", "b"]
    assert store.delete("a") is True
    assert store.list() == ["b"]
    assert store.delete("a") is False


def test_session_id_sanitized(tmp_path):
    store = SessionStore(tmp_path)
    store.save("../../evil", [HumanMessage(content="x")])
    # Must land inside the sessions root, never outside it.
    inside = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(inside) == 1 and inside[0].name.replace("_", "")
    assert not (tmp_path.parent / "evil").exists()
