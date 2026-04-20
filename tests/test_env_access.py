from rag_core.config.env_access import parse_env_bool


def test_parse_env_bool() -> None:
    assert parse_env_bool('true') is True
    assert parse_env_bool('false') is False
    assert parse_env_bool('wat') is None
