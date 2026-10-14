from cachellm.normalize import parse_json_object


def test_parse_json_object_preserves_unicode_and_empty_values():
    assert parse_json_object('{"city": "Montréal", "query": "", "tags": []}') == {
        "city": "Montréal",
        "query": "",
        "tags": [],
    }


def test_parse_json_object_returns_none_for_malformed_tool_arguments():
    assert parse_json_object('{"city":') is None
