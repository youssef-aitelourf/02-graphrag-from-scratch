from graphrag_mvp.ollama_client import EXTRACTION_JSON_SCHEMA


def test_schema_is_object_with_lists():
    assert EXTRACTION_JSON_SCHEMA["type"] == "object"
    assert "entities" in EXTRACTION_JSON_SCHEMA["properties"]
