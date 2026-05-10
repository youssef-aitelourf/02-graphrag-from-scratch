from graphrag_mvp.config import Settings
from graphrag_mvp.graph_store import build_artifacts, expand_entities


def test_build_and_leiden():
    extractions = [
        {
            "chunk_id": "c1",
            "entities": [
                {
                    "name": "Alice",
                    "entity_type": "person",
                    "description": "Engineer",
                },
                {
                    "name": "Bob",
                    "entity_type": "person",
                    "description": "Manager",
                },
            ],
            "relations": [
                {
                    "source": "Alice",
                    "target": "Bob",
                    "relation_type": "reports_to",
                    "description": "org",
                },
            ],
        },
        {
            "chunk_id": "c2",
            "entities": [
                {
                    "name": "Bob",
                    "entity_type": "person",
                    "description": "Manager",
                },
                {
                    "name": "Contoso",
                    "entity_type": "org",
                    "description": "Company",
                },
            ],
            "relations": [
                {
                    "source": "Bob",
                    "target": "Contoso",
                    "relation_type": "works_at",
                    "description": "employment",
                },
            ],
        },
    ]
    settings = Settings.from_env()
    art = build_artifacts(settings, extractions)
    assert art.graph.number_of_nodes() >= 3
    assert art.partition
    seeds = art.chunk_to_entities["c1"]
    expanded = expand_entities(seeds, art.graph, max_depth=2)
    assert len(expanded) >= len(seeds)
