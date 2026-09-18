import sys

path = 'backend/services/knowledge_graph.py'
with open(path, 'r') as f:
    code = f.read()

code = code.replace(
    'def add_entity(name: str, entity_type: str = "concept", description: str = "") -> int:',
    'def add_entity(name: str, entity_type: str = "concept", description: str = "", user_id: str | None = None) -> int:'
)
code = code.replace(
    '"INSERT OR IGNORE INTO entities (name, entity_type, description) VALUES (?, ?, ?)",\n            (name, entity_type, description),',
    '"INSERT OR IGNORE INTO entities (name, entity_type, description, user_id) VALUES (?, ?, ?, ?)",\n            (name, entity_type, description, user_id),'
)

code = code.replace(
    'def add_relationship(source: str, target: str, relation: str = "related_to", weight: float = 1.0) -> int:',
    'def add_relationship(source: str, target: str, relation: str = "related_to", weight: float = 1.0, user_id: str | None = None) -> int:'
)
code = code.replace(
    'add_entity(source)\n    add_entity(target)',
    'add_entity(source, user_id=user_id)\n    add_entity(target, user_id=user_id)'
)
code = code.replace(
    '"INSERT INTO relationships (source, target, relation, weight) VALUES (?, ?, ?, ?)",\n        (source, target, relation, weight),',
    '"INSERT INTO relationships (source, target, relation, weight, user_id) VALUES (?, ?, ?, ?, ?)",\n        (source, target, relation, weight, user_id),'
)

code = code.replace(
    'def get_graph(limit: int = 200) -> Dict[str, Any]:',
    'def get_graph(limit: int = 200, user_id: str | None = None) -> Dict[str, Any]:'
)
code = code.replace(
    'entities = conn.execute("SELECT name, entity_type, description FROM entities LIMIT ?", (limit,)).fetchall()',
    'entities = conn.execute("SELECT name, entity_type, description FROM entities WHERE user_id = ? OR ? IS NULL LIMIT ?", (user_id, user_id, limit,)).fetchall()'
)
code = code.replace(
    'relationships = conn.execute("SELECT source, target, relation, weight FROM relationships LIMIT ?", (limit * 3,)).fetchall()',
    'relationships = conn.execute("SELECT source, target, relation, weight FROM relationships WHERE user_id = ? OR ? IS NULL LIMIT ?", (user_id, user_id, limit * 3,)).fetchall()'
)

code = code.replace(
    'def get_entity(name: str) -> Optional[Dict[str, Any]]:',
    'def get_entity(name: str, user_id: str | None = None) -> Optional[Dict[str, Any]]:'
)
code = code.replace(
    'entity = conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()',
    'entity = conn.execute("SELECT * FROM entities WHERE name = ? AND (user_id = ? OR ? IS NULL)", (name, user_id, user_id)).fetchone()'
)
code = code.replace(
    'rels = conn.execute(\n        "SELECT * FROM relationships WHERE source = ? OR target = ?", (name, name)\n    ).fetchall()',
    'rels = conn.execute(\n        "SELECT * FROM relationships WHERE (source = ? OR target = ?) AND (user_id = ? OR ? IS NULL)", (name, name, user_id, user_id)\n    ).fetchall()'
)

code = code.replace(
    'def query_path(source: str, target: str) -> List[str]:',
    'def query_path(source: str, target: str, user_id: str | None = None) -> List[str]:'
)
code = code.replace(
    'edges = conn.execute("SELECT source, target FROM relationships").fetchall()',
    'edges = conn.execute("SELECT source, target FROM relationships WHERE user_id = ? OR ? IS NULL", (user_id, user_id)).fetchall()'
)

code = code.replace(
    'def extract_from_text(text: str) -> Dict[str, Any]:',
    'def extract_from_text(text: str, user_id: str | None = None) -> Dict[str, Any]:'
)
code = code.replace(
    'add_entity(entity, entity_type="named_entity")',
    'add_entity(entity, entity_type="named_entity", user_id=user_id)'
)
code = code.replace(
    'add_relationship(entities_found[i], entities_found[j], "co_occurs")',
    'add_relationship(entities_found[i], entities_found[j], "co_occurs", user_id=user_id)'
)
code = code.replace(
    'add_entity(word, entity_type="concept")',
    'add_entity(word, entity_type="concept", user_id=user_id)'
)

with open(path, 'w') as f:
    f.write(code)
print('Done!')
