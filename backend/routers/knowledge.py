"""Knowledge graph router — entity and relationship management."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services import knowledge_graph as kg

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class EntityRequest(BaseModel):
    name: str
    entity_type: str = "concept"
    description: str = ""


class RelationshipRequest(BaseModel):
    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0


class PathQuery(BaseModel):
    source: str
    target: str


class ExtractRequest(BaseModel):
    text: str


@router.get("/graph")
def get_graph(limit: int = 200):
    """Get the knowledge graph (nodes + edges)."""
    return kg.get_graph(limit=limit)


@router.get("/entity/{name}")
def get_entity(name: str):
    """Get an entity with its relationships."""
    result = kg.get_entity(name)
    if not result:
        return {"error": "Entity not found"}
    return result


@router.post("/entity")
def add_entity(req: EntityRequest):
    """Add a new entity to the knowledge graph."""
    entity_id = kg.add_entity(req.name, req.entity_type, req.description)
    return {"id": entity_id, "name": req.name}


@router.post("/relationship")
def add_relationship(req: RelationshipRequest):
    """Add a relationship between two entities."""
    rel_id = kg.add_relationship(req.source, req.target, req.relation, req.weight)
    return {"id": rel_id}


@router.post("/path")
def find_path(req: PathQuery):
    """Find a path between two entities."""
    path = kg.query_path(req.source, req.target)
    return {"path": path, "found": len(path) > 0}


@router.post("/extract")
def extract_from_text(req: ExtractRequest):
    """Extract entities and relationships from text."""
    return kg.extract_from_text(req.text)
