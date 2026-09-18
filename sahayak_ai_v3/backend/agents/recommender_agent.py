import os
import logging
from typing import Any, Optional
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from sahayak_ai_v3.backend.agents.graph_state import GraphState

logger = logging.getLogger("sahayak.recommender_agent")

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

async def _extract_entities_via_llm(query: str) -> str:
    """Uses Groq to extract the core search entity from the user's prompt."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_retries=2)
    prompt = f"Extract the single most important noun, skill, or project from this query to search in a graph database. Return ONLY the exact entity string, nothing else. Query: '{query}'"
    try:
        res = await llm.ainvoke([{"role": "user", "content": prompt}])
        return str(res.content).strip().strip("'\"")
    except Exception:
        return ""

async def _execute_graph_query(entity: str) -> tuple[list[dict], list[dict]]:
    """Runs a parameterized 1-to-2 hop Cypher query and serializes the subgraph."""
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise ServiceUnavailable("Neo4j credentials missing.")

    nodes_map = {}
    links = []
    
    # Read-only query fetching up to 15 relationships to prevent memory bloat
    cypher_query = """
    MATCH (n)-[r]-(m)
    WHERE toLower(n.name) CONTAINS toLower($entity) 
       OR toLower(m.name) CONTAINS toLower($entity)
       OR toLower(type(r)) CONTAINS toLower($entity)
    RETURN n, r, m
    LIMIT 15
    """
    
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        async with driver.session() as session:
            result = await session.run(cypher_query, entity=entity)
            records = await result.data()
            
            for record in records:
                # Process Node n
                n = record.get("n")
                if n and getattr(n, "element_id", None):
                    nodes_map[n.element_id] = {
                        "id": n.element_id, 
                        "name": dict(n).get("name", "Unknown"), 
                        "group": list(n.labels)[0] if n.labels else "Node"
                    }
                    
                # Process Node m
                m = record.get("m")
                if m and getattr(m, "element_id", None):
                    nodes_map[m.element_id] = {
                        "id": m.element_id, 
                        "name": dict(m).get("name", "Unknown"), 
                        "group": list(m.labels)[0] if m.labels else "Node"
                    }
                    
                # Process Relationship r
                r = record.get("r")
                if r and getattr(r, "start_node", None) and getattr(r, "end_node", None):
                    links.append({
                        "source": r.start_node.element_id,
                        "target": r.end_node.element_id,
                        "label": r.type
                    })
    finally:
        await driver.close()
        
    return list(nodes_map.values()), links

async def recommender_agent(state: GraphState) -> dict[str, Any]:
    """LangGraph node: Graph DB lookup, state serialization, and recommendation synthesis."""
    query_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            query_text = str(msg.content)
            break

    if not query_text.strip():
        return {"messages": [AIMessage(content="I need more details to make a recommendation.")], "error": "EMPTY_QUERY"}

    graph_error: Optional[str] = None
    nodes = []
    links = []
    formatted_graph_context = ""

    try:
        entity = await _extract_entities_via_llm(query_text)
        if entity:
            nodes, links = await _execute_graph_query(entity)
            
        if nodes:
            context_lines = [f"Found {len(nodes)} related entities and {len(links)} connections."]
            for link in links:
                src = next((n["name"] for n in nodes if n["id"] == link["source"]), "Unknown")
                tgt = next((n["name"] for n in nodes if n["id"] == link["target"]), "Unknown")
                context_lines.append(f"- {src} [{link['label']}] {tgt}")
            formatted_graph_context = "\n".join(context_lines)
        else:
            formatted_graph_context = "No specific connections found in the graph database."
            
    except (Neo4jError, ServiceUnavailable) as exc:
        logger.error("Neo4j graph query failed: %s", exc)
        graph_error = "GRAPH_DB_UNAVAILABLE"
        formatted_graph_context = "Knowledge graph service is temporarily offline."
    except Exception as exc:
        logger.error("Unexpected recommender error: %s", exc)
        graph_error = f"RECOMMENDER_ERROR: {str(exc)}"

    # LLM Synthesis
    try:
        system_prompt = (
            "You are the Sahayak AI Recommender Agent. Provide actionable recommendations based ONLY "
            "on the provided graph database relationships. Format clearly using bullet points.\n\n"
            f"Graph Context:\n{formatted_graph_context}"
        )
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=2)
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text}
        ])
        final_answer = str(response.content)
    except Exception as exc:
        logger.error("Groq generation failed in recommender agent: %s", exc)
        final_answer = "I'm having trouble analyzing the recommendations right now. Please try again."
        graph_error = "LLM_GENERATION_FAILED"

    return {
        "messages": [AIMessage(content=final_answer)],
        "graph_data": {"nodes": nodes, "links": links} if nodes else None,
        "error": graph_error
    }
