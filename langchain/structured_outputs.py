from pydantic import BaseModel, Field
from typing import List

class GraphEntry(BaseModel):
    """A single relationship in the knowledge graph"""
    entity_1: str = Field(description="First entity in the relationship.")
    entity_2: str = Field(description="Second entity in the relationship.")
    edge: str = Field(description="The relationship between entity_1 and entity_2.")


class EntityEntry(BaseModel):
    """A single entity in the knowledge graph"""
    entity: str = Field(description="The entity.")
    classes: List[str] = Field(description="Classes of the entity.")


class KnowledgeGraph(BaseModel):
    """Graph representation of the statement and the question asked by the user"""

    graph: List[GraphEntry] = Field(
        description="Graph representation of the statement. "
                    "Each entry should define a relationship with 'entity_1', 'entity_2', and 'edge'."
    )

    entity_classes: List[EntityEntry] = Field(
        description="Classes of each entity in the graph. An entity can have multiple classes."
    )

class RelationshipResponse(BaseModel):
    """Response from the LLM indicating the relationship between two entities"""
    answer: str = Field(
        description="A single word from the valid list that represents the relationship between the first and last person in the sequence."
    )
    reason: str = Field(
        description="Step-by-step reasoning explaining how the answer was derived, showing the intermediate relationships from the first to last person using only valid terms."
    )

class RevisedResponse(BaseModel):
    """Response from the LLM indicating the relationship between two entities"""
    previous_reasoning_errors: str = Field(
        description="Errors found in the previous reasoning steps."
    )
    revised_answer: str = Field(
        description="A single word from the valid list that represents the relationship between the first and last person in the sequence."
    )
    reason: str = Field(
        description="Step-by-step reasoning explaining how the answer was derived, showing the intermediate relationships from the first to last person using only valid terms."
    )

class ConventionalResponse(BaseModel):
    """Response from the LLM indicating the relationship between two entities"""
    answer: str = Field(
        description="A single word answer"
    )
    reason: str = Field(
        description="reasoning"
    )