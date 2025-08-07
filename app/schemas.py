from enum import StrEnum
from pydantic import BaseModel




class MemoryOwnerEnum(StrEnum):
    AI = "ai"
    HUMAN = "human"

class MemoryEntry(BaseModel):
    text : str
    role : MemoryOwnerEnum

class TextRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    current_message: str
    memory: list[MemoryEntry]
    product : str