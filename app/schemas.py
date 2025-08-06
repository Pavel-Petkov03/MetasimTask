from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    text: str
    memory: list[str]