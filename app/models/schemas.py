from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history, oldest first",
    )


class Assessment(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Assessment] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
