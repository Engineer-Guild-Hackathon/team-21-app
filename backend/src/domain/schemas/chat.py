from typing import Dict, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None


class ChatResponse(BaseModel):
    message: str
    emotion: Dict[str, float]
    action: Dict[str, str]


class ChatHistory(BaseModel):
    user_message: str
    ai_response: str
    emotion: Dict[str, float]
    timestamp: str
