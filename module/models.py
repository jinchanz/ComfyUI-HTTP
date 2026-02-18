"""
数据模型定义
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    modalities: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    extendParams: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(BaseModel):
    message: Dict[str, Any]


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: Optional[bool] = None
    message: Optional[str] = None
    request_id: Optional[str] = Field(default=None, alias="requestId")
    choices: Optional[List[ChatCompletionChoice]] = None
