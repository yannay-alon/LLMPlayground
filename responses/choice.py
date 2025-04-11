from pydantic import BaseModel
from enum import StrEnum
from typing import TypeVar
from responses.tool_call import ToolCall

ParsedType = TypeVar("ParsedType", bound=BaseModel)


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    NONE = "none"


class Choice(BaseModel):
    content: str
    finish_reason: FinishReason
    tool_calls: list[ToolCall] | None = None
    parsed: ParsedType | None = None
