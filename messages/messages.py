from __future__ import annotations

from typing import Literal
import warnings

from pydantic import BaseModel


class BaseMessage(BaseModel):
    role: str
    content: str


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"


class ToolMessage(BaseMessage):
    role: Literal["tool"] = "tool"


class MessageFactory:
    @staticmethod
    def create_message(role: str, content: str) -> BaseMessage:
        match role:
            case "user":
                return UserMessage(content=content)
            case "system":
                return SystemMessage(content=content)
            case "assistant":
                return AssistantMessage(content=content)
            case "tool":
                return ToolMessage(content=content)
            case _:
                warnings.warn(f"Could not find a specific type for message with role {role}")
                return BaseMessage(role=role, content=content)
