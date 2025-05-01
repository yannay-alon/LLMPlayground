import uuid
import warnings
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


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
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["tool"] = "tool"
    identifier: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="id")


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
