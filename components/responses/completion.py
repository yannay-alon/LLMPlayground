from typing import Generic

from pydantic import BaseModel
from components.responses.choice import Choice, ParsedType
from components.responses.usage import Usage


class Completion(BaseModel, Generic[ParsedType]):
    choices: list[Choice[ParsedType]]
    usage: Usage | None = None
