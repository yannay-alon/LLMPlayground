from pydantic import BaseModel
from components.responses.choice import Choice
from components.responses.usage import Usage


class Completion(BaseModel):
    choices: list[Choice]
    usage: Usage | None = None
