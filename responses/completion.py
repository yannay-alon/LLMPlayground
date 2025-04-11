from pydantic import BaseModel
from responses.choice import Choice
from responses.usage import Usage


class Completion(BaseModel):
    choices: list[Choice]
    usage: Usage | None = None
