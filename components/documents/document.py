from pydantic import BaseModel, Field, ConfigDict


class Document(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    identifier: str | None = Field(default=None, alias="id")
    content: str

    def __str__(self) -> str:
        return self.content
