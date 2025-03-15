from pydantic import BaseModel, Field, ConfigDict, model_serializer


class Document(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    identifier: str = Field(alias="id")
    title: str
    text: str

    @model_serializer
    def serialize_model(self) -> dict[str, str | dict[str, str]]:
        return {
            "id": self.identifier,
            "data": self._serialize_data()
        }

    def _serialize_data(self) -> dict[str, str]:
        return {"title": self.title, "text": self.text}

    def __str__(self):
        return f"{self.title}\n{self.text}"
