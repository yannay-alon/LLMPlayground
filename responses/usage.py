from pydantic import BaseModel, computed_field


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
