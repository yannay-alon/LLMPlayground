from pydantic import BaseModel, TypeAdapter
from typing import TypeVar

OutputType = TypeVar("OutputType", bound=BaseModel)


def clean_json_string(json_string: str) -> str:
    cleaned_json = json_string.strip("`").strip("json").strip()
    return cleaned_json


def parse_json(json_string: str, output_type: type[OutputType]) -> OutputType:
    cleaned_json = clean_json_string(json_string)
    return output_type.model_validate_json(cleaned_json)


def parse_json_array(json_string: str, output_type: type[OutputType]) -> list[OutputType]:
    cleaned_json = clean_json_string(json_string)
    multiple_outputs_type = TypeAdapter(list[output_type])
    return multiple_outputs_type.validate_json(cleaned_json)
