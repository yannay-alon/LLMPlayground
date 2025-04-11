from typing import Any

from pydantic import BaseModel, field_validator, ValidationInfo

from tools import Tool


class ToolCall(BaseModel):
    identifier: str
    tool: Tool
    arguments_values: dict[str, Any]

    def __call__(self):
        return self.tool(**self.arguments_values)

    @field_validator("arguments_values")
    @classmethod
    def validate_arguments(cls, arguments_values: dict[str, Any], validation_info: ValidationInfo) -> dict[str, Any]:
        tool = validation_info.data["tool"]
        validated_arguments = {}
        for argument in tool.arguments:
            if argument.required and argument.name not in arguments_values:
                raise ValueError(f"Missing required argument: {argument.name}")
            elif argument.name in arguments_values:
                validated_arguments[argument.name] = arguments_values[argument.name]
                if argument.annotation == Any:
                    continue
                if not isinstance(arguments_values[argument.name], argument.annotation):
                    raise TypeError(
                        f"Argument '{argument.name}' should be of type '{argument.annotation}', "
                        f"but got '{type(arguments_values[argument.name])}'"
                    )
        return validated_arguments
