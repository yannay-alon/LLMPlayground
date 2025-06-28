from typing import Any

from pydantic import BaseModel, field_validator, ValidationInfo, TypeAdapter, ValidationError

from components.tools import Tool


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
                argument_value = arguments_values[argument.name]
                try:
                    validated_argument = TypeAdapter(argument.annotation).validate_strings(argument_value)
                    validated_arguments[argument.name] = validated_argument
                except ValidationError:
                    raise TypeError(
                        f"Argument '{argument.name}' should be of type '{argument.annotation}', "
                        f"but got '{type(arguments_values[argument.name])}'"
                    )
        return validated_arguments
