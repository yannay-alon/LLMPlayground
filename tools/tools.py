import inspect
from copy import deepcopy
from typing import Self, Callable, ParamSpec, TypeVar, cast, Any, Generic

import griffe
from pydantic import BaseModel, computed_field, TypeAdapter

from tools.docstring_style import infer_docstring_style

ToolInput = ParamSpec("ToolInput")
ToolOutput = TypeVar("ToolOutput")


class Argument(BaseModel):
    name: str
    description: str
    annotation: type
    required: bool = True

    @computed_field
    @property
    def type(self) -> str:
        return TypeAdapter(self.annotation).json_schema().get("type", "string")


class Tool(BaseModel, Generic[ToolInput, ToolOutput]):
    name: str
    description: str
    arguments: list[Argument]
    function: Callable[ToolInput, ToolOutput]

    def __init__(self, *args, **kwargs):
        # Check whether this is used as a decorator
        function = None
        if len(args) == 0 and len(kwargs) == 1 and "function" in kwargs:
            function = kwargs.pop("function")
        elif len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], (Tool, Callable)):
            function = args[0]

        if function is not None:
            tool = self.from_function(function)
            super().__init__(
                name=tool.name,
                description=tool.description,
                arguments=tool.arguments,
                function=tool.function,
            )
        else:
            super().__init__(**kwargs)

    @classmethod
    def from_function(cls, function: Callable[ToolInput, ToolOutput] | "Tool") -> Self:
        if isinstance(function, Tool):
            return deepcopy(function)

        signature = inspect.signature(function)
        description, arguments_descriptions = documentation_descriptions(function, signature)

        arguments = []
        for parameter in signature.parameters.values():
            argument_description = arguments_descriptions.get(parameter.name, "")
            argument_annotation = parameter.annotation if parameter.annotation != inspect.Parameter.empty else Any
            arguments.append(
                Argument(
                    name=parameter.name,
                    description=argument_description,
                    annotation=argument_annotation,
                    required=(parameter.default == inspect.Parameter.empty)
                )
            )

        return cls(
            name=function.__name__,
            description=description,
            arguments=arguments,
            function=function,
        )

    def __call__(self, *args: ToolInput.args, **kwargs: ToolInput.kwargs) -> ToolOutput:
        return self.function(*args, **kwargs)


def documentation_descriptions(
        function: Callable[ToolInput, ToolOutput],
        signature: inspect.Signature,
) -> tuple[str, dict[str, str]]:
    main_description_default = ""
    parameters_descriptions_default = {}

    documentation = function.__doc__
    if documentation is None:
        return main_description_default, parameters_descriptions_default

    parent = cast(griffe.Object, signature)

    docstring_style = infer_docstring_style(documentation)
    docstring = griffe.Docstring(documentation, lineno=1, parser=docstring_style, parent=parent)
    sections = docstring.parse()

    if parameters := next((
            parameter for parameter in sections
            if parameter.kind == griffe.DocstringSectionKind.parameters),
            parameters_descriptions_default
    ):
        parameters = {parameter.name: parameter.description for parameter in parameters.value}

    main_description = main_description_default
    if main := next((p for p in sections if p.kind == griffe.DocstringSectionKind.text), None):
        main_description = main.value

    return main_description, parameters
