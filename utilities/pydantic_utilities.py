from typing import get_args, get_origin

from pydantic import BaseModel, create_model, ConfigDict


def make_strict_model(model: type[BaseModel]) -> type[BaseModel]:
    if not issubclass(model, BaseModel):
        return model

    fields = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        default = field.default if field.default is not None else ...

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            annotation = make_strict_model(annotation)

        origin = get_origin(annotation)
        if origin:
            args = get_args(annotation)
            new_args = tuple(
                make_strict_model(arg) if isinstance(arg, type) and issubclass(arg, BaseModel) else arg for arg in args)
            annotation = origin[new_args]

        fields[name] = (annotation, default)

    return create_model(
        model.__name__,
        __config__=ConfigDict(**model.model_config | dict(extra="forbid")),
        __module__=model.__module__,
        **fields
    )
