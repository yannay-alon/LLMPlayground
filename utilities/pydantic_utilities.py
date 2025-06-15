from typing import Any

from pydantic import BaseModel


def make_strict_model(model: type[BaseModel]) -> dict:
    json_schema = model.model_json_schema(by_alias=True)
    return _ensure_strict_json_schema(json_schema)


def _ensure_strict_json_schema(
        json_schema: dict[str, Any],
        *,
        path: tuple[str, ...] | None = None,
        root: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if path is None:
        path = tuple()
    if root is None:
        root = json_schema

    defs = json_schema.get("$defs")
    if isinstance(defs, dict):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = json_schema.get("definitions")
    if isinstance(definitions, dict):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(definition_schema, path=(*path, "definitions", definition_name), root=root)

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    properties = json_schema.get("properties")
    if isinstance(properties, dict):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # arrays
    items = json_schema.get("items")
    if isinstance(items, dict):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    # unions
    any_of = json_schema.get("anyOf")
    if isinstance(any_of, dict):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) == 1:
            json_schema.update(_ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root))
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]

    if "default" in json_schema:
        json_schema.pop("default")

    json_reference = json_schema.get("$ref")
    if json_reference and len(json_schema.keys()) > 1:
        assert isinstance(json_reference, str), f"Received non-string $ref - {json_reference}"

        resolved = resolve_reference(root=root, reference=json_reference)
        if not isinstance(resolved, dict):
            raise ValueError(f"Expected `$ref: {json_reference}` to resolved to a dictionary but got {resolved}")

        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def resolve_reference(*, root: dict[str, object], reference: str) -> object:
    reference_prefix = "#/"
    if not reference.startswith(reference_prefix):
        raise ValueError(f"Unexpected $ref format {reference!r}; Does not start with {reference_prefix}")

    path = reference[len(reference_prefix):].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert isinstance(value, dict), f"encountered non-dictionary entry while resolving {reference} - {resolved}"
        resolved = value

    return resolved
