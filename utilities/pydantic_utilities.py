from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode
from pydantic_core import CoreSchema

SCHEMA_TO_RECURSE = TypeVar("SCHEMA_TO_RECURSE")


def clear_empty_fields(
            model: BaseModel, **kwargs: Any
    ) -> dict[str, Any]:
    """
    Get key-value pairs of all non-empty fields in a Pydantic model and any additional fields.
    
    :param model: The Pydantic model instance from which to extract fields.
    :param kwargs: Any additional fields to include in the output.
    :return: A dictionary containing only the non-empty fields of the model and additional fields.
    """
    optional_arguments = {
        **{
            key: value for key in model.model_fields
            if (value := getattr(model, key)) is not None
        },
        **kwargs
    }
    non_empty_arguments = {
        key: value for key, value in optional_arguments.items() if value is not None
    }
    return non_empty_arguments

def make_strict_model(model: type[BaseModel]) -> dict:
    json_schema = model.model_json_schema(schema_generator=CompatibilitySchemaGenerator)
    return json_schema


class CompatibilitySchemaGenerator(GenerateJsonSchema):
    """
    Custom schema generator that:
        1. Inlines all references for LLM tools (resolves refs/defs).
        2. Reorders keys for LLM optimization.
        3. Removes redundant titles

    Adapted from [tooldantic](https://github.com/nicholishen/tooldantic)
    """

    key_order = (
        "name",
        "title",
        "type",
        "description",
        "strict",
        "format",
        "enum",
        "properties",
        "required",
        "items",
        "additionalProperties",
    )

    is_reordered_keys = True
    is_removed_titles = False
    is_inlined_refs = True
    is_apply_strict_rules = True

    def generate(self, schema: CoreSchema, mode: JsonSchemaMode = "validation") -> dict[str, Any]:
        json_schema = super().generate(schema, mode)
        title = json_schema.get("title")
        if self.is_apply_strict_rules:
            json_schema = self._ensure_strict_json_schema(json_schema, path=())
        if self.is_inlined_refs and "$defs" in json_schema:
            definitions = json_schema.pop("$defs")
            json_schema = self._inline_references(json_schema, definitions)
            json_schema = self._inline_all_of(json_schema)
        if self.is_reordered_keys:
            json_schema = self._reorder_keys(json_schema)
        if self.is_removed_titles:
            top_level_title = json_schema.pop("title", title)
            json_schema = self._remove_target_key(json_schema, "title")
            # This will get popped downstream and used as the name
            json_schema["title"] = top_level_title
        return json_schema

    def _inline_references(
            self,
            schema: dict[str, Any],
            definitions: dict[str, Any],
            visited: set | None = None,
            parent: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if visited is None:
            visited = set()

        if isinstance(schema, dict):
            for key, value in list(schema.items()):
                if key == "$ref":
                    ref_key = value.split("/")[-1]
                    if ref_key in visited:
                        # Directly reference the definition to indicate recursion simply.
                        schema[key] = "#"
                        continue
                    visited.add(ref_key)
                    if ref_key in definitions:
                        schema.update(definitions[ref_key])
                        schema.pop("$ref")
                    # Pass the current schema as parent to detect direct recursions
                    self._inline_references(schema, definitions, visited, schema)
                else:
                    schema[key] = self._inline_references(
                        value, definitions, visited, parent
                    )
        elif isinstance(schema, list):
            schema = [
                self._inline_references(item, definitions, visited, parent)
                for item in schema
            ]
        return schema

    def _reorder_keys(self, schema: SCHEMA_TO_RECURSE) -> SCHEMA_TO_RECURSE:
        if not isinstance(schema, dict):
            return schema
        ordered_dict = {key: schema.pop(key) for key in self.key_order if key in schema}
        # Add remaining keys
        ordered_dict.update({key: self._reorder_keys(values) for key, values in schema.items()})
        return {key: self._reorder_keys(value) for key, value in ordered_dict.items()}

    def _remove_target_key(self, schema: SCHEMA_TO_RECURSE, target_key: str = "title") -> SCHEMA_TO_RECURSE:
        if isinstance(schema, dict):
            new_dict = {}
            for key, value in schema.items():
                if key == target_key and isinstance(value, str):
                    continue  # Skip string titles
                new_dict[key] = self._remove_target_key(value, target_key)
            return new_dict
        elif isinstance(schema, list):
            return [self._remove_target_key(item, target_key) for item in schema]
        return schema

    def _inline_all_of(self, schema: SCHEMA_TO_RECURSE) -> SCHEMA_TO_RECURSE:
        """Inlines allOf schemas if the allOf list contains only one item."""
        if isinstance(schema, dict):
            if "allOf" in schema and len(schema["allOf"]) == 1:
                # Replace the allOf construct with its single contained schema
                inlined_schema = self._inline_all_of(schema["allOf"][0])
                # If the inlined schema is a dictionary, merge it with the current schema
                if isinstance(inlined_schema, dict):
                    schema.update(inlined_schema)
                    schema.pop("allOf")
                return schema
            # Recursively apply this method to all dictionary values
            for key, value in schema.items():
                schema[key] = self._inline_all_of(value)
        elif isinstance(schema, list):
            # Recursively apply this method to all items in the list
            return [self._inline_all_of(item) for item in schema]
        return schema

    def _ensure_strict_json_schema(
            self,
            json_schema: dict[str, Any] | list[Any],
            path: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Mutates the given JSON schema to ensure it conforms to the `strict` standard
        that the API expects.
        """
        if not isinstance(json_schema, dict):
            raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

        property_type = json_schema.get("type")
        if property_type == "object" and "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False

        # object types
        # { 'type': 'object', 'properties': { 'a':  {...} } }
        properties = json_schema.get("properties")
        if isinstance(properties, dict):
            json_schema["required"] = [prop for prop in properties.keys()]
            json_schema["properties"] = {
                key: self._ensure_strict_json_schema(
                    property_schema, path=(*path, "properties", key)
                )
                for key, property_schema in properties.items()
            }
        # arrays
        # { 'type': 'array', 'items': {...} }
        items = json_schema.get("items")
        if isinstance(items, list):
            json_schema["items"] = self._ensure_strict_json_schema(
                items, path=(*path, "items")
            )

        for operator_key in ["anyOf", "allOf"]:
            operator_value = json_schema.get(operator_key)
            if isinstance(operator_value, list):
                json_schema[operator_key] = [
                    self._ensure_strict_json_schema(
                        entry, path=(*path, operator_key, str(index))
                    )
                    for index, entry in enumerate(operator_value)
                ]

        for definition_key in ["$defs", "definitions"]:
            definitions = json_schema.get(definition_key)
            if isinstance(definitions, dict):
                for name, definition_schema in definitions.items():
                    self._ensure_strict_json_schema(
                        definition_schema, path=(*path, definition_key, name)
                    )
        return json_schema
