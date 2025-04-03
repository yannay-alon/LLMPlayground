from __future__ import annotations

from enum import StrEnum


class ModelFamily(StrEnum):
    COMMAND_A = "command-a"
    COMMAND_R = "command-r"
    LLAMA = "llama"

    @staticmethod
    def infer_family(model_name: str) -> ModelFamily:
        for model_family in ModelFamily:
            if model_family in model_name.lower():
                return ModelFamily(model_family)

        raise ValueError(f"Model family could not be inferred from model name: {model_name}")
