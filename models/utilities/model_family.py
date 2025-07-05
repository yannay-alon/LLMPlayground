from __future__ import annotations

from enum import StrEnum


class ModelFamily(StrEnum):
    COMMAND_A = "command-a"
    COMMAND_R = "command-r"
    GPT = "gpt"
    LLAMA = "llama"

    @staticmethod
    def infer_family(model_name: str) -> ModelFamily:
        for model_family in ModelFamily:
            if model_family in model_name.lower():
                return ModelFamily(model_family)

        raise ValueError(f"Model family could not be inferred from model name: {model_name}")


class Provider(StrEnum):
    OPENAI = "openai"
    COHERE = "cohere"
    META = "meta"
    OLLAMA = "ollama"

    @staticmethod
    def infer_provider(model_name: str) -> Provider:
        model_family = ModelFamily.infer_family(model_name)

        match model_family:
            case ModelFamily.GPT:
                return Provider.OPENAI
            case ModelFamily.COMMAND_A | ModelFamily.COMMAND_R:
                return Provider.COHERE
            case ModelFamily.LLAMA:
                return Provider.META
            case _:
                return Provider.OLLAMA
