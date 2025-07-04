from abc import ABC

from models.generation.language_model import LanguageModel


class APIModel(LanguageModel, ABC):
    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
    ):
        super().__init__(model_name=model_name)
        self.api_key = api_key
        self.base_url = base_url
