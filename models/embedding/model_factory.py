from models.embedding.api_model import APIModel
from models.embedding.openai_model import OpenAIModel

from models.utilities import ConnectionDetails


class ModelFactory:

    @classmethod
    def get_model(
            cls,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
            **kwargs
    ) -> APIModel:
        if api_key is None:
            api_key = ConnectionDetails.get_api_key(model_name)
        if base_url is None:
            base_url = ConnectionDetails.get_base_url(model_name)

        return OpenAIModel(model_name, api_key, base_url, **kwargs)
