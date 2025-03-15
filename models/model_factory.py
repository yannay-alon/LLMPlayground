import warnings

from models.api_model import APIModel
from models.openai_model import OpenAIModel
from models.cohere_model import CohereModel

from model_utilities import ModelFamily, ConnectionDetails


class ModelFactory:
    default_model_class = OpenAIModel

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

        model_family = ModelFamily.infer_family(model_name)

        if model_family in [ModelFamily.COMMAND_A, ModelFamily.COMMAND_R]:
            return CohereModel(model_name, api_key, base_url, **kwargs)

        warnings.warn(
            f"Could not find a specific model class for {model_name}. Defaults to {cls.default_model_class.__name__}")
        return cls.default_model_class(model_name, api_key, base_url, **kwargs)
