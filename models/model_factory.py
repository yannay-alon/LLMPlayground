import warnings

from models.api_model import APIModel
from models.openai_model import OpenAIModel
from models.cohere_model import CohereModel

from model_utilities import ModelFamily


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
        model_family = ModelFamily.infer_family(model_name)

        if model_family in [ModelFamily.COMMAND_A, ModelFamily.COMMAND_R]:
            return CohereModel(model_name, api_key, base_url, **kwargs)

        warnings.warn(
            f"Could not find a specific model class for {model_name}. Defaults to {cls.default_model_class.__name__}")
        return cls.default_model_class(model_name, api_key, base_url, **kwargs)
