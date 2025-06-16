import warnings

from models.generation.api_model import APIModel
from models.generation.openai_model import OpenAIModel
from models.generation.cohere_model import CohereModel

from models.utilities import ModelFamily, ConnectionDetails


class ModelFactory:
    default_model_class = OpenAIModel

    @classmethod
    def get_model(
            cls,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
            provider: str | None = None,
            *,
            silent: bool = True,
            **kwargs,
    ) -> APIModel:
        if api_key is None:
            api_key = ConnectionDetails.get_api_key(model_name, provider=provider)
        if base_url is None:
            base_url = ConnectionDetails.get_base_url(model_name, provider=provider)

        model_family = ModelFamily.infer_family(model_name)

        match model_family:
            case ModelFamily.GPT:
                return OpenAIModel(model_name, api_key, base_url, **kwargs)
            case ModelFamily.COMMAND_A | ModelFamily.COMMAND_R:
                return CohereModel(model_name, api_key, base_url)
            case _:
                if not silent:
                    warnings.warn(
                        f"Could not find a specific model class for {model_name}. "
                        f"Defaults to {cls.default_model_class.__name__}"
                    )
                return cls.default_model_class(model_name, api_key, base_url, **kwargs)
