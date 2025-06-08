import os
from dotenv import load_dotenv

from models.utilities.model_family import ModelFamily

load_dotenv()


class ConnectionDetails:
    @staticmethod
    def get_api_key(model_name: str, provider: str | None = None, default_api_key: str = "default") -> str:
        if provider is not None:
            prefix = ConnectionDetails._normalize_provider(provider)
        else:
            model_family = ModelFamily.infer_family(model_name)
            prefix = ConnectionDetails._normalize_model_family(model_family)

        api_key = os.getenv(f"{prefix}_API_KEY", default_api_key)
        return api_key

    @staticmethod
    def get_base_url(model_name: str, provider: str | None = None) -> str:
        if provider is not None:
            prefix = ConnectionDetails._normalize_provider(provider)
        else:
            model_family = ModelFamily.infer_family(model_name)
            prefix = ConnectionDetails._normalize_model_family(model_family)

        base_url = os.getenv(f"{prefix}_BASE_URL")
        return base_url

    @staticmethod
    def _normalize_model_family(model_family: ModelFamily) -> str:
        """
        Normalize the model family string to a standard format.
        """
        return model_family.upper().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        """
        Normalize the provider string to a standard format.
        """
        return provider.upper()
