import os

from dotenv import load_dotenv

from models.utilities.model_family import Provider

load_dotenv()


class ConnectionDetails:
    @staticmethod
    def get_api_key(model_name: str, provider: Provider | None = None, default_api_key: str = "default") -> str | None:
        if provider is None:
            provider = Provider.infer_provider(model_name)
        prefix = ConnectionDetails._normalize_provider(provider)

        api_key = os.getenv(f"{prefix}_API_KEY", default_api_key)
        return api_key

    @staticmethod
    def get_base_url(model_name: str, provider: Provider | None = None) -> str | None:
        if provider is None:
            provider = Provider.infer_provider(model_name)
        prefix = ConnectionDetails._normalize_provider(provider)

        base_url = os.getenv(f"{prefix}_BASE_URL")
        return base_url

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        """
        Normalize the provider string to a standard format.
        """
        return provider.upper()
