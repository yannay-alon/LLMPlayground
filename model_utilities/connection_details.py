import os

from model_utilities.model_family import ModelFamily


class ConnectionDetails:
    @staticmethod
    def get_api_key(model_name: str, default_api_key: str = "default") -> str:
        model_family = ModelFamily.infer_family(model_name)

        api_key = os.getenv(f"{model_family.value}_API_KEY", default_api_key)
        return api_key

    @staticmethod
    def get_base_url(model_name: str, provider: str) -> str:
        model_family = ModelFamily.infer_family(model_name)

        base_url = os.getenv(f"{model_family.value}_{provider}_BASE_URL")
        return base_url
