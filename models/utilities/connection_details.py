import os
from dotenv import load_dotenv

from models.utilities.model_family import ModelFamily

load_dotenv()


class ConnectionDetails:
    @staticmethod
    def get_api_key(model_name: str, default_api_key: str = "default") -> str:
        model_family = ModelFamily.infer_family(model_name)

        api_key = os.getenv(f"{ConnectionDetails._normalize_model_family(model_family)}_API_KEY", default_api_key)
        return api_key

    @staticmethod
    def get_base_url(model_name: str, provider: str = "default") -> str:
        model_family = ModelFamily.infer_family(model_name)

        base_url = os.getenv(f"{ConnectionDetails._normalize_model_family(model_family)}_{provider}_BASE_URL")
        return base_url

    @staticmethod
    def _normalize_model_family(model_family: ModelFamily) -> str:
        """
        Normalize the model family string to a standard format.
        """
        return model_family.upper().replace("-", "_").replace(" ", "_")
