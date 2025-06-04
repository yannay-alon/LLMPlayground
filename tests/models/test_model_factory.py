from typing import Any

from models import ModelFactory


class TestModelFactory:
    def test_model_creation(self, mocker: Any) -> None:
        mock_connection_details = mocker.patch("models.utilities.ConnectionDetails")
        mock_connection_details.get_api_key.return_value = "default-key"
        mock_connection_details.get_base_url.return_value = "default-url"

        model = ModelFactory.get_model("command-a-test")

        assert isinstance(model, ModelFactory.default_model_class)
        assert model.model_name == "command-a-test"
        assert model.api_key == "default-key"
        assert model.base_url == "default-url"

    def test_model_creation_with_custom_credentials(self) -> None:
        model = ModelFactory.get_model(
            "command-a-test",
            api_key="custom-key",
            base_url="custom-url"
        )

        assert model.api_key == "custom-key"
        assert model.base_url == "custom-url"
