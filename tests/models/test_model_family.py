import pytest

from models.utilities import ModelFamily


class TestModelFamily:
    def test_model_family_inference(self):
        assert ModelFamily.infer_family("command-a-test") == ModelFamily.COMMAND_A
        assert ModelFamily.infer_family("command-r-model") == ModelFamily.COMMAND_R
        assert ModelFamily.infer_family("llama-7b") == ModelFamily.LLAMA

    def test_model_family_inference_case_insensitive(self):
        assert ModelFamily.infer_family("COMMAND-A-TEST") == ModelFamily.COMMAND_A
        assert ModelFamily.infer_family("Command-R-Model") == ModelFamily.COMMAND_R
        assert ModelFamily.infer_family("LLaMA-7b") == ModelFamily.LLAMA

    def test_model_family_inference_invalid(self):
        with pytest.raises(ValueError):
            ModelFamily.infer_family("unknown-model")
