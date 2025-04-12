from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models.utilities.model_family import ModelFamily

__tokenizer_directory = Path(__file__).parent


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    model_family = ModelFamily.infer_family(model_name)
    return AutoTokenizer.from_pretrained(__tokenizer_directory / model_family)
