## Adding new LLMs

To add a new LLM that is based on API calls, you need to inherit from `APIModel` and implement the following methods:

- For LLM calling:
  - _`invoke`
  - _`async_invoke`
- For prompt creation (mostly for token usage tracking):
  - _`process_arguments_for_prompt_creation`

Note that `_invoke` and `async_invoke` get the same inputs and should handle both streaming and non-streaming options.
Their output should be OpenAI-compatible as defined by the methods signatures.
Your model should be able to handle documents and tools as inputs.
You can take a look in the `openai_model.py` file for an example of how to implement these methods.

For the prompt creation, we utilize HuggingFace's `transformers` library, specifically the `AutoTokenizer` class.
To allow for automatic tokenization, you should:
1. Add a directory under `model_utilities/tokenizer` with the name of your model's family.
2. Add a `tokenizer_config.json`, `special_tokens_map.json`, and `tokenizer.json` file to the directory.
3. Add your model's family to the `ModelFamily` enum in `model_utilities/model_family.py`.

