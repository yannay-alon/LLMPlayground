## Adding new LLMs

To add a new LLM based on API calls, you need to inherit from `APIModel` and implement the following methods:

- For LLM calling:
    - `_invoke`
    - `_async_invoke`
- For prompt creation (mostly for token usage tracking):
    - `_process_arguments_for_prompt_creation`

Note that `_invoke` and `async_invoke` get the same inputs and should handle both streaming and non-streaming options.
Your model should be able to handle documents, tools and structured-output as inputs.
You can take a look in the `openai_model.py` file for an example of how to implement these methods.

<u>Optional</u>:<br>
For the prompt creation, we use HuggingFace's `transformers` library, specifically the `AutoTokenizer` class.
To allow for automatic tokenization, you should:

1. Add a directory under `models/utilities/tokenizer` with the name of your model's family.
2. Add a `tokenizer_config.json`, a `special_tokens_map.json`, and a `tokenizer.json` file to the directory.
3. Add your model's family to the `ModelFamily` enum in `modeld/utilities/model_family.py`.

If you do not add the tokenizer, the `create_prompt` method will not be available.

It is recommended to use a `NamedTuple` to define the LLM-compatible arguments. <br>
This allows for an intermediate step between the provider-agnostic arguments and the provider-specific arguments. <br>
This will help in creating a clean interface for the LLM and make it easier to manage the arguments.
