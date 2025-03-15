from typing import Any

from openai import AsyncStream, Stream, Client, AsyncClient
from openai.types.chat import ChatCompletionToolParam, ChatCompletion, ChatCompletionChunk
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel

from documents import Document
from messages import BaseMessage, UserMessage
from models.api_model import APIModel
from tools import Tool


class OpenAIModel(APIModel):
    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
            strict_mode: bool = True,
            **client_arguments
    ):
        super().__init__(model_name, api_key, base_url, **client_arguments)
        self.strict_mode = strict_mode

        self.client = Client(api_key=api_key, base_url=base_url, **client_arguments)
        self.async_client = AsyncClient(api_key=api_key, base_url=base_url, **client_arguments)

    def _process_tools(
            self,
            tools: list[Tool] | None,
    ) -> list[ChatCompletionToolParam] | None:
        if tools is None:
            return None

        for tool in tools:
            ChatCompletionToolParam(
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters={
                        "type": "object",
                        "properties": {
                            parameter.name: {
                                "type": parameter.annotation,
                                "description": parameter.description,
                            } for parameter in tool.parameters
                        },
                        "required": [
                            parameter.name for parameter in tool.parameters if parameter.required or self.strict_mode
                        ],
                        "additionalProperties": False
                    },
                    strict=self.strict_mode,
                ),
                type="function",
            )

    def _add_documents_to_messages(
            self,
            messages: list[BaseMessage],
            documents: list[Document]
    ) -> list[BaseMessage]:
        formatted_documents = "\n\n".join(
            f"Document: {document_index}\n{document}"
            for document_index, document in enumerate(documents)
        )
        formatted_documents = f"Documents:\n{formatted_documents}"

        *history, last_message = messages

        messages_with_documents = history + [UserMessage(content=formatted_documents), last_message]
        return messages_with_documents

    def _invoke(
            self,
            messages: list[BaseMessage],
            stream: bool,
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None,
            max_tokens: int | None,
            temperature: float
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        dumped_messages, open_ai_compatible_tools = self._prepare_arguments(messages, tools, documents)

        return self.client.chat.completions.create(
            messages=dumped_messages,
            model=self.model_name,
            tools=open_ai_compatible_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    async def _async_invoke(
            self,
            messages: list[BaseMessage],
            stream: bool,
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None,
            max_tokens: int | None,
            temperature: float
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        dumped_messages, open_ai_compatible_tools = self._prepare_arguments(messages, tools, documents)

        return await self.async_client.chat.completions.create(
            messages=dumped_messages,
            model=self.model_name,
            tools=open_ai_compatible_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    def _prepare_arguments(
            self,
            messages: list[BaseMessage],
            tools: list[Tool] | None,
            documents: list[Document] | None
    ) -> tuple[list[dict[str, str]], list[ChatCompletionToolParam] | None]:
        messages_with_documents = self._add_documents_to_messages(messages, documents)
        dumped_messages = [message.model_dump(by_alias=True) for message in messages_with_documents]
        open_ai_compatible_tools = self._process_tools(tools)

        return dumped_messages, open_ai_compatible_tools

    def _process_arguments_for_prompt_creation(
            self,
            messages: list[BaseMessage],
            tools: list[Tool] | None,
            documents: list[Document] | None,
            response_format: type[BaseModel] | None
    ) -> tuple[list[dict[str, str]], list[dict[str, str]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
        dumped_messages, open_ai_compatible_tools = self._prepare_arguments(messages, tools, documents)
        additional_tokenization_arguments = {}
        return dumped_messages, open_ai_compatible_tools, None, additional_tokenization_arguments
