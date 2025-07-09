from collections import defaultdict
from typing import AsyncIterable, Iterable, Any

from ollama import (
    ChatResponse,
    AsyncClient,
    Client,
    Tool as OllamaTool,
    Message as OllamaMessage,
    ResponseError,
)
from pydantic import BaseModel
from tqdm import tqdm

from components.documents import Document
from components.messages import BaseMessage
from components.responses import Completion, ToolCall, Usage
from components.responses.choice import ParsedType, Choice, FinishReason
from components.tools import Tool
from models.generation.language_model import LanguageModel
from utilities.pydantic_utilities import clear_empty_fields

OllamaFunction = OllamaTool.Function
OllamaParameters = OllamaFunction.Parameters
OllamaProperty = OllamaParameters.Property


class OllamaCompatibleArguments(BaseModel):
    messages: list[OllamaMessage]
    tools: list[OllamaTool] | None = None
    format: dict[str, Any] | None = None


class OllamaModel(LanguageModel):
    def __init__(
            self,
            model_name: str,
    ):
        super().__init__(model_name=model_name)

        self.client = Client()
        self.async_client = AsyncClient()

        self._ensure_model_available()

    def _ensure_model_available(self):
        try:
            self.client.show(model=self.model_name)
        except ResponseError as error:
            if error.status_code == 404:
                self._download_model(self.model_name)
            else:
                raise

    def _download_model(self, model_name: str):
        print(f"Model {self.model_name} is not downloaded. Attempting to pull...")
        progress_bar = None
        try:
            for status in self.client.pull(model_name, stream=True):
                if "total" in status and "completed" in status:
                    if progress_bar is None:
                        progress_bar = tqdm(
                            total=status["total"],
                            unit="B", unit_scale=True,
                            desc=f"Downloading {model_name}"
                        )
                    progress_bar.update(status["completed"] - progress_bar.n)
            print(f"Successfully pulled model {self.model_name}")
        except ResponseError as error:
            raise RuntimeError(f"Failed to pull model {self.model_name}: {str(error)}") from error
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def delete(self):
        self.client.delete(model=self.model_name)

    def _invoke(self, messages: list[BaseMessage], stream: bool, tools: dict[str, Tool] | None,
                documents: list[Document] | None, response_format: type[ParsedType] | None, max_tokens: int | None,
                temperature: float) -> Completion[ParsedType] | Iterable[Completion[ParsedType]]:
        arguments = self._prepare_arguments(
            messages=messages,
            tools=tools,
            documents=documents,
            response_format=response_format,
        )
        non_empty_arguments = clear_empty_fields(arguments)
        response = self.client.chat(
            model=self.model_name,
            stream=stream,
            **non_empty_arguments
        )
        if stream:
            def streaming_generator() -> Iterable[Completion]:
                for chunk in response:
                    stream_choices = [
                        self._build_chioce(
                            chunk,
                            response_format=None,  # No support for structured output in streaming mode
                            tools=tools,
                        )
                    ]
                    yield Completion(choices=stream_choices, usage=None)

            return streaming_generator()
        else:
            return Completion(
                choices=[
                    self._build_chioce(
                        response,
                        response_format=response_format,
                        tools=tools
                    )
                ],
                usage=Usage(
                    input_tokens=response.prompt_eval_count,
                    output_tokens=response.eval_count,
                )
            )

    async def _async_invoke(self, messages: list[BaseMessage], stream: bool, tools: dict[str, Tool] | None,
                            documents: list[Document] | None, response_format: type[ParsedType] | None,
                            max_tokens: int | None, temperature: float
                            ) -> Completion[ParsedType] | AsyncIterable[Completion[ParsedType]]:
        arguments = self._prepare_arguments(
            messages=messages,
            tools=tools,
            documents=documents,
            response_format=response_format,
        )
        non_empty_arguments = clear_empty_fields(arguments)
        response = await self.async_client.chat(
            model=self.model_name,
            stream=stream,
            **non_empty_arguments
        )
        if stream:
            async def streaming_generator() -> AsyncIterable[Completion]:
                async for chunk in response:
                    stream_choices = [
                        self._build_chioce(
                            chunk,
                            response_format=None,  # No support for structured output in streaming mode
                            tools=tools,
                        )
                    ]
                    yield Completion(choices=stream_choices, usage=None)

            return streaming_generator()
        else:
            return Completion(
                choices=[
                    self._build_chioce(
                        response,
                        response_format=response_format,
                        tools=tools
                    )
                ],
                usage=Usage(
                    input_tokens=response.prompt_eval_count,
                    output_tokens=response.eval_count,
                )
            )

    def _build_chioce(
            self,
            response: ChatResponse,
            response_format: type[ParsedType] | None,
            tools: dict[str, Tool] | None = None,
    ) -> Choice[ParsedType]:
        if tools is None:
            tools = {}
        tool_mapping = defaultdict()
        for tool_name, tool in tools.items():
            tool_mapping[tool_name] = tool

        content = response.message.content
        if content is None:
            content = response.message.thinking

        return Choice(
            content=content,
            finish_reason=FinishReason(response.done_reason) if response.done_reason is not None else None,
            parsed=response_format.model_validate_json(response.message.content) if response_format else None,
            tool_calls=[
                ToolCall(
                    identifier=str(tool_call_index),
                    tool=tool_mapping.get(tool_call.function.name),
                    arguments_values=tool_call.function.arguments,
                )
                for tool_call_index, tool_call in enumerate(response.message.tool_calls or [])
            ]
        )

    def _prepare_arguments(
            self,
            messages: list[BaseMessage],
            tools: dict[str, Tool] | None = None,
            documents: list[Document] | None = None,
            response_format: type[ParsedType] | None = None,
    ) -> OllamaCompatibleArguments:
        messages_with_documents = self._add_documents_to_messages(messages, documents)
        dumped_messages = [
            OllamaMessage(
                role=message.role,
                content=message.content,
            )
            for message in messages_with_documents
        ]
        ollama_compatible_tools = self._process_tools(tools)
        ollama_compatible_response_format = self._process_response_format(response_format)

        return OllamaCompatibleArguments(
            messages=dumped_messages,
            tools=ollama_compatible_tools,
            format=ollama_compatible_response_format
        )

    @staticmethod
    def _add_documents_to_messages(
            messages: list[BaseMessage],
            documents: list[Document] | None
    ) -> list[BaseMessage]:
        if documents is None:
            return messages

        formatted_documents = "\n\n".join(
            f"Document: {document_index}\n{document}"
            for document_index, document in enumerate(documents)
        )
        formatted_documents = f"Documents:\n{formatted_documents}"

        *history, last_message = messages

        last_message.content += f"\n\n{formatted_documents}"

        messages_with_documents = history.append(last_message)
        return messages_with_documents

    def _process_tools(
            self,
            tools: dict[str, Tool] | None,
    ) -> list[OllamaTool] | None:
        if tools is None:
            return None

        ollama_compatible_tools = []
        for tool_name, tool in tools.items():
            dumped_tool = tool.model_dump(by_alias=True)
            tool_defs = dumped_tool.get("$defs", None)
            tool_items = dumped_tool.get("items", None)

            properties = dict()
            required_arguments = []
            for argument in tool.arguments:
                dumped_argument = argument.model_dump(by_alias=True)
                properties[argument.name] = OllamaProperty(
                    type=dumped_argument.get("type", None),
                    items=dumped_argument.get("items", None),
                    description=dumped_argument.get("description", ""),
                    enum=dumped_argument.get("enum", None),
                )
                if argument.required:
                    required_arguments.append(argument.name)

            ollama_compatible_tools.append(
                OllamaTool(
                    function=OllamaFunction(
                        name=tool_name,
                        description=tool.description,
                        parameters=OllamaParameters(
                            defs=tool_defs,
                            items=tool_items,
                            required=required_arguments,
                            properties=properties,
                        ),
                    )
                )
            )
        return ollama_compatible_tools

    def _process_response_format(
            self,
            response_format: type[ParsedType] | None,
    ) -> dict[str, Any] | None:
        if response_format is None:
            return None

        return response_format.model_json_schema(by_alias=True)
