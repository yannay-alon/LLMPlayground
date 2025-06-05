from typing import Any

import httpx
from openai import Client, AsyncClient
from openai.types import CreateEmbeddingResponse

from models.embedding.api_model import APIModel


class OpenAIModel(APIModel):

    def __init__(
            self,
            model_name: str,
            api_key: str | None = None,
            base_url: str | None = None,
            sync_client_arguments: dict[str, Any] | None = None,
            async_client_arguments: dict[str, Any] | None = None,
    ):
        super().__init__(model_name, api_key, base_url)

        sync_client_arguments = sync_client_arguments or dict(
            http_client=httpx.Client()
        )
        self.client = Client(
            api_key=api_key,
            base_url=base_url,
            **sync_client_arguments
        )

        async_client_arguments = async_client_arguments or dict(
            http_client=httpx.AsyncClient()
        )
        self.async_client = AsyncClient(
            api_key=api_key,
            base_url=base_url,
            **async_client_arguments
        )

    def _embed(
            self,
            text: str | list[str],
            output_dimensions: int | None = None
    ) -> list[float] | list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=output_dimensions
        )

        is_batched = not isinstance(text, str)
        return self._extract_embedding(response, is_batched)

    async def _async_embed(
            self,
            text: str | list[str],
            output_dimensions: int | None = None
    ) -> list[float] | list[list[float]]:
        response = await self.async_client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=output_dimensions
        )

        is_batched = not isinstance(text, str)
        return self._extract_embedding(response, is_batched)

    @staticmethod
    def _extract_embedding(
            response: CreateEmbeddingResponse,
            is_batched: bool
    ) -> list[float] | list[list[float]]:
        embedded_text = []
        for embedding in response.data:
            embedded_text.append(embedding.embedding)

        if not is_batched:
            return embedded_text[0]
        return embedded_text
