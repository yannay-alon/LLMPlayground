from abc import ABC, abstractmethod
from typing import overload


class APIModel(ABC):
    def __init__(self, model_name: str, api_key: str | None = None, base_url: str | None = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    # <editor-fold desc="Synchronous">
    @overload
    def embed(
            self,
            text: str,
            output_dimensions: int | None = None,
    ) -> list[float]:
        ...

    @overload
    def embed(
            self,
            text: list[str],
            output_dimensions: int | None = None,
    ) -> list[list[float]]:
        ...

    def embed(
            self,
            text: str | list[str],
            output_dimensions: int | None = None,
    ) -> list[float] | list[list[float]]:
        """
        Embed the given text using the model.

        :param text: The text to embed.
        :param output_dimensions: The desired output dimensions of the embedding.
        :return: The embedding of the text.
        """
        return self._embed(text, output_dimensions)

    @abstractmethod
    def _embed(
            self,
            text: str | list[str],
            output_dimensions: int | None = None,
    ) -> list[float] | list[list[float]]:
        pass

    # </editor-fold>

    # <editor-fold desc="Asynchronous">
    @overload
    async def async_embed(
            self,
            text: str,
            output_dimensions: int | None = None,
    ) -> list[float]:
        ...

    @overload
    async def async_embed(
            self,
            text: list[str],
            output_dimensions: int | None = None,
    ) -> list[list[float]]:
        ...

    async def async_embed(
            self,
            text: str | list[str],
            output_dimensions: int | None = None,
    ) -> list[float] | list[list[float]]:
        """
        Asynchronously embed the given text using the model.

        :param text: The text to embed.
        :param output_dimensions: The desired output dimensions of the embedding.
        :return: The embedding of the text.
        """
        return await self._async_embed(text, output_dimensions)

    @abstractmethod
    async def _async_embed(
            self,
            text: str | list[str],
            output_dimensions: int | None = None,
    ) -> list[float] | list[list[float]]:
        pass
    # </editor-fold>
