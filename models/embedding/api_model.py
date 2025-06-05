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

    @abstractmethod
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

    @abstractmethod
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
        pass
    # </editor-fold>
