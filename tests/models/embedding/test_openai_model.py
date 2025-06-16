import httpx
import pytest
from openai import Client, AsyncClient

from models.embedding.openai_model import OpenAIModel


class TestOpenAIModel:
    @pytest.fixture
    def mock_client(self, mocker) -> Client:
        client = mocker.Mock()
        client.http_client = httpx.Client()
        mocker.patch("models.embedding.openai_model.Client", return_value=client)
        return client

    @pytest.fixture
    def mock_async_client(self, mocker) -> AsyncClient:
        client = mocker.AsyncMock()
        client.http_client = httpx.AsyncClient()
        mocker.patch("models.embedding.openai_model.AsyncClient", return_value=client)
        return client

    @pytest.fixture
    def model(self, mock_client: Client, mock_async_client: AsyncClient) -> OpenAIModel:
        return OpenAIModel(
            model_name="test-embedding-model",
            api_key="test-key",
            base_url="https://test.openai.com/v1"
        )

    def test_embed_single_text(self, model: OpenAIModel, mock_client: Client, mocker):
        test_text = "Test text"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_data = mocker.Mock()
        mock_data.embedding = mock_embedding
        mock_response = mocker.Mock()
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response

        result = model.embed(test_text)

        assert isinstance(result, list)
        assert result == mock_embedding
        mock_client.embeddings.create.assert_called_once_with(
            model=model.model_name,
            input=test_text,
            dimensions=None
        )

    def test_embed_multiple_texts(self, model, mock_client, mocker):
        test_texts = ["Text 1", "Text 2"]
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_data = []
        for embedding in mock_embeddings:
            mock_single_data = mocker.Mock()
            mock_single_data.embedding = embedding
            mock_data.append(mock_single_data)
        mock_response = mocker.Mock()
        mock_response.data = mock_data
        mock_client.embeddings.create.return_value = mock_response

        result = model.embed(test_texts)

        assert isinstance(result, list)
        assert all(isinstance(embedding, list) for embedding in result)
        assert result == mock_embeddings
        mock_client.embeddings.create.assert_called_once_with(
            model=model.model_name,
            input=test_texts,
            dimensions=None
        )

    @pytest.mark.asyncio
    async def test_async_embed_single_text(self, model: OpenAIModel, mock_async_client: AsyncClient, mocker):
        test_text = "Test text"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_data = mocker.Mock()
        mock_data.embedding = mock_embedding
        mock_response = mocker.Mock()
        mock_response.data = [mock_data]
        mock_async_client.embeddings.create.return_value = mock_response

        result = await model.async_embed(test_text)

        assert isinstance(result, list)
        assert result == mock_embedding
        mock_async_client.embeddings.create.assert_called_once_with(
            model=model.model_name,
            input=test_text,
            dimensions=None
        )

    @pytest.mark.asyncio
    async def test_async_embed_multiple_texts(self, model: OpenAIModel, mock_async_client: AsyncClient, mocker):
        test_texts = ["Text 1", "Text 2"]
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_data = []
        for embedding in mock_embeddings:
            mock_single_data = mocker.Mock()
            mock_single_data.embedding = embedding
            mock_data.append(mock_single_data)
        mock_response = mocker.Mock()
        mock_response.data = mock_data
        mock_async_client.embeddings.create.return_value = mock_response

        result = await model.async_embed(test_texts)

        assert isinstance(result, list)
        assert all(isinstance(embedding, list) for embedding in result)
        assert result == mock_embeddings
        mock_async_client.embeddings.create.assert_called_once_with(
            model=model.model_name,
            input=test_texts,
            dimensions=None
        )
