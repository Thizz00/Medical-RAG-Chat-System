import pytest
from unittest.mock import Mock, patch
from components.vectorstore import init_vectorstore
from components.config import (
    VECTORSTORE_CACHE_FOLDER,
    VECTORSTORE_MODEL_NAME,
    VECTORSTORE_COLLECTION_NAME,
    VECTORSTORE_PERSIST_DIR,
)


@pytest.fixture
def mock_embeddings():
    with patch("components.vectorstore.HuggingFaceEmbeddings") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_chroma():
    with patch("components.vectorstore.Chroma") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def sample_documents():
    return [
        {"text": "Sample text 1", "metadata": {"source": "doc1", "page": "1"}},
        {"text": "Sample text 2", "metadata": {"source": "doc2", "page": "2"}},
    ]


class TestVectorstore:
    def test_init_vectorstore_success(self, mock_embeddings, mock_chroma):
        mock_chroma_instance = mock_chroma.return_value

        result = init_vectorstore()

        mock_embeddings.assert_called_once_with(
            model_name=VECTORSTORE_MODEL_NAME, cache_folder=VECTORSTORE_CACHE_FOLDER
        )

        mock_chroma.assert_any_call(
            collection_name=VECTORSTORE_COLLECTION_NAME,
            embedding_function=mock_embeddings.return_value,
            persist_directory=VECTORSTORE_PERSIST_DIR,
        )

        assert mock_chroma.call_count == 2
        assert result == mock_chroma_instance
