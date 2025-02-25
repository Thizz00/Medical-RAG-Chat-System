import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_ollama_response():
    return {"response": "Test response"}


@pytest.fixture
def mock_vectorstore():
    return Mock()


@pytest.fixture
def sample_documents():
    return [
        {"text": "Sample text 1", "metadata": {"source": "doc1", "page": "1"}},
        {"text": "Sample text 2", "metadata": {"source": "doc2", "page": "2"}},
    ]
