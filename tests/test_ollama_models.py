import pytest
from unittest.mock import patch
from components.llm import BaseLLM, OllamaLLM, LlamaMedLLM
from components.config import (
    DEESEEK_SYSTEM_PROMPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MEDLLAMA_MODEL_NAME,
    MEDLLAMA_SYSTEM_PROMPT,
    MEDLLAMA_TEMPERATURE,
    MEDLLAMA_TOP_P,
)


@pytest.fixture
def base_llm():
    class TestLLM(BaseLLM):
        @property
        def _llm_type(self) -> str:
            return "test"

    return TestLLM(
        model_name="test-model", system_prompt="test prompt", temperature=0.7, top_p=0.9
    )


@pytest.fixture
def ollama_llm():
    return OllamaLLM()


@pytest.fixture
def llama_med_llm():
    return LlamaMedLLM()


class TestBaseLLM:
    def test_filter_response(self, base_llm):
        test_cases = [
            ("<think>internal thought</think>Hello", "Hello"),
            ("Normal text", "Normal text"),
            ("<thinking>process</thinking>Result", "Result"),
            ("Text\n\n\nwith\n\nextra\n\nlines", "Text\nwith\nextra\nlines"),
        ]

        for input_text, expected in test_cases:
            assert base_llm._filter_response(input_text) == expected

    @patch("ollama.generate")
    def test_call_success(self, mock_generate, base_llm):
        mock_generate.return_value = {"response": "Test response"}

        result = base_llm._call("Test prompt")

        assert result == "Test response"
        mock_generate.assert_called_once()

    @patch("ollama.generate")
    def test_call_error(self, mock_generate, base_llm):
        mock_generate.side_effect = Exception("Test error")

        result = base_llm._call("Test prompt")

        assert result == "I cannot provide an answer at this time."


class TestOllamaLLM:
    def test_default_values(self, ollama_llm):
        assert ollama_llm.model_name == DEFAULT_MODEL_NAME
        assert ollama_llm.temperature == DEFAULT_TEMPERATURE
        assert ollama_llm.top_p == DEFAULT_TOP_P
        assert ollama_llm.system_prompt == DEESEEK_SYSTEM_PROMPT

    def test_llm_type(self, ollama_llm):
        assert ollama_llm._llm_type == DEFAULT_MODEL_NAME


class TestLlamaMedLLM:
    def test_default_values(self, llama_med_llm):
        assert llama_med_llm.model_name == MEDLLAMA_MODEL_NAME
        assert llama_med_llm.temperature == MEDLLAMA_TEMPERATURE
        assert llama_med_llm.top_p == MEDLLAMA_TOP_P
        assert llama_med_llm.system_prompt == MEDLLAMA_SYSTEM_PROMPT

    def test_llm_type(self, llama_med_llm):
        assert llama_med_llm._llm_type == MEDLLAMA_MODEL_NAME
