import ollama
import logging
import re
from typing import Any, Dict, List, Optional
from langchain.llms.base import LLM

from components.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEESEEK_SYSTEM_PROMPT,
    MEDLLAMA_MODEL_NAME,
    MEDLLAMA_NUM_CTX,
    MEDLLAMA_NUM_THREAD,
    MEDLLAMA_SYSTEM_PROMPT,
    MEDLLAMA_TEMPERATURE,
    MEDLLAMA_TOP_P,
)

logger = logging.getLogger(__name__)


class BaseLLM(LLM):
    """Base class for Ollama LLM implementations with response filtering for deepseek :)"""

    model_name: str
    system_prompt: str
    temperature: float
    top_p: float

    def _filter_response(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        text = re.sub(
            r"<(think|thinking|thought)>.*?</(think|thinking|thought)>",
            "",
            text,
            flags=re.DOTALL,
        )

        text = re.sub(r"\n\s*\n", "\n", text)
        text = text.strip()

        return text

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        try:
            response: Dict[str, Any] = ollama.generate(
                model=self.model_name,
                system=self.system_prompt,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    **kwargs.get("options", {}),
                },
            )

            raw_response = response.get("response", "")
            filtered_response = self._filter_response(raw_response)

            return filtered_response or "I cannot provide an answer at this time."

        except Exception as e:
            logger.error(
                f"Error generating response in {self.__class__.__name__}: {str(e)}"
            )
            return "I cannot provide an answer at this time."


class OllamaLLM(BaseLLM):
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    system_prompt: str = DEESEEK_SYSTEM_PROMPT

    @property
    def _llm_type(self) -> str:
        return DEFAULT_MODEL_NAME


class LlamaMedLLM(BaseLLM):
    model_name: str = MEDLLAMA_MODEL_NAME
    temperature: float = MEDLLAMA_TEMPERATURE
    top_p: float = MEDLLAMA_TOP_P
    system_prompt: str = MEDLLAMA_SYSTEM_PROMPT

    @property
    def _llm_type(self) -> str:
        return MEDLLAMA_MODEL_NAME

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        return super()._call(
            prompt,
            stop,
            options={
                "num_ctx": MEDLLAMA_NUM_CTX,
                "num_thread": MEDLLAMA_NUM_THREAD,
                **kwargs.get("options", {}),
            },
        )
