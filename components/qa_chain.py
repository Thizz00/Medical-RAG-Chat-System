import logging
import streamlit as st
from typing import Dict, Any
from langchain.chains import RetrievalQA

from .llm import LlamaMedLLM, OllamaLLM
from .vectorstore import init_vectorstore
from transformers import pipeline
from components.config import (
    QA_SEARCH_TYPE,
    QA_SEARCH_K,
    ZERO_SHOT_LABELS,
    ZERO_SHOT_MODEL,
    ZERO_SHOT_THRESHOLD,
    DEFAULT_MODEL_NAME,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def init_qa_chain() -> RetrievalQA:
    logger.info("Initializing QA chain for medical queries")
    try:
        llm = LlamaMedLLM()
        vectorstore = init_vectorstore()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type=QA_SEARCH_TYPE, search_kwargs={"k": QA_SEARCH_K}
            ),
            return_source_documents=False,
            verbose=True,
        )

        logger.info("QA chain initialized successfully with LlamaMedLLM")
        return qa_chain

    except Exception as e:
        logger.error(f"Error initializing QA chain: {str(e)}")
        raise


@st.cache_resource(show_spinner=False)
def get_zero_shot_classifier():
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)


def is_medical_query(query: str, threshold: float = ZERO_SHOT_THRESHOLD) -> bool:

    classifier = get_zero_shot_classifier()
    candidate_labels = ZERO_SHOT_LABELS
    result = classifier(query, candidate_labels)
    if result["labels"][0] == ZERO_SHOT_LABELS[0] and result["scores"][0] >= threshold:
        return True
    return False


def process_query(
    qa_chain: Any, user_input: str, llm_instance: OllamaLLM
) -> Dict[str, Any]:

    logger.info(f"Processing question: {user_input}")

    if is_medical_query(user_input):
        logger.info("Query classified as medical. Using QA chain with LlamaMedLLM.")
        result = qa_chain.invoke({"query": user_input})
        answer = result["result"]
        sources = result.get("sources", []) or result.get("source_documents", [])
    else:
        logger.info(
            f"Query classified as general. Using direct call with OllamaLLM ({DEFAULT_MODEL_NAME})."
        )
        answer = llm_instance._call(prompt=user_input)
        sources = []

    return {"answer": answer, "sources": sources}
