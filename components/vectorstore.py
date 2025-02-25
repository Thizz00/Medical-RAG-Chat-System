import logging
from typing import Dict, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
from components.config import (
    VECTORSTORE_CACHE_FOLDER,
    VECTORSTORE_COLLECTION_NAME,
    VECTORSTORE_MODEL_NAME,
    VECTORSTORE_PERSIST_DIR,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def init_vectorstore() -> Chroma:
    logger.info("Initializing vector store")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=VECTORSTORE_MODEL_NAME, cache_folder=VECTORSTORE_CACHE_FOLDER
        )

        logger.info("Clearing existing data in the vector store...")
        Chroma(
            collection_name=VECTORSTORE_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTORSTORE_PERSIST_DIR,
        ).delete_collection()

        logger.info("Vector store initialized successfully and data cleared")

        vectorstore = Chroma(
            collection_name=VECTORSTORE_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTORSTORE_PERSIST_DIR,
        )

        return vectorstore

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise


def index_data(vectorstore: Chroma, documents: List[Dict[str, Dict[str, str]]]) -> None:
    logger.info("Indexing data into vector store")
    try:
        texts: List[str] = [doc["text"] for doc in documents]
        metadatas: List[Dict[str, str]] = [doc["metadata"] for doc in documents]

        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        logger.info(f"Successfully indexed {len(texts)} documents")

    except Exception as e:
        logger.error(f"Error indexing data: {str(e)}")
        raise
