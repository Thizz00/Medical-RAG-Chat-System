import logging

import streamlit as st

from components.qa_chain import init_qa_chain, process_query
from components.llm import OllamaLLM
from components.config import LOGGING_FORMAT, LOGGING_LEVEL

logging.basicConfig(
    level=LOGGING_LEVEL, format=LOGGING_FORMAT, handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

st.markdown(
    """
  <style>
  div.stSpinner > div {
    text-align:center;
    align-items: center;
    justify-content: center;
  }
  </style>""",
    unsafe_allow_html=True,
)


def display_response(answer: str) -> None:

    st.chat_message("ai").write(answer)


def main() -> None:
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Medical RAG Chat System</h1>",
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    user_input: str = st.chat_input("Your question:")

    if user_input:
        st.chat_message("human").write(user_input)
        st.session_state.messages.append({"role": "human", "content": user_input})

        qa_chain = init_qa_chain()
        llm_instance = OllamaLLM()
        with st.spinner("Generating response..."):
            try:
                response = process_query(qa_chain, user_input, llm_instance)
                display_response(response["answer"])
                st.session_state.messages.append(
                    {"role": "ai", "content": response["answer"]}
                )
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error("Error generating response")


if __name__ == "__main__":
    logger.info("Starting application")
    main()
