# Medical RAG Chat System

A medical chat system that enables user interaction with an AI assistant capable of answering both medical and general queries. Depending on the type of query, the appropriate processing method is selected-either a QA chain with vector store search or a direct LLM invocation.

![streamlit](/docs/streamlit.gif)

## UML Diagram

![UML Screenshot](/docs/drawio.png)


## Main Components

### vector_store.py
- **Vector Store Initialization:** Utilizes Chroma and HuggingFaceEmbeddings to store and search medical documents.
- **Data Indexing:** The `index_data` function adds documents (texts and metadata) to the vector store, enabling the later retrieval of similar content.

### qa_chain.py
- **QA Chain:** Initializes a RetrievalQA chain using LlamaMedLLM as the model for medical queries.
- **Query Classification:** Uses a zero-shot model (e.g., `facebook/bart-large-mnli`) to determine whether a query is medical or general.
- **Query Processing:** The `process_query` function decides whether to handle a query with the QA chain (with vector store search) for medical queries or via a direct LLM call (OllamaLLM) for general queries.

### llm.py
- **Base LLM Definition:** The `BaseLLM` class contains common methods, including `_call` for communicating with the model via API (`ollama.generate`) and response filtering (removing unnecessary tags).
- **Model Implementations:**
  - **OllamaLLM:** For general queries, it uses the default model and prompt (`DEESEEK_SYSTEM_PROMPT`).
  - **LlamaMedLLM:** Tailored for medical queries with custom parameters (e.g., thread count, context) and prompt (`MEDLLAMA_SYSTEM_PROMPT`).

### data_loader.py
- **Medical Data Fetching:** Contains classes and functions to retrieve and parse articles from external sources such as PubMed and Drugs.com.
- **Web Scraping:** Uses libraries like requests, BeautifulSoup, and Selenium (for handling server restrictions) to fetch and process webpage content.

### config.py
- **Configuration:** Contains constants and settings for the entire project, including logging parameters, HTTP request settings, LLM model details, prompts, QA chain configuration, vector store settings, and the zero-shot model configuration.


### main.py
- **User Interface:** Uses Streamlit to build the chat interface, display message history, and accept user queries.
- **Query Processing Logic:** When a query is submitted, it is added to the session and passed to the `process_query` function, which decides whether to use the QA chain (for medical queries) or a direct LLM call (for general queries).

## Deployment and Initialization Setup

### entrypoint.sh
- **Testing & Code Quality:** Runs tests using `pytest`, formats code with Black, and checks code quality with Flake8.
- **Data Preparation:** Fetches and indexes medical data by calling the data loader before starting the Streamlit application.
- **Startup:** Exports necessary environment variables and launches the Streamlit app.

### Dockerfile
- **Base Image & Environment Setup:** Uses Python 3.12 as the base image, sets up the working directory, and installs required system packages.
- **Dependency Installation:** Installs Python dependencies from `requirements.txt` along with development tools (Black, Flake8, pytest, Selenium).
- **Pre-fetching Models:** Downloads necessary transformer models (e.g., from sentence-transformers and facebook/bart-large-mnli) during the build process.
- **Application Files & Permissions:** Copies application files, sets environment variables, exposes port 8501, and designates `entrypoint.sh` as the entry point.

### docker-compose.yml
- **Service Orchestration:** Defines and manages three services:
  - **ollama:**  
    - Runs the Ollama server, pulls required LLM models (`medllama2:latest` and `deepseek-r1:latest`), and exposes port 11434.
    - Includes a healthcheck and uses a persistent volume for model data.
  - **selenium:**  
    - Provides a Selenium Standalone Chrome container for web scraping tasks, exposing port 4444.
  - **app:**  
    - Builds and runs the main Medical RAG Chat System (Streamlit app).
    - Depends on the `ollama` and `selenium` services, configured with the appropriate environment variable (`OLLAMA_HOST`) and port exposure (8501).
- **Networking & Volumes:** Connects services using a custom bridge network and manages persistent storage for the `ollama` service.

## Running the Application

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Thizz00/Medical-RAG-Chat-System.git
   ```

2. **Build and Start Services with Docker Compose:**

```bash
docker-compose up --build
```

## Testing and Code Quality

1. **Run Tests:**

```bash
python -m pytest -s -v --durations=0
```
2. **Format and Lint Code:**

```bash
black .
flake8 --max-line-length=88 --ignore=E501,F841,W291,F401 .
```
