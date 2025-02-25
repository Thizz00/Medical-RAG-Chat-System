import logging

# Logger configuration
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# HTTP Request configuration
USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

REQUEST_TIMEOUT = 30

# PubMed API configuration
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_PARAMS = {"db": "pubmed", "term": "all[sb]", "retmax": 250, "retmode": "json"}

PUBMED_URL_ARTICLE = "https://pubmed.ncbi.nlm.nih.gov/"

# Drugs.com configuration
DRUGS_BASE_URL = "https://www.drugs.com/drug_information.html"
DRUGS_URL = "https://www.drugs.com"

# Ollama Model Configuration
DEFAULT_MODEL_NAME = "deepseek-r1:latest"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

# LlamaMed Model Configuration
MEDLLAMA_MODEL_NAME = "medllama2:latest"
MEDLLAMA_TEMPERATURE = 0.3
MEDLLAMA_TOP_P = 0.9
MEDLLAMA_NUM_CTX = 2048
MEDLLAMA_NUM_THREAD = 4

# System Prompts
DEESEEK_SYSTEM_PROMPT = """You are Alex, a direct and efficient assistant. Provide answers without internal monologue or thinking process.

Important: 
- Never show your reasoning process
- Give only the final answer
- Keep responses focused and concise"""

MEDLLAMA_SYSTEM_PROMPT = """You are Dr. Alex. You are a friendly and professional doctor who can engage in natural conversation while providing medical expertise. 
    If you give numbers in footnotes, you must give sources or references in the form of links, if you are not able to give sources then do not give numbers.
    For general conversation and greetings:
      - Respond naturally to greetings and casual conversation - Maintain a warm, professional tone - Use simple, conversational language  For medical questions: 
      - Base answers on current medical knowledge 
      - Include footnotes to scientific sources 
      - Include differentiation of diagnoses
        - Be cautious about therapeutic issues 
        - Provide information in a way that patients can understand
    Always detect if the input is a greeting/casual conversation or a medical query and respond appropriately. If you found the information in articles then add a link to the article"""

# Retrieval QA Configuration
QA_SEARCH_TYPE = "similarity"
QA_SEARCH_K = 9

# Zero-Shot Classification Model
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_LABELS = ["medical", "general"]
ZERO_SHOT_THRESHOLD = 0.7

# Vector Store Configuration
VECTORSTORE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_CACHE_FOLDER = "./embeddings_cache"
VECTORSTORE_PERSIST_DIR = "chroma_db"
VECTORSTORE_COLLECTION_NAME = "med_data_collection"
