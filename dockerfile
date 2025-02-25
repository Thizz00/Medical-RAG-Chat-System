FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    unzip \
    jq

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install black flake8 pytest pytest-cov selenium

RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoModel.from_pretrained('facebook/bart-large-mnli'); \
    AutoTokenizer.from_pretrained('facebook/bart-large-mnli')"

COPY . .

RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/app
ENV STREAMLIT_EMAIL=""

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
