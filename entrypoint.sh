#!/bin/bash
set -e 

echo "Running tests..."
python -m pytest -s -v --durations=0

echo "Formatting Python code with Black and checking with Flake8..."
black .
flake8 --max-line-length=88 --ignore=E501,F841,W291,F401 .

echo "Fetching and indexing medical data before starting Streamlit..."
python -c "from components.data_loader import fetch_medical_data; data = fetch_medical_data(); print(f'Fetched {len(data)} records.')"

export STREAMLIT_EMAIL=""

exec streamlit run main.py
