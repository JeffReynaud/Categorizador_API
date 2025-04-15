#!/bin/bash

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo de spaCy para español
python -m spacy download es_core_news_lg 