#!/bin/bash

# ==============================================================================
# SETUP SCRIPT PER IL MODELLO DIRECT CON ENCODER T5GEMMA
# ==============================================================================
# Questo script prepara l'ambiente completo per eseguire il training.
# ATTENZIONE: Eseguire questo script dopo essersi autenticati con Hugging Face
# usando il comando: huggingface-cli login
# ==============================================================================

# 1. Creazione delle directory necessarie
echo "--- [FASE 1/4] Creazione delle directory ---"
mkdir -p logs
mkdir -p outputs
mkdir -p datasets
mkdir -p nltk
mkdir -p cache # La cartella per il modello T5Gemma locale
echo "Directory create con successo."
echo ""

# 2. Download e preparazione dei dataset (mantenuto come l'originale)
# Le righe di wget sono commentate, decommentale se devi scaricare i file.
echo "--- [FASE 2/4] Preparazione dei dataset ---"
# echo "Download in corso..."
# wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
# wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
# wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz
# wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz

# Decompressione dei file (fallirà se i file .gz non sono presenti)
echo "Decompressione dei dataset..."
gzip -d -f reviews_CDs_and_Vinyl_5.json.gz > /dev/null 2>&1
gzip -d -f reviews_Clothing_Shoes_and_Jewelry_5.json.gz > /dev/null 2>&1
gzip -d -f reviews_Toys_and_Games_5.json.gz > /dev/null 2>&1
gzip -d -f reviews_Video_Games_5.json.gz > /dev/null 2>&1

# Spostamento dei dataset nella cartella designata
echo "Spostamento dei file JSON..."
mv reviews_CDs_and_Vinyl_5.json ./datasets/ 2>/dev/null
mv reviews_Clothing_Shoes_and_Jewelry_5.json ./datasets/ 2>/dev/null
mv reviews_Toys_and_Games_5.json ./datasets/ 2>/dev/null
mv reviews_Video_Games_5.json ./datasets/ 2>/dev/null
echo "Dataset pronti."
echo ""

# 3. Setup dell'ambiente virtuale con le versioni corrette
echo "--- [FASE 3/4] Installazione delle dipendenze Python ---"
# Librerie standard
pip install nltk scikit-learn pandas fire

# Installazione di PyTorch (versione nightly >= 2.6 per CUDA 12.1)
# Questo è necessario per la compatibilità con l'ultima versione di transformers.
echo "Installazione di PyTorch (versione nightly)..."
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Installazione di Transformers (versione più recente direttamente da GitHub)
# Questo è necessario per il supporto all'architettura T5Gemma.
echo "Installazione di Transformers (da GitHub)..."
pip install git+https://github.com/huggingface/transformers.git

echo "Dipendenze Python installate."
echo ""

# 4. Download delle dipendenze per l'uso offline
echo "--- [FASE 4/4] Download dei componenti per l'uso offline ---"
# Download dei dati di NLTK (stopwords)
echo "Download di NLTK stopwords..."
python -c "import nltk; nltk.download('stopwords', download_dir='./nltk')"

# Download e caching del modello T5Gemma nella cartella locale './cache'
# Questo script deve essere configurato correttamente.
echo "Download del modello T5Gemma tramite cache_model.py..."
python cache_model.py

echo ""
echo "=============================================================================="
echo "Setup completato!"
echo "Ora puoi lanciare i tuoi job di training con sbatch."
echo "=============================================================================="
