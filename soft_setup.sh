# dir generation
mkdir logs
mkdir outputs
mkdir datasets
mkdir nltk
mkdir -p cache/Tokenizer
mkdir -p cache/Model

# environment setup
echo Setup environmen
pip install nltk
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn
pip install transformers
pip install pandas

# downloading dependencies (for offline processing)
python -c "import nltk; nltk.download('stopwords', download_dir='./nltk')"
python -c "import transformers; transformers.BertTokenizer.from_pretrained('prajjwal1/bert-small', cache_dir='./cache/Tokenizer')"
python -c "import transformers; transformers.BertModel.from_pretrained('prajjwal1/bert-small', cache_dir='./cache/Model', use_safetensors=True)"
