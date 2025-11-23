# dir generation
mkdir logs
mkdir outputs
mkdir datasets
mkdir nltk
mkdir -p cache/Tokenizer
mkdir -p cache/Model

# Dataset retrieval
echo Downloading datasets
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz
#wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz

# unpack dataset
gzip -d reviews_CDs_and_Vinyl_5.json.gz
gzip -d reviews_Clothing_Shoes_and_Jewelry_5.json.gz
gzip -d reviews_Toys_and_Games_5.json.gz
gzip -d reviews_Video_Games_5.json.gz

# moving dataset into designated folder
mv reviews_CDs_and_Vinyl_5.json ./datasets/
mv reviews_Clothing_Shoes_and_Jewelry_5.json ./datasets/
mv reviews_Toys_and_Games_5.json ./datasets/
mv reviews_Video_Games_5.json ./datasets/

# environment setup
echo Setup environmen
pip install nltk
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn
pip install transformers
pip install pandas

# downloading dependencies (for offline processing)
#python -c "import nltk; nltk.download('stopwords', download_dir='./nltk')"
#python -c "import transformers; transformers.BertTokenizer.from_pretrained('prajjwal1/bert-small', cache_dir='./cache/Tokenizer')"
#python -c "import transformers; transformers.BertModel.from_pretrained('prajjwal1/bert-small', cache_dir='./cache/Model', use_safetensors=True)"
