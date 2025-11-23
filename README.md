# LLM-Enhanced DIRECT Recommender System 🚀

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Evolution of Recommender Systems with Ranking-Based Approach and LLM Embeddings**

This repository contains the implementation of a hybrid Recommender System that integrates **Large Language Model (LLM)** embeddings to improve ranking performance. It is built upon the DIRECT framework but significantly refactored to support state-of-the-art semantic representations and efficient training pipelines.

## 🌟 Key Features & Engineering

Unlike standard implementations, this project focuses on **scalability**, **ranking optimization**, and **software engineering best practices**:

* **LLM Integration (T5-Gemma):** Replaced legacy BERT embeddings with **google/t5gemma-2b**, enabling superior semantic understanding of user reviews and item descriptions.
* **Offline Pre-calculation Pipeline:** Designed a multi-stage data ingestion pipeline (`precompute_embeddings.py` and `consolidate_embeddings.py`) to generate and store 100GB+ of embeddings offline. This removes the runtime bottleneck of LLM forward passes, drastically speeding up training.
* **Ranking-Oriented Loss (BPR):** Refactored the training objective from simple rating prediction (MSE) to **Bayesian Personalized Ranking (BPR) Loss**, aligning the model optimization with the ranking task (Top-K recommendation).
* **Memory Optimization:** Implemented a custom `MetaIndex` loader that handles massive embedding matrices efficiently using direct RAM allocation strategies (replacing standard memory mapping for performance gains on high-memory nodes).

## 📂 Repository Structure

* `Main_T5Gemma.py`: Main entry point for training and grid search.
* `Test_Embeddings.py`: Script for evaluating trained models and testing embedding quality.
* `models/DIRECT.py`: Refactored model architecture (Encoder removed to accept pre-computed embeddings).
* `models/Losses.py`: Implementation of BPR Loss.
* `precompute_embeddings.py`: Script to generate T5-Gemma embeddings from raw text using batch processing.
* `consolidate_embeddings.py`: Utility to merge fragmented batch files into unified NumPy arrays for fast loading.
* `cache_model.py`: Helper script to download and cache the LLM locally.
* `datas/`: Data loading logic and custom `Dataset` classes.

## 🛠️ Setup & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/AMarz4400/llm-ranking-direct.git
cd llm-ranking-direct

# Install dependencies
pip install -r requirements.txt
2. Data Preparation Pipeline

This project requires a multi-step data preparation to handle the LLM computational load upfront.

Step A: Download & Setup

Downloads datasets (Amazon Reviews), installs NLTK data, and caches the T5-Gemma model.

sh gemma_setup.sh

Step B: Pre-compute EmbeddingsGenerates semantic embeddings for Users and Items using T5-Gemma. Requires GPU.

python precompute_embeddings.py

Step C: Consolidate DataMerges the thousands of batch files generated in Step B into optimized .npy binaries for fast training access.

python consolidate_embeddings.py


🚀 UsageTraining (BPR Loss)To train the model using the BPR ranking setup on the pre-computed data:

(example)
python Main_T5Gemma.py train \
    --lr 0.0001 \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json \
    --aspc_num 5 \
    --num_epochs 50


Hyperparameter Grid SearchTo run a Grid Search for hyperparameter optimization (saves results in outputs/):

(example)
python Main_T5Gemma.py grid \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json


Evaluation & TestingTo evaluate a trained model checkpoint:

(example)
python Test_Embeddings.py test \
    --lr 0.0001 \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json \
    --parameters "./outputs/reviews_Clothing_Shoes_and_Jewelry_5/YOUR_BEST_MODEL.pth"


📊 Performance
By switching to BPR Loss and T5-Gemma embeddings, the model achieves improved results on sparse datasets compared to traditional baselines, on ranking metrics like Recall, Precision, Hit Rate.

📚 References & Acknowledgements
This project is an evolution of the DIRECT framework.Modifications were made to integrate LLM capabilities and ranking objectives.If you use this code, please cite the original 

If you use this code, please cite the original DIRECT paper and the model providers:

**Original DIRECT Paper:**
> *Wu, X., Wan, H., Tan, Q., Yao, W., & Liu, N. (2024). "DIRECT: Dual Interpretable Recommendation with Multi-aspect Word Attribution." ACM Transactions on Intelligent Systems and Technology (TIST).*

**LLM Backbone:**
> *Google DeepMind. (2024). "T5Gemma: Encoder-Decoder Large Language Models."*

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.