# LLM-Enhanced DIRECT Recommender System 🚀

**Evolution of Recommender Systems with Ranking-Based Approach and LLM Embeddings**

Questo repository contiene l'implementazione di un **Sistema di Raccomandazione ibrido** che integra embedding di **Large Language Model (LLM)** per migliorare le performance di ranking. È basato sul framework DIRECT, ma significativamente rifattorizzato per supportare rappresentazioni semantiche all'avanguardia e pipeline di training efficienti.

---

## 🌟 Caratteristiche Chiave & Ingegneria

A differenza delle implementazioni standard, questo progetto si concentra su **scalabilità**, **ottimizzazione del ranking** e **best practice di ingegneria del software**:

- **Integrazione LLM (T5-Gemma):** Sostituiti gli embedding legacy di BERT con `google/t5gemma-2b`, consentendo una comprensione semantica superiore delle recensioni degli utenti e delle descrizioni degli articoli.
- **Pipeline di Pre-calcolo Offline:** Progettata una pipeline di ingestione dati multi-stadio (`precompute_embeddings.py` e `consolidate_embeddings.py`) per generare e archiviare **100GB+ di embedding offline**.
- **Loss Orientata al Ranking (BPR):** Rifattorizzato l'obiettivo di training da MSE a **Bayesian Personalized Ranking (BPR) Loss**.
- **Ottimizzazione della Memoria:** Implementato un loader `MetaIndex` per gestire massive matrici di embedding in RAM.

---

## 📂 Struttura del Repository

| File/Cartella | Descrizione |
|---------------|-------------|
| `Main_T5Gemma.py` | Punto di ingresso principale per il training e la grid search. |
| `Test_Embeddings.py` | Script per valutare i modelli addestrati e testare la qualità degli embedding. |
| `models/DIRECT.py` | Architettura del modello rifattorizzata (encoder rimosso per embedding pre-calcolati). |
| `models/Losses.py` | Implementazione della BPR Loss. |
| `precompute_embeddings.py` | Generazione embedding T5-Gemma tramite batch processing. |
| `consolidate_embeddings.py` | Consolidamento dei file batch in array NumPy. |
| `cache_model.py` | Script per scaricare e mettere in cache l'LLM localmente. |
| `datas/` | Logica di caricamento dati e dataset personalizzati. |

---

## 🛠️ Setup & Installazione

### 1. Configurazione dell'Ambiente

Clona il repository:

```bash
git clone https://github.com/AMarz4400/llm-ranking-direct.git
cd llm-ranking-direct

## Installa le dipendenze principali:

```bash
sh gemma_setup.sh

## Installa le utility rimanenti:

```bash
pip install -r requirements.txt

## 2. Pipeline di Preparazione dei Dati

Questo progetto utilizza una pipeline multi-step:

Passaggio	Script	Descrizione	Requisiti
A. Download & Setup:	
bash
sh gemma_setup.sh	

Scarica i dataset Amazon Reviews, dati NLTK e T5-Gemma.	—

B. Pre-calcolo	

bash
python precompute_embeddings.py	
Genera embedding semantici per utenti e articoli.	GPU

C. Consolidamento	
bash
python consolidate_embeddings.py	Unisce i batch di embedding in file .npy.	—

## 🚀 Utilizzo
### Training (BPR Loss)

bash
python Main_T5Gemma.py train \
    --lr 0.0001 \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json \
    --aspc_num 5 \
    --num_epochs 50


### Hyperparameter Grid Search

bash
python Main_T5Gemma.py grid \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json


### Valutazione & Testing
bash
python Test_Embeddings.py test \
    --lr 0.0001 \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json \
    --parameters "./outputs/reviews_Clothing_Shoes_and_Jewelry_5/YOUR_BEST_MODEL.pth"

## 📊 Performance
La combinazione tra BPR Loss ed embedding T5-Gemma migliora significativamente le metriche di ranking (Recall, Precision, Hit Rate) rispetto alle baseline classiche.

## 📚 Riferimenti & Riconoscimenti
Questo progetto è un'estensione del framework DIRECT.

### Articolo DIRECT originale:
Wu, X., Wan, H., Tan, Q., Yao, W., & Liu, N. (2024). DIRECT: Dual Interpretable Recommendation with Multi-aspect Word Attribution. ACM TIST.

### Backbone LLM:
Google DeepMind (2024). T5Gemma: Encoder-Decoder Large Language Models.

## 📜 Licenza
Rilasciato sotto licenza MIT. Vedi il file LICENSE.