# LLM-Enhanced DIRECT Recommender System 🚀

**Evolution of Recommender Systems with Ranking-Based Approach and LLM Embeddings**

Questo repository contiene l'implementazione di un **Sistema di Raccomandazione ibrido** che integra embedding di **Large Language Model (LLM)** per migliorare le performance di ranking. È basato sul framework DIRECT, ma significativamente rifattorizzato per supportare rappresentazioni semantiche all'avanguardia e pipeline di training efficienti.

---

## 🌟 Caratteristiche Chiave & Ingegneria

A differenza delle implementazioni standard, questo progetto si concentra su **scalabilità**, **ottimizzazione del ranking** e **best practice di ingegneria del software**:

* **Integrazione LLM (T5-Gemma):** Sostituiti gli embedding legacy di BERT con **`google/t5gemma-2b`**, consentendo una comprensione semantica superiore delle recensioni degli utenti e delle descrizioni degli articoli.
* **Pipeline di Pre-calcolo Offline:** Progettata una pipeline di ingestione dati multi-stadio (`precompute_embeddings.py` e `consolidate_embeddings.py`) per generare e archiviare **100GB+ di embedding offline**. Questo rimuove il collo di bottiglia a runtime dei forward pass LLM, accelerando drasticamente il training.
* **Loss Orientata al Ranking (BPR):** Rifattorizzato l'obiettivo di training dalla semplice predizione del rating (MSE) alla **Bayesian Personalized Ranking (BPR) Loss**, allineando l'ottimizzazione del modello al task di ranking (raccomandazione Top-K).
* **Ottimizzazione della Memoria:** Implementato un loader **`MetaIndex`** personalizzato che gestisce matrici di embedding massicce in modo efficiente utilizzando strategie di allocazione RAM diretta (sostituendo il memory mapping standard per guadagni di performance sui nodi ad alta memoria).

---

## 📂 Struttura del Repository

| File/Cartella | Descrizione |
| :--- | :--- |
| `Main_T5Gemma.py` | Punto di ingresso principale per il training e la grid search. |
| `Test_Embeddings.py` | Script per la valutazione dei modelli addestrati e il testing della qualità degli embedding. |
| `models/DIRECT.py` | Architettura del modello rifattorizzata (Encoder rimosso per accettare embedding pre-calcolati). |
| `models/Losses.py` | Implementazione della BPR Loss. |
| `precompute_embeddings.py` | Script per generare gli embedding T5-Gemma dal testo raw tramite batch processing. |
| `consolidate_embeddings.py` | Utility per unire i file batch frammentati in array NumPy unificati per un caricamento veloce. |
| `cache_model.py` | Script di supporto per scaricare e mettere in cache l'LLM localmente. |
| `datas/` | Logica di caricamento dati e classi `Dataset` personalizzate. |

---

## 🛠️ Setup & Installazione

### 1. Configurazione dell'Ambiente

Clona il repository:

```bash
git clone [https://github.com/AMarz4400/llm-ranking-direct.git](https://github.com/AMarz4400/llm-ranking-direct.git)
cd llm-ranking-direct
Installa le dipendenze principali e configura i dataset:Questo script gestisce PyTorch Nightly, Transformers (GitHub), e i dati NLTK.Bashsh gemma_setup.sh
Installa le utility rimanenti:Bashpip install -r requirements.txt
2. Pipeline di Preparazione dei DatiQuesto progetto richiede una preparazione dei dati in più fasi per gestire in anticipo il carico computazionale dell'LLM.PassaggioScriptDescrizioneRequisitiA. Download & Setupsh gemma_setup.shScarica i dataset (Amazon Reviews), installa i dati NLTK e mette in cache il modello T5-Gemma.B. Pre-calcolo degli Embeddingpython precompute_embeddings.pyGenera gli embedding semantici per Utenti e Articoli usando T5-Gemma.Richiede GPUC. Consolidamento dei Datipython consolidate_embeddings.pyUnisce le migliaia di file batch generati nel Passaggio B in binari .npy ottimizzati per un accesso veloce durante il training.🚀 UtilizzoTraining (BPR Loss)Per addestrare il modello utilizzando la configurazione di ranking BPR sui dati pre-calcolati:Bashpython Main_T5Gemma.py train \
    --lr 0.0001 \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json \
    --aspc_num 5 \
    --num_epochs 50
Hyperparameter Grid SearchPer eseguire una Grid Search per l'ottimizzazione degli iperparametri (salva i risultati in outputs/):Bashpython Main_T5Gemma.py grid \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json
Valutazione & TestingPer valutare un checkpoint del modello addestrato:Bashpython Test_Embeddings.py test \
    --lr 0.0001 \
    --setup BPR \
    --datafile ./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json \
    --parameters "./outputs/reviews_Clothing_Shoes_and_Jewelry_5/YOUR_BEST_MODEL.pth"
📊 PerformancePassando a BPR Loss e agli embedding T5-Gemma, il modello ottiene risultati migliorati sui dataset sparsi rispetto alle baseline tradizionali, sulle metriche di ranking come Recall, Precision e Hit Rate.📚 Riferimenti & RiconoscimentiQuesto progetto è un'evoluzione del framework DIRECT. Sono state apportate modifiche per integrare le capacità LLM e gli obiettivi di ranking.Se utilizzi questo codice, ti preghiamo di citare l'articolo DIRECT originale e i fornitori del modello:Articolo DIRECT Originale:Wu, X., Wan, H., Tan, Q., Yao, W., & Liu, N. (2024). "DIRECT: Dual Interpretable Recommendation with Multi-aspect Word Attribution." ACM Transactions on Intelligent Systems and Technology (TIST).Backbone LLM:Google DeepMind. (2024). "T5Gemma: Encoder-Decoder Large Language Models."📜 LicenzaQuesto progetto è rilasciato sotto licenza MIT - consulta il file LICENSE per i dettagli.