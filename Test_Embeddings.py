import math
import os
import copy
from datetime import datetime
import fire
from itertools import product
import json
import csv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score # Aggiunto per evaluate_recommendation

import reset_seed
import random

SEED = 242
reset_seed.frozen(SEED)

# Gestione robusta del device
import torch as tc
device = "cuda:0" if tc.cuda.is_available() else "cpu"
print(f"INFO: Using device: {device}")
if device.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[-1]

# Non più necessario per le sole funzioni di test
# from metrics.evaluate import CTREvaluator 
import numpy as np
import pandas as pd
import tqdm

from models.DIRECT import DIRECT
from datas.dataset import Dataset, MetaIndex, DocumentDataset
from datas.logger import print
from datas.preprocess import initialize_dataset
# Non più necessario
# from models.Losses import BPRLoss 

# Impostato il nuovo modello T5Gemma
plm = "google/t5gemma-2b-2b-prefixlm-it"
amazon = ("reviewerID", "asin", "reviewText", "overall")
yelp = ("user_id", "business_id", "text", "stars")

DATA_CONFIG = {
    "init": {"valid": 0.1,
             "test": 0.2,
             "seed": SEED,
             "min_freq": 1,
             "pretrain": plm, # Mantenuto per compatibilità con la creazione dei path
             "num_worker": 8,
             "force_init": False,
             },
    "meta": {"tokenizer": plm, # Mantenuto per compatibilità con la firma di MetaIndex
             "num_sent": None,
             "len_sent": None,
             "num_hist": 30,
             "len_doc": 510,
             "cache_freq": 1,
             "keep_head": 1.0,
             "drop_stopword": False
             },
    "data": {"sampling": None,
             "cache_freq": 1}}

# Rimuoviamo 'plm' da MODEL_CONFIG
MODEL_CONFIG = {
    "aspc_num": 5,
    "dropout": 0.3,
    "aspc_dim": 64,
    "gamma1": 5e-3,
    "gamma2": 1e-6,
    "gamma3": 2.5,
    "beta": 0.1,
    "sampling": 0.1,
    "threshold": 0.2,
    "device": device,
}

TRAIN_CONFIG = {
    "learn_rate": 1e-4, 
    "use_amp": True,
    "batch_size": 64, 
    "workers": 4,
    "num_epochs": 50,
    "decay_rate": 0.1,
    "decay_tol": 3,
    "early_stop": 3,
    "weight_decay": 1e-6,
    "optimizer": "AdamW",
    "max_norm": 1.0,
    "frozen_train_size": 30,
    "log_frequency": 200000
}

STANDARD_HYPERPARAMETERS = {
    'lr': [1e-3, 2e-3, 1e-2],
    'g2': [1e-6, 5e-3, 1e-3],
    'aspc_num': [3, 4, 5]
}
T = "{'lr': [1e-3],'g2': [1e-6],'aspc_num': [5]}"


# --- Funzione get_subsets da mainGEMMA1.py ---
# Questa è la versione corretta che carica gli embedding
def get_subsets(datafile, setup, format_, configs, splits=("train", "valid", "test")):
    assert isinstance(configs, dict) and len(configs) == 3
    assert "init" in configs and "data" in configs and "meta" in configs
    configs = copy.deepcopy(configs)

    # 1. Questa chiamata prepara le directory e i file di testo base.
    root_info = initialize_dataset(datafile, format_, dotokenize=True, **configs["init"])
    data_root = root_info["root"]

    # 2. Creiamo l'oggetto MetaIndex UNA SOLA VOLTA
    main_meta = MetaIndex(data_root, **configs["meta"])

    configs["init"]["valid"] = configs["init"]["test"] = 0.0

    # 3. Le chiamate successive a initialize_dataset servono solo a creare i file json per gli split.
    train_info = initialize_dataset(os.path.join(data_root, "train.json"), format_, users=main_meta.users,
                                    items=main_meta.items, **configs["init"])

    # 4. Passiamo l'oggetto 'main_meta' già creato
    subsets = [Dataset(train_info["root"], "train", format_, main_meta, **configs["data"], setup=setup)]
    paths = {"train": train_info["root"]}

    for split in splits[1:]:
        splitfile = os.path.join(data_root, f"{split}.json")
        info = initialize_dataset(splitfile, format_, users=main_meta.users, items=main_meta.items, **configs["init"])

        # 5. Riutilizziamo 'main_meta'
        subsets.append(Dataset(info["root"], "train", format_, main_meta, **configs["data"],
                               paths=paths, split=split, setup=setup))
        if split == "valid":
            paths['valid'] = info["root"]

    # 6. Questa parte rimane per compatibilità
    documents = DocumentDataset(os.path.join(data_root, "item_doc.txt"),
                                configs["meta"]["tokenizer"], configs["meta"]["len_doc"],
                                configs["meta"]["keep_head"], 1)

    # 7. Ritorniamo l'istanza 'main_meta' che contiene i dati caricati.
    return main_meta, subsets, documents


# --- Funzioni di Test da mainTEST.py (con modifiche) ---

def evaluate_recommendation(model, data, k=10, N=200, U=128):
    model.eval()
    device = next(model.parameters()).device

    users = list(data.positive.keys())
    if 0 < U < len(users):
        users = random.sample(users, U)

    recalls = []
    precisions = []
    hits = []
    aucs = []

    all_items = list(range(1, data.num_item + 1))
    valid_ids = set(range(len(data.meta.item_hist)))

    with tc.no_grad():
        for user in users:
            positives_raw = data.positive[user]
            positives = [data.items[i] for i in positives_raw if i in data.items]

            if not positives:
                continue

            interacted = []
            if hasattr(data, 'interacted_train') and user in data.interacted_train:
                interacted += [data.items[i] for i in data.interacted_train[user] if i in data.items]
            if hasattr(data, 'interacted_val') and user in data.interacted_val:
                interacted += [data.items[i] for i in data.interacted_val[user] if i in data.items]

            candidates = [i for i in all_items if i in valid_ids and i not in interacted and i not in positives]
            if 0 < N < len(candidates):
                candidates = random.sample(candidates, N)

            items = candidates + positives

            tmp = [user] * len(items)
            pairs = list(zip(tmp, items))

            x = [data.meta.get_feed_dict(user, item, "") for user, item in pairs]
            
            # --- MODIFICA APPLICATA ---
            # Allineato alla logica di predict_bpr in mainGEMMA1.py
            batch = {k: tc.tensor(np.array([f[k] for f in x]), device=device) for k in x[0]}
            
            Y_pred = model(**batch).squeeze().cpu().numpy()

            Y_real = np.array([1 if item in positives else 0 for item in items])

            idx = np.argsort(-Y_pred)
            top_k = idx[:k]

            # recall@k
            recall = Y_real[top_k].sum() / len(positives)
            recalls.append(recall)

            # precision@k
            prec = Y_real[top_k].sum() / k
            precisions.append(prec)

            # hit rate@k
            hit = 1.0 if Y_real[top_k].sum() > 0 else 0.0
            hits.append(hit)

            # AUC
            try:
                auc = roc_auc_score(Y_real, Y_pred)
            except ValueError:
                auc = np.nan
            aucs.append(auc)

    mean_precision = np.nanmean(precisions)
    mean_hit = np.nanmean(hits)
    mean_auc = np.nanmean(aucs)
    mean_recall = np.nanmean(recalls)

    model.train()
    return mean_recall, mean_precision, mean_hit, mean_auc


# --- Funzione predict_bpr da mainGEMMA1.py ---
# Questa è la versione già aggiornata
def predict_bpr(model, data, k=10, N=200, U=128):
    opt = MODEL_CONFIG["device"]
    if N == -1:
        N = float("inf")
    if U == -1:
        U = float("inf")

    recall_k = 0.0
    model.eval()

    with tc.no_grad():
        users = list(data.positive.keys())
        if len(users) > U:
            users = random.sample(users, U)

        all_items = list(range(1, data.num_item + 1))  # numerici

        for user in users:
            # Item positivi (stringa → numerico)
            positives_raw = list(data.positive[user])
            positives = [data.items[i] for i in positives_raw if i in data.items]

            if not positives:
                continue

            interacted = []

            # Interazioni da train e val (anche stringa → numerico)
            if hasattr(data, "interacted_train") and user in data.interacted_train:
                interacted += [data.items[i] for i in data.interacted_train[user] if i in data.items]
            if hasattr(data, "interacted_val") and user in data.interacted_val:
                interacted += [data.items[i] for i in data.interacted_val[user] if i in data.items]

            valid_item_ids = set(range(len(data.meta.item_hist)))  # numerici

            candidate_items = [
                item for item in all_items
                if item in valid_item_ids and item not in interacted and item not in positives
            ]

            if len(candidate_items) > N:
                candidate_items = random.sample(candidate_items, N)

            positives_valid = [item for item in positives if item in valid_item_ids]
            candidate_items += positives_valid

            tmp = [user for _ in range(len(candidate_items))]
            pairs = list(zip(tmp, candidate_items))
            x = [data.meta.get_feed_dict(user, item, "") for user, item in pairs]
            x = {k: tc.tensor(np.array([d[k] for d in x]), device=opt) for k in x[0]}
            Y = model(**x).squeeze().tolist()

            scored_items = list(zip(candidate_items, Y))
            scored_items = sorted(scored_items, key=lambda pair: pair[1], reverse=True)

            top_k = scored_items[:k]
            total_relevant = len(positives)
            count = sum(1 for item, _ in top_k if item in positives)
            recall_k += (count / total_relevant)

        avg_recall = recall_k / len(users)

    model.train()

    return -recall_k, -avg_recall


# --- Funzione test (setup BPR) da mainTEST.py ---
def test(lr: float = TRAIN_CONFIG["learn_rate"],
         aspc_num: int = 3,
         g2: float = MODEL_CONFIG["gamma2"],
         setup: str = "BPR",
         datafile: str = "./datasets/reviews_Toys_and_Games_5.json",
         grid_seed: str = "False",
         parameters: str = "./outputs/reviews_Toys_and_Games_5/setup-BPR_SEED-242_GRIDSEED-aspc_3_DIRECT.pth"):
    print("\n===== CONFIGURAZIONE ESPERIMENTO =====")
    print(f"Datafile:       {datafile}")
    print(f"Setup:          {setup}")
    print(f"Learning rate:  {lr}")
    print(f"aspc_num:       {aspc_num}")
    print(f"gamma2:         {g2}")
    print(f"Grid Seed:      {grid_seed}")
    print(f"Parameters:     {parameters}")
    print("======================================\n")

    format_ = yelp if "yelp" in datafile else amazon
    MODEL_CONFIG["aspc_num"] = aspc_num
    MODEL_CONFIG["gamma2"] = g2
    TRAIN_CONFIG["learn_rate"] = lr

    meta, datas, item_doc = get_subsets(datafile, setup, format_, DATA_CONFIG)
    model = DIRECT(user_num=len(meta.users),
                   item_num=len(meta.items),
                   **MODEL_CONFIG)
    
    # --- MODIFICA APPLICATA ---
    # model.prepare_item_embedding(item_doc) # RIMOSSO
    
    model.load_state_dict(tc.load(parameters))

    now = datetime.now()

    best_score, best_model = 0, ""

    print(f"Inizio Testing: {now}")

    # Chiamiamo evaluate_recommendation e salviamo i risultati in variabili esplicite
    mean_recall, mean_precision, mean_hit, mean_auc = evaluate_recommendation(model, datas[2], k=10, N=2000, U=1280)

    done = datetime.now()
    print(f"Fine Testing: {done}")
    print(f"Tempo impiegato: {done - now}")

    # Stampiamo i risultati delle metriche di raccomandazione
    print(f"\n===== RISULTATI DEL TEST (Raccomandazione) =====")
    print(f"Recall@{10}:    {mean_recall:.4f}")
    print(f"Precision@{10}: {mean_precision:.4f}")
    print(f"Hit@{10}:       {mean_hit:.4f}")
    print(f"AUC:            {mean_auc:.4f}")
    print(f"================================================\n")

    done = datetime.now()
    print(f"Tempo impiegato: {done - now}")

    return best_score, model, best_model


# --- Funzione predict_default da mainTEST.py ---
# (Nessuna modifica necessaria)
def predict_default(model, data):
    model.eval() # Metti il modello in modalità valutazione
    device = next(model.parameters()).device # Ottieni il dispositivo del modello

    y_reals = [] # Lista per i valori reali
    y_preds = [] # Lista per i valori predetti

    # Crea un DataLoader per il set di dati di input (test o validazione)
    dataloader = tc.utils.data.DataLoader(data, batch_size=TRAIN_CONFIG["batch_size"],
                                         shuffle=False, num_workers=TRAIN_CONFIG["workers"])

    with tc.no_grad(): # Disabilita il calcolo dei gradienti per le inferenze
        for batch in dataloader:
            # Sposta il batch sul dispositivo corretto
            # NOTA: questo .to(device) non è necessario se il Dataset
            # sposta già i dati su CUDA, ma è una buona pratica difensiva.
            batch = {k: v.to(device) if isinstance(v, tc.Tensor) else v for k, v in batch.items()}


            y_real = batch["score"] # Estrai i punteggi reali
            y_pred = model(**batch) # Ottieni i punteggi predetti dal modello

            y_reals.extend(y_real.cpu().numpy()) # Aggiungi i reali (su CPU) alla lista
            y_preds.extend(y_pred.squeeze().cpu().numpy()) # Aggiungi i predetti (su CPU e 'squeezati') alla lista

    y_reals = np.array(y_reals) # Converti in array NumPy
    y_preds = np.array(y_preds)

    # Calcola le metriche
    mse = np.mean((y_reals - y_preds)**2) # Mean Squared Error
    rmse = np.sqrt(mse) # Root Mean Squared Error
    mae = np.mean(np.abs(y_reals - y_preds)) # Mean Absolute Error

    model.train() # Riporta il modello in modalità addestramento
    return mse, rmse, mae


# --- Funzione test_default da mainTEST.py ---
def test_default(lr: float = TRAIN_CONFIG["learn_rate"],
                 aspc_num: int = MODEL_CONFIG["aspc_num"],
                 g2: float = MODEL_CONFIG["gamma2"],
                 setup: str = "default", # Assicurati che il setup sia "default"
                 datafile: str = "./datasets/reviews_Toys_and_Games_5.json",
                 grid_seed: str = "False",
                 parameters: str = "./outputs/reviews_Toys_and_Games_5/setup-default_SEED-242_GRIDSEED-False_DIRECT.pth"): # Percorso del modello salvato
    print("\n===== CONFIGURAZIONE ESPERIMENTO DI TEST =====")
    print(f"Datafile:        {datafile}")
    print(f"Setup:           {setup}")
    print(f"Learning rate:   {lr}")
    print(f"aspc_num:        {aspc_num}")
    print(f"gamma2:          {g2}")
    print(f"Grid Seed:       {grid_seed}")
    print(f"Parameters:      {parameters}")
    print("==============================================\n")

    format_ = yelp if "yelp" in datafile else amazon
    MODEL_CONFIG["aspc_num"] = aspc_num # Aggiorna la configurazione del modello
    MODEL_CONFIG["gamma2"] = g2
    TRAIN_CONFIG["learn_rate"] = lr # Aggiorna la configurazione di addestramento

    # Prepara i set di dati (meta, [train, valid, test], item_doc)
    meta, datas, item_doc = get_subsets(datafile, setup, format_, DATA_CONFIG)

    # Inizializza il modello DIRECT
    model = DIRECT(user_num=len(meta.users),
                   item_num=len(meta.items),
                   **MODEL_CONFIG)
    
    # --- MODIFICA APPLICATA ---
    # model.prepare_item_embedding(item_doc) # RIMOSSO
    
    model.load_state_dict(tc.load(parameters)) # Carica i pesi del modello pre-addestrato

    now = datetime.now()
    print(f"Inizio Test: {now}")

    # Esegui la previsione sul set di test (datas[2] è il set di test)
    mse, rmse, mae = predict_default(model, datas[2])

    done = datetime.now()
    print(f"Fine Test: {done}")
    print(f"Tempo impiegato: {done - now}")

    # Stampa i risultati
    print(f"\n===== RISULTATI DEL TEST =====")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"==============================\n")

    return mse, rmse, mae, model # Restituisci le metriche e il modello


if __name__ == "__main__":
    fire.Fire()
