import math
import os
import copy
from datetime import datetime
import fire
from itertools import product
import json
import csv
import torch.nn.functional as F

import reset_seed
import random

SEED = 242
reset_seed.frozen(SEED)

# MODIFICA: Gestione robusta del device
import torch as tc
device = "cuda:0" if tc.cuda.is_available() else "cpu"
print(f"INFO: Using device: {device}")
if device.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[-1]

from metrics.evaluate import CTREvaluator
import numpy as np
import pandas as pd
import tqdm

from models.DIRECT import DIRECT
from datas.dataset import Dataset, MetaIndex, DocumentDataset
from datas.logger import print
from datas.preprocess import initialize_dataset
from models.Losses import BPRLoss

# MODIFICA: Impostato il nuovo modello T5Gemma
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
             # MODIFICA: Imposta a False dopo aver generato i dati una volta
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

# --- MODIFICA 1: Rimuoviamo 'plm' da MODEL_CONFIG ---
# Il nuovo modello DIRECT non ha più l'encoder, quindi non necessita di questo parametro.
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

# --- MODIFICA 2: Riduzione del Learning Rate per stabilità ---
# Come scoperto dal log precedente, un LR più basso previene l'esplosione dei gradienti (loss=nan)
TRAIN_CONFIG = {
    "learn_rate": 1e-4, # Ridotto da 0.001
    "use_amp": True,
    "batch_size": 64, # Mantenuto come da tua configurazione funzionante
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



# get_subsets() adesso ha 'datafile' (ex 'root') e setup (rimosso dalle variabili globali)

def get_subsets(datafile, setup, format_, configs, splits=("train", "valid", "test")):
    assert isinstance(configs, dict) and len(configs) == 3
    assert "init" in configs and "data" in configs and "meta" in configs
    configs = copy.deepcopy(configs)

    # 1. Questa chiamata prepara le directory e i file di testo base.
    root_info = initialize_dataset(datafile, format_, dotokenize=True, **configs["init"])
    data_root = root_info["root"]

    # 2. --- MODIFICA CHIAVE: Creiamo l'oggetto MetaIndex UNA SOLA VOLTA ---
    # Questa è l'unica chiamata che caricherà i 138GB di embedding in RAM.
    main_meta = MetaIndex(data_root, **configs["meta"])

    configs["init"]["valid"] = configs["init"]["test"] = 0.0

    # 3. Le chiamate successive a initialize_dataset servono solo a creare i file json per gli split.
    train_info = initialize_dataset(os.path.join(data_root, "train.json"), format_, users=main_meta.users,
                                    items=main_meta.items, **configs["init"])

    # 4. --- MODIFICA CHIAVE: Passiamo l'oggetto 'main_meta' già creato ---
    # Non creiamo un nuovo MetaIndex, ma riutilizziamo quello esistente.
    subsets = [Dataset(train_info["root"], "train", format_, main_meta, **configs["data"], setup=setup)]
    paths = {"train": train_info["root"]}

    for split in splits[1:]:
        splitfile = os.path.join(data_root, f"{split}.json")
        info = initialize_dataset(splitfile, format_, users=main_meta.users, items=main_meta.items, **configs["init"])

        # 5. --- MODIFICA CHIAVE: Anche qui, riutilizziamo 'main_meta' ---
        subsets.append(Dataset(info["root"], "train", format_, main_meta, **configs["data"],
                               paths=paths, split=split, setup=setup))
        if split == "valid":
            paths['valid'] = info["root"]

    # 6. Questa parte rimane per compatibilità, anche se non usata dal modello pre-calcolato.
    documents = DocumentDataset(os.path.join(data_root, "item_doc.txt"),
                                configs["meta"]["tokenizer"], configs["meta"]["len_doc"],
                                configs["meta"]["keep_head"], 1)

    # 7. Ritorniamo l'istanza 'main_meta' che contiene i dati caricati.
    return main_meta, subsets, documents



# Sostituisci la tua funzione fit_BPR esistente con questa
def fit_BPR(datafile, grid_seed, setup, datas, model, optimizer, learn_rate, batch_size, num_epochs, max_norm,
            log_frequency,
            frozen_train_size,
            decay_rate, decay_tol, early_stop, weight_decay, use_amp, workers):
    optimizer = getattr(tc.optim, optimizer)(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)

    # Inizializzazione dello Scaler per la mixed precision
    scaler = tc.amp.GradScaler(device='cuda', enabled=use_amp)

    train = tc.utils.data.DataLoader(datas[0], batch_size=batch_size, shuffle=True, num_workers=workers)

    subfolder = (os.path.split(datafile)[-1]).replace(".json", "")
    folder = "outputs/" + subfolder
    os.makedirs(folder, exist_ok=True)

    best_model = folder + "/setup-%s_SEED-%d_GRIDSEED-%s_DIRECT.pth" % (setup, SEED, grid_seed)

    # Logica originale: si cerca di minimizzare la recall (che è negativa)
    best_recall = float("inf")
    best_avg_recall = float("inf")

    total_tol, current_tol = 0, 0
    bpr_loss_func = BPRLoss()

    num_batches_per_epoch = math.ceil(len(datas[0]) / batch_size)
    progress = tqdm.tqdm(total=num_batches_per_epoch * num_epochs)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (positives, negatives) in enumerate(train):
            optimizer.zero_grad()

            # Contesto autocast per la Mixed Precision
            with tc.amp.autocast(device_type="cuda", enabled=use_amp):
                pos_scores = model(**positives)
                neg_scores = model(**negatives)
                loss = bpr_loss_func(pos_scores, neg_scores)

            # Scaler per la Backward Pass
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress.update(1)

        avg_epoch_loss = epoch_loss / num_batches_per_epoch
        print(f"\n[Fine Epoca {epoch}] Loss Media: {avg_epoch_loss:.4f}")

        recall_at_10, avg_recall = predict_bpr(model, datas[1])
        # NOTA: La recall viene restituita come valore negativo da predict_bpr
        print(f"Valid Epoch={epoch} | Recall@10={recall_at_10:.4f} | AVG_Recall={avg_recall:.4f}")

        # --- LOGICA DI EARLY STOPPING ORIGINALE RIPRISTINATA ---
        if recall_at_10 < best_recall:
            current_tol = 0
            best_recall = recall_at_10
            best_avg_recall = avg_recall
            print(f"Nuovo migliore score (min recall). Salvataggio del modello in {best_model}")
            tc.save(model.state_dict(), best_model)
        else:
            current_tol += 1
            if current_tol == decay_tol:
                print("Reducing learning rate by %.4f" % decay_rate)
                scheduler.step()
                model.load_state_dict(tc.load(best_model))
                current_tol = 0
                total_tol += 1
            if total_tol == early_stop + 1:
                progress.close()
                print(f"Early stop all'epoca {epoch} con best_recall={best_recall:.4f}")
                break

    progress.close()

    print(f"\n--- Riepilogo Training ---")
    print(f"\tMigliore Score (min recall): {best_recall:.4f}")
    print(f"\tCorrispondente AVG Recall: {best_avg_recall:.4f}")

    return best_recall, model, best_model


def fit(datafile, grid_seed, setup, datas, model, optimizer, learn_rate, batch_size, num_epochs, max_norm,
        log_frequency,
        frozen_train_size,
        decay_rate, decay_tol, early_stop, weight_decay, use_amp, workers):
    optimizer = getattr(tc.optim, optimizer)(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
    progress = tqdm.tqdm(total=math.ceil(len(datas[0]) / batch_size) * num_epochs)
    scaler = tc.amp.GradScaler(device='cuda', enabled=use_amp)
    ctr_grader = CTREvaluator(threshold=4.0)

    train = tc.utils.data.DataLoader(datas[0], batch_size=batch_size, shuffle=True, num_workers=workers)
    valid = tc.utils.data.DataLoader(datas[1], batch_size=batch_size, shuffle=False, num_workers=2)

    epoch, total_tol, current_tol = 0, 0, 0

    subfolder = (os.path.split(datafile)[-1]).replace(".json", "")
    folder = "outputs/" + subfolder
    os.makedirs(folder, exist_ok=True)

    best_model = folder + "/setup-%s_SEED-%d_GRIDSEED-%s_DIRECT.pth" % (setup, SEED, grid_seed)

    best_score = float("inf")
    for epoch in range(num_epochs):
        model.train()

        for batch in train:
            if batch["recommend"].shape[0] != batch_size:
                continue
            optimizer.zero_grad()

            if use_amp:
                with tc.amp.autocast(device_type="cuda"):
                    y_real = batch["score"]
                    y_pred = model(**batch)
                    loss = model.compute_loss(y_real, y_pred)
                scaler.scale(loss).backward()
                tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                y_real = batch["score"]
                y_pred = model(**batch)
                loss = model.compute_loss(y_real, y_pred)
                loss.backward()
                tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
            progress.update(1)

        # mse = predict(model, valid)

        auc, acc, f1, mse, mae, loss = ctr_grader.evaluate(model, valid)
        print("Valid Epoch=%d | Accuracy=%.4f | AUC=%.4f | F1=%.4f | MSE=%.4f | MAE=%.4f" % (
            epoch, acc, auc, f1, mse, mae))

        if mse < best_score - 5e-5:
            current_tol = 0
            best_score = mse
            tc.save(model.state_dict(), best_model)
        else:
            current_tol += 1
            if current_tol == decay_tol:
                print("Reducing learning rate by %.4f" % decay_rate)
                scheduler.step()
                model.load_state_dict(tc.load(best_model))
                current_tol = 0
                total_tol += 1
            if total_tol == early_stop + 1:
                progress.close()
                print("Early stop at epoch %s with MSE=%.4f" % (epoch, mse))
                break

    print(f"\tBest MSE: {best_score:.4f}")

    return best_score, model, best_model


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


def preconfigure(
        setup: str = "BPR",
        datafile: str = "./datasets/reviews_Toys_and_Games_5.json"):

    format_ = yelp if "yelp" in datafile else amazon
    get_subsets(datafile, setup, format_, DATA_CONFIG)
    return


def train(lr: float = TRAIN_CONFIG["learn_rate"],
          aspc_num: int = MODEL_CONFIG["aspc_num"],
          g2: float = MODEL_CONFIG["gamma2"],
          setup: str = "BPR",
          datafile: str = "./datasets/reviews_Toys_and_Games_5.json",
          grid_seed: str = "False"):
    print("\n===== CONFIGURAZIONE ESPERIMENTO =====")
    print(f"Datafile:       {datafile}")
    print(f"Setup:          {setup}")
    print(f"Learning rate:  {lr}")
    print(f"aspc_num:       {aspc_num}")
    print(f"gamma2:         {g2}")
    print(f"Grid Seed:      {grid_seed}")
    print("======================================\n")

    format_ = yelp if "yelp" in datafile else amazon
    MODEL_CONFIG["aspc_num"] = aspc_num
    MODEL_CONFIG["gamma2"] = g2
    TRAIN_CONFIG["learn_rate"] = lr

    meta, datas, item_doc = get_subsets(datafile, setup, format_, DATA_CONFIG)
    model = DIRECT(user_num=len(meta.users),
                   item_num=len(meta.items),
                   **MODEL_CONFIG)

    # --- MODIFICA 3: Rimuoviamo la chiamata a prepare_item_embedding ---
    # Questa funzione è ora obsoleta e non fa nulla, quindi la chiamata può essere rimossa.
    # model.prepare_item_embedding(item_doc)

    now = datetime.now()
    best_score, best_model = 0, ""
    print(f"Inizio Training: {now}")

    if setup.upper() == 'BPR':
        best_score, model, best_model = fit_BPR(datafile, grid_seed, 'BPR', datas, model, **TRAIN_CONFIG)
    if setup.lower() == 'default':
        best_score, model, best_model = fit(datafile, grid_seed, 'default', datas, model, **TRAIN_CONFIG)

    done = datetime.now()
    print(f"Fine Training: {done}")
    print(f"Tempo impiegato: {done - now}")

    return best_score, model, best_model


def grid(HYPERPARAMETERS=None,
         setup: str = "default",
         grid_seed: str = "default",
         datafile: str = "./datasets/reviews_Toys_and_Games_5.json"):
    if HYPERPARAMETERS is None:
        HYPERPARAMETERS = STANDARD_HYPERPARAMETERS

    if setup.upper() == "BPR":
        del HYPERPARAMETERS['gamma2']

    subfolder = (os.path.split(datafile)[-1]).replace(".json", "")
    folder = "outputs/" + subfolder
    os.makedirs(folder + "/finished", exist_ok=True)

    # Creo i path necessari
    gridFile_path = folder + "/setup-%s_SEED-%d_GRIDSEED-%s_DIRECT.csv" % (setup, SEED, grid_seed)
    best_model = folder + "/setup-%s_SEED-%d_GRIDSEED-%s_DIRECT_BEST.pth" % (setup, SEED, grid_seed)
    complete_grid = folder + "/finished/setup-%s_SEED-%d_GRIDSEED-%s_DIRECT.csv" % (setup, SEED, grid_seed)

    # Controllo se la grid sia stata già eseguita in passato
    if os.path.exists(complete_grid):
        raise ValueError("Questa ricerca Grid è stata già completata")

    # Creo il file di grid.csv se non presente
    if not os.path.exists(gridFile_path):
        with open(gridFile_path, mode='w', newline='') as file_csv:
            writer = csv.DictWriter(file_csv, fieldnames=["parameters", "best_score"])
            writer.writeheader()

    # Inizializzo le variabili necessarie
    best_score = float("inf")
    best_parameters = {}
    executed = []
    df = pd.read_csv(gridFile_path)

    # Recupero il miglior score e parametri passati (situazione post-crash)
    for row in df.itertuples(index=False):
        combination = json.loads(row.parameters)
        score = row.best_score
        executed.append(combination)
        if score < best_score:
            best_score = score
            best_parameters = combination
    del df

    # Creo tutte le possibili combinazioni per la grid (Escluse quelle già calcolate)
    keys = list(HYPERPARAMETERS.keys())
    values = list(HYPERPARAMETERS.values())
    combinations = [dict(zip(keys, v)) for v in product(*values) if dict(zip(keys, v)) not in executed]

    # Eseguo la grid con le combinazioni da calcolare
    for combination in combinations:

        # Faccio una copia a parte (combination mi serve dopo per aggiornare il file grid.csv)
        tmp = copy.deepcopy(combination)
        tmp['grid_seed'] = grid_seed
        tmp['setup'] = setup
        tmp['datafile'] = datafile

        score, _, path_model = train(**tmp)

        # Aggiorno il nuovo score
        if score < best_score:
            best_score = score
            best_parameters = combination

            # Cambio il file .pth della grid migliore
            os.replace(path_model, best_model)

        # Aggiorno il file di combinazioni eseguite
        with open(gridFile_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([json.dumps(combination), score])

    # Operazioni finali della grid
    os.replace(gridFile_path, complete_grid)

    # Resoconto della grid
    print(f"Best parameters found: {best_parameters}")
    print(f"Best Score: {best_score}")


if __name__ == "__main__":
    fire.Fire()
