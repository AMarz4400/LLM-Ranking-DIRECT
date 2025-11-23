import torch as tc
import numpy as np
import os
from tqdm import tqdm
from datas.tokenizer import Tokenizer
from models.modules import PLM
import math

# --- CONFIGURAZIONE ---
PLM_NAME = "google/t5gemma-2b-2b-prefixlm-it"
# DATASET_NAME è stato rimosso qui, verrà preso dal ciclo
SEED = 242
BATCH_SIZE = 64
LEN_DOC = 510

datasets = ["reviews_Clothing_Shoes_and_Jewelry_5"]
# --------------------

def batch_generator(lines, batch_size):
    """Generatore per suddividere un elenco di linee in batch."""
    for i in range(0, len(lines), batch_size):
        yield lines[i:i + batch_size]


def generate_embeddings_batched(plm_model, tokenizer, device, source_file_path, output_dir_embs, output_dir_masks):
    """Genera embedding e maschere per i documenti in un file, salvandoli in batch."""
    print(f"Inizio il processo per il file: {source_file_path}")
    os.makedirs(output_dir_embs, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ATTENZIONE: File non trovato a: {source_file_path}. Salto questo file.")
        return

    if not lines:
        print(f"ATTENZIONE: Il file {source_file_path} è vuoto. Salto.")
        return

    num_batches = math.ceil(len(lines) / BATCH_SIZE)
    processed_count = 0

    # Utilizzo di tqdm per mostrare lo stato di avanzamento per il file corrente
    for batch_lines in tqdm(batch_generator(lines, BATCH_SIZE), total=num_batches,
                            desc=f"Generando embedding per {os.path.basename(source_file_path)}"):

        batch_input_ids = []
        batch_attention_masks = []

        for line in batch_lines:
            doc_text = line.strip().replace("\t", " ")
            # Assumiamo che tokenizer.transform() gestisca la tokenizzazione
            input_ids, attention_mask = tokenizer.transform(doc_text.split())
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)

        # Converti le liste in tensori PyTorch sul device specificato
        input_ids_tensor = tc.tensor(batch_input_ids, device=device)
        attention_mask_tensor = tc.tensor(batch_attention_masks, device=device)

        with tc.no_grad():
            # Ottieni gli embedding dal modello PLM
            embedding_output = plm_model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)

        # Sposta gli embedding e le maschere sulla CPU per il salvataggio
        embeddings_cpu = embedding_output.cpu().numpy()
        masks_cpu = attention_mask_tensor.cpu().numpy()

        # Salva ogni singolo embedding e maschera come file .npy
        for i in range(len(batch_lines)):
            current_index = processed_count + i
            # Salva l'embedding
            np.save(os.path.join(output_dir_embs, f"{current_index}.npy"), embeddings_cpu[i])
            # Salva la maschera di attenzione
            np.save(os.path.join(output_dir_masks, f"{current_index}.npy"), masks_cpu[i])

        processed_count += len(batch_lines)
    
    print(f"Completato il processo per il file: {source_file_path}. Totale documenti elaborati: {processed_count}")


def main():
    # --- Inizializzazione Unica ---
    device = "cuda:0" if tc.cuda.is_available() else "cpu"
    print(f"Usando il device: {device}")

    print("Caricamento del modello PLM e del tokenizer (una sola volta)...")
    plm_model = PLM(PLM_NAME, dropout=0.0).to(device)
    plm_model.eval() # Imposta il modello in modalità valutazione
    tokenizer = Tokenizer(PLM_NAME, max_word=LEN_DOC)
    print("Modello e tokenizer caricati con successo! ✅")

    print("\n--- Inizio l'elaborazione di tutti i dataset ---")

    # --- Ciclo su tutti i dataset ---
    for dataset_name in datasets:
        print(f"\n========================================================")
        print(f"Inizio l'elaborazione per il dataset: {dataset_name}")
        print(f"========================================================")
        
        # Aggiorna il percorso di base per il dataset corrente
        data_root = f"./datasets/{dataset_name}_seed{SEED}"
        user_doc_file = os.path.join(data_root, "user_doc.txt")
        item_doc_file = os.path.join(data_root, "item_doc.txt")

        # 1. Genera gli embedding per i documenti Utente
        generate_embeddings_batched(plm_model, tokenizer, device, user_doc_file,
                                    os.path.join(data_root, "user_embeddings"),
                                    os.path.join(data_root, "user_masks"))

        # 2. Genera gli embedding per i documenti Elemento (Item)
        generate_embeddings_batched(plm_model, tokenizer, device, item_doc_file,
                                    os.path.join(data_root, "item_embeddings"),
                                    os.path.join(data_root, "item_masks"))

        print(f"Terminata l'elaborazione per il dataset: {dataset_name}.")
        
    print("\n\n🎉 Pre-calcolo degli embedding completato con successo per tutti i dataset! 🎉")


if __name__ == "__main__":
    main()
