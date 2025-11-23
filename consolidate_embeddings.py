import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURAZIONE ---
# Questa variabile non è più usata direttamente da main(), ma la lista datasets sì.
# DATASET_NAME = "reviews_Toys_and_Games_5" 
SEED = 242

datasets = ["reviews_Clothing_Shoes_and_Jewelry_5"]
# --------------------

def consolidate_files(source_directory, output_file_path):
    """
    Scansiona una directory di file .npy numerati, li carica e li consolida
    in un'unica grande matrice NumPy, salvandola nel file di output.
    """
    print(f"\n--- Inizio consolidamento per: {source_directory} ---")

    if not os.path.isdir(source_directory):
        # Usiamo un avviso invece di un'eccezione per continuare con gli altri dataset
        print(f"ATTENZIONE: La directory sorgente non è stata trovata: {source_directory}. Salto.")
        return

    # Trova tutti i file .npy e determina il numero totale
    npy_files = [f for f in os.listdir(source_directory) if f.endswith('.npy')]
    
    # Ordina i file per numero per garantire la sequenza corretta
    # Assumiamo che i nomi siano "0.npy", "1.npy", etc.
    num_files = len(npy_files)
    if num_files == 0:
        print(f"Nessun file .npy trovato in {source_directory}. Salto.")
        return

    print(f"Trovati {num_files} file .npy da consolidare.")

    # Carica il primo file (0.npy) per determinare la forma e il tipo di dati
    first_file_path = os.path.join(source_directory, "0.npy")
    try:
        sample_array = np.load(first_file_path)
    except FileNotFoundError:
        print(f"Errore: File iniziale 0.npy non trovato in {source_directory}. Salto.")
        return
        
    shape = sample_array.shape
    dtype = sample_array.dtype

    # Crea la grande matrice vuota che conterrà tutti i dati
    consolidated_array = np.zeros((num_files, *shape), dtype=dtype)
    print(f"Creata matrice consolidata con forma: {consolidated_array.shape} e tipo: {dtype}")

    # Itera su tutti gli indici attesi (da 0 a num_files-1)
    for i in tqdm(range(num_files), desc=f"Consolidando {os.path.basename(source_directory)}"):
        file_path = os.path.join(source_directory, f"{i}.npy")
        try:
            consolidated_array[i] = np.load(file_path)
        except FileNotFoundError:
             # Questo gestisce il caso in cui ci sia un buco nella numerazione
             print(f"ATTENZIONE: File {i}.npy non trovato. Potrebbe esserci un problema di numerazione. Continuo.")
             # Potresti voler gestire in modo più rigoroso se l'integrità dei dati è critica
             # Per ora, usiamo la riga di zeri predefinita per l'array in questa posizione
             continue


    # Salva la matrice grande su disco
    print(f"Salvataggio del file consolidato in: {output_file_path}")
    # Assicurati che la directory di output esista prima di salvare
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    np.save(output_file_path, consolidated_array)
    print("Salvataggio completato. ✅")
    print("-" * 50)


def main():
    print("--- Inizio l'elaborazione di tutti i dataset ---")

    # Ciclo su tutti i nomi di dataset definiti in 'datasets'
    for dataset_name in datasets:
        print(f"\n========================================================")
        print(f"Inizio il consolidamento per il dataset: {dataset_name}")
        print(f"========================================================")
        
        # Definisci il percorso principale dei dati per il dataset corrente
        data_root = f"./datasets/{dataset_name}_seed{SEED}"

        # Definisci i percorsi Source e Output che USANO il data_root CORRENTE
        paths_to_process = [
            # Source Directory,                             Output File Path
            (os.path.join(data_root, "user_embeddings"),      os.path.join(data_root, "all_user_embeddings.npy")),
            (os.path.join(data_root, "user_masks"),           os.path.join(data_root, "all_user_masks.npy")),
            (os.path.join(data_root, "item_embeddings"),      os.path.join(data_root, "all_item_embeddings.npy")),
            (os.path.join(data_root, "item_masks"),           os.path.join(data_root, "all_item_masks.npy")),
        ]

        # Esegui il consolidamento per ogni set di dati del dataset corrente
        for source_dir, output_file in paths_to_process:
            consolidate_files(source_dir, output_file)

        print(f"Terminato il consolidamento per il dataset: {dataset_name}.")

    print("\n\n🎉 Consolidamento di tutti gli embedding completato con successo per tutti i dataset! 🎉")
    print("Ora puoi procedere con la modifica di 'datas/dataset.py'.")



if __name__ == "__main__":
    main()
