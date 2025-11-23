from transformers import AutoTokenizer, T5EncoderModel
import os # NUOVO: Importiamo il modulo 'os' per creare la cartella

# Il nome del modello che il tuo script principale 'train.py' utilizza.
model_name = "google/t5gemma-2b-2b-prefixlm-it"

# NUOVO: Definiamo la cartella locale dove salvare il modello, la stessa usata dal tuo codice
cache_directory = "./cache"

print(f"Inizio il download del modello '{model_name}' nella cartella locale '{cache_directory}'")
print("Questo va eseguito una sola volta sul nodo di login con accesso a internet.")

try:
    # NUOVO: Assicuriamoci che la cartella esista prima di scaricare
    os.makedirs(cache_directory, exist_ok=True)

    # MODIFICATO: Aggiungiamo l'argomento 'cache_dir' per specificare dove salvare
    print("Download del tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
    tokenizer.save_pretrained(cache_directory) # Salva anche altri file necessari
    print(f"Tokenizer salvato in '{cache_directory}'.")

    # MODIFICATO: Aggiungiamo l'argomento 'cache_dir' anche qui
    print("Download del modello (potrebbe richiedere tempo)...")
    model = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_directory)
    model.save_pretrained(cache_directory) # Salva il modello e la config
    print(f"Modello salvato in '{cache_directory}'.")

    print(f"\nOperazione completata. I file sono ora nella cartella '{cache_directory}'.")
    print("Il tuo script di training ora li troverà localmente.")

except Exception as e:
    print(f"\nSi è verificato un errore: {e}")
    print("Verifica che:")
    print("1. Il nome del modello sia corretto.")
    print("2. Tu sia autenticato (hai eseguito 'huggingface-cli login').")
    print("3. Il nodo di login abbia accesso a internet.")
