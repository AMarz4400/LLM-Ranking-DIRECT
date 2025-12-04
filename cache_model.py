from transformers import AutoTokenizer, T5EncoderModel
import os

model_name = "google/t5gemma-2b-2b-prefixlm-it"

# cartella locale dove salvare il modello
cache_directory = "./cache"

print(f"Inizio il download del modello '{model_name}' nella cartella locale '{cache_directory}'")


try:

    os.makedirs(cache_directory, exist_ok=True)

    # argomento 'cache_dir' per specificare dove salvare
    print("Download del tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_directory)
    tokenizer.save_pretrained(cache_directory) # Salva anche altri file necessari
    print(f"Tokenizer salvato in '{cache_directory}'.")


    print("Download del modello (potrebbe richiedere tempo)...")
    model = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_directory)
    model.save_pretrained(cache_directory) # Salva il modello e la config
    print(f"Modello salvato in '{cache_directory}'.")

    print(f"\nOperazione completata. I file sono ora nella cartella '{cache_directory}'.")

except Exception as e:
    print(f"\nSi è verificato un errore: {e}")
    print("Verifica che:")
    print("1. Il nome del modello sia corretto.")
    print("2. Tu sia autenticato (hai eseguito 'huggingface-cli login').")
    print("3. Il nodo di login abbia accesso a internet.")
