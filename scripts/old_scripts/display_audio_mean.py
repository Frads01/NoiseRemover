import argparse
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import os
import sys
import warnings

# Sopprime gli avvisi di runtime di Pydub/Numpy (opzionale)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Stringa richiesta alla fine del nome del file
REQUIRED_SUFFIX = "_zeromean.wav"

def check_audio_mean(filepath: str):
    """
    Carica un file audio, calcola la sua media e la stampa.

    Args:
        filepath: Il percorso del file audio da analizzare.

    Returns:
        True se l'analisi ha avuto successo, False altrimenti.
    """
    filename = os.path.basename(filepath)
    print(f"--- Analizzo: '{filename}'")

    # --- Validazione preliminare (anche se il chiamante dovrebbe aver già filtrato) ---
    if not filename.endswith(REQUIRED_SUFFIX):
        print(f"  -> Errore: Il file '{filename}' non termina con '{REQUIRED_SUFFIX}'. Questo non dovrebbe accadere se il filtro funziona.")
        return False # Tecnicamente un fallimento per questa funzione

    # --- Caricamento ---
    try:
        # Poiché ci aspettiamo file .wav, di solito non serve specificare il formato
        audio = AudioSegment.from_file(filepath)
    except CouldntDecodeError:
         print(f"  -> Errore: Impossibile decodificare il file audio '{filename}'.")
         print("      Verifica che FFmpeg sia installato (se necessario per WAV non standard) e che il file non sia corrotto.")
         return False
    except FileNotFoundError:
        print(f"  -> Errore: File non trovato '{filepath}'")
        return False
    except Exception as e:
        print(f"  -> Errore durante il caricamento del file '{filename}': {e}")
        return False

    # --- Calcolo Media ---
    try:
        # Ottieni i campioni come array numpy, converti in float64 per precisione
        samples = np.array(audio.get_array_of_samples()).astype(np.float64)

        # Calcola la media
        mean_value = np.mean(samples)

        # Stampa il risultato
        print(f"  -> Media calcolata: {mean_value:.8f}") # Usa più cifre decimali per vedere valori piccoli
        return True

    except Exception as e:
        print(f"  -> Errore durante il calcolo della media per '{filename}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=f"Calcola e stampa la media (DC offset) dei file audio il cui nome termina "
                    f"esattamente con '{REQUIRED_SUFFIX}'. Accetta un file o una directory."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help=f"Percorso del file audio (con suffisso '{REQUIRED_SUFFIX}') o della directory contenente tali file."
    )

    args = parser.parse_args()
    input_path = args.input_path

    processed_count = 0
    error_count = 0
    skipped_count = 0

    if not os.path.exists(input_path):
        print(f"Errore: Il percorso specificato non esiste: '{input_path}'")
        sys.exit(1)

    # --- Modalità File Singolo ---
    if os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        print(f"Modalità file singolo: Controllo '{filename}'")
        if filename.endswith(REQUIRED_SUFFIX):
            if check_audio_mean(input_path):
                processed_count += 1
            else:
                error_count += 1
        else:
            print(f"  -> Ignorato: Il nome del file non termina con '{REQUIRED_SUFFIX}'.")
            skipped_count += 1

    # --- Modalità Directory ---
    elif os.path.isdir(input_path):
        print(f"Modalità directory: Scansione di '{input_path}' per file che terminano con '{REQUIRED_SUFFIX}' (non ricorsivo)")
        found_matching_files = False
        all_entries = []
        try:
            with os.scandir(input_path) as it:
                for entry in it:
                    all_entries.append(entry.name)
            all_entries.sort() # Ordina per output consistente
        except OSError as e:
             print(f"Errore durante la lettura della directory '{input_path}': {e}")
             sys.exit(1)

        if not all_entries:
            print("La directory è vuota.")

        for entry_name in all_entries:
            filepath = os.path.join(input_path, entry_name)

            # Verifica se è un file e se il nome corrisponde
            if os.path.isfile(filepath):
                if entry_name.endswith(REQUIRED_SUFFIX):
                    found_matching_files = True
                    if check_audio_mean(filepath):
                        processed_count += 1
                    else:
                        error_count += 1
                else:
                    # File ma non corrisponde al suffisso
                    if not entry_name.startswith('.'): # Ignora file nascosti comuni
                        print(f"--- Ignoro (suffisso non corrispondente): '{entry_name}'")
                    skipped_count += 1
            else:
                # Non è un file (directory, link simbolico, ecc.)
                if not entry_name.startswith('.'): # Ignora directory nascoste comuni
                    print(f"--- Ignoro (non è un file): '{entry_name}'")
                skipped_count += 1

        if not found_matching_files and skipped_count == 0 and not all_entries:
             pass # Già gestito il caso directory vuota
        elif not found_matching_files and skipped_count > 0:
             print(f"\nNessun file terminante con '{REQUIRED_SUFFIX}' trovato nella directory.")
        elif found_matching_files:
            print("\nScansione directory completata.")


    # --- Percorso non valido ---
    else:
        print(f"Errore: Il percorso specificato '{input_path}' non è né un file né una directory valida.")
        sys.exit(1)

    # --- Riepilogo ---
    print("\n--- Riepilogo ---")
    print(f"File analizzati con successo: {processed_count}")
    print(f"Errori durante l'analisi:     {error_count}")
    print(f"File/Elementi ignorati:       {skipped_count}")

    if error_count > 0:
        print("\nAttenzione: Si sono verificati errori durante l'analisi di alcuni file.")
        sys.exit(1)
    elif processed_count == 0 and skipped_count == 0 and error_count == 0:
        # Caso strano, forse file singolo non valido o directory vuota già segnalata
         if os.path.isfile(input_path): # Se era un file singolo, è stato skippato
              print("\nNessun file analizzato (il file fornito non corrispondeva al criterio).")
         # else: # Directory vuota o senza file validi (già segnalato)
         #    pass
         sys.exit(0) # Nessun errore, ma nessun lavoro fatto
    else:
        print("\nAnalisi completata.")
        sys.exit(0)


if __name__ == "__main__":
    main()