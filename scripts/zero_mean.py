import argparse
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import os
import sys
import warnings

# Sopprime gli avvisi di runtime di Pydub/Numpy (opzionale)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Tolleranza per considerare la media "sufficientemente vicina" a zero
ZERO_MEAN_TOLERANCE = 1e-9 # Potrebbe essere necessario aggiustarla leggermente per la verifica post-salvataggio

# Estensioni file considerate audio (aggiungi se necessario)
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma'}

def make_audio_zero_mean(filepath: str):
    """
    Controlla se un file audio ha media nulla. Se non ce l'ha,
    crea un nuovo file .wav a media nulla con suffisso _zeromean.

    Args:
        filepath: Il percorso del file audio da processare.

    Returns:
        True se l'operazione ha avuto successo o non era necessaria,
        False se si è verificato un errore durante l'elaborazione di questo file.
    """
    print(f"\nAnalisi del file: '{os.path.basename(filepath)}'...")
    original_mean = None # Per tener traccia se abbiamo calcolato la media originale
    output_file_created = False # Flag per sapere se abbiamo creato un file

    try:
        # Determina il formato originale per il caricamento
        file_ext = os.path.splitext(filepath)[1].lower()
        if not file_ext:
             print(f"  Errore: Impossibile determinare l'estensione del file da '{filepath}'")
             return False

        # Carica il file audio usando pydub
        try:
            if file_ext != '.wav':
                 audio = AudioSegment.from_file(filepath, format=file_ext[1:]) # Rimuovi il punto
            else:
                 audio = AudioSegment.from_file(filepath)
        except CouldntDecodeError:
             print(f"  Errore: Impossibile decodificare il file audio '{filepath}'.")
             print("  Verifica che FFmpeg sia installato e accessibile nel PATH,")
             print("  e che il file non sia corrotto o in un formato non supportato.")
             return False
        except Exception as e:
            print(f"  Errore durante il caricamento del file '{filepath}': {e}")
            return False

        # Ottieni i campioni audio come array numpy
        samples = np.array(audio.get_array_of_samples()).astype(np.float64)

        # Calcola la media dei campioni
        mean_value = np.mean(samples)
        original_mean = mean_value # Memorizza la media originale

        print(f"  - Media attuale: {mean_value:.6f}")

        # Controlla se la media è già vicina a zero
        if abs(mean_value) < ZERO_MEAN_TOLERANCE:
            print(f"  - Il file ha già una media vicina a zero. Nessuna modifica necessaria.")
            # Rimuovi eventuali file _zeromean obsoleti se l'originale è ora a media zero
            base_name_orig, _ = os.path.splitext(os.path.basename(filepath))
            potential_zeromean_output = os.path.join(os.path.dirname(filepath), f"{base_name_orig}_zeromean.wav")
            if os.path.exists(potential_zeromean_output):
                 try:
                     os.remove(potential_zeromean_output)
                     print(f"  - Rimosso file obsoleto: '{os.path.basename(potential_zeromean_output)}'")
                 except Exception as e:
                     print(f"  - Attenzione: impossibile rimuovere file obsoleto '{os.path.basename(potential_zeromean_output)}': {e}")
            return True # Operazione "riuscita"

        print(f"  - La media non è zero. Procedo con la creazione del file corretto...")

        # Rimuovi l'offset DC (rendi la media zero)
        centered_samples = samples - mean_value

        # --- Verifica e Clipping ---
        # ... (codice di clipping invariato) ...
        if audio.sample_width == 1:
            min_val, max_val = 0, 255
            dtype = np.uint8
            centered_samples += 128 # Sposta per unsigned
        elif audio.sample_width == 2:
            min_val, max_val = -32768, 32767
            dtype = np.int16
        elif audio.sample_width == 4:
            min_val, max_val = -2147483648, 2147483647
            dtype = np.int32
        else:
            print(f"  Errore: Sample width non supportato: {audio.sample_width} per {filepath}")
            return False

        clipped_samples = np.clip(centered_samples, min_val, max_val)
        corrected_samples_int = clipped_samples.astype(dtype)

        # Crea un nuovo oggetto AudioSegment con i dati corretti
        modified_audio = AudioSegment(
            data=corrected_samples_int.tobytes(),
            sample_width=audio.sample_width,
            frame_rate=audio.frame_rate,
            channels=audio.channels
        )

        # --- Nuovo nome file ---
        base_name, _ = os.path.splitext(os.path.basename(filepath))
        dir_name = os.path.dirname(filepath)
        output_filename = f"{base_name}_zeromean.wav"
        output_filepath = os.path.join(dir_name, output_filename)

        # --- Controlla se esiste per messaggio informativo ---
        overwrite_message = ""
        if os.path.exists(output_filepath):
            overwrite_message = " (sovrascrivendo file esistente)"

        # Salva il file modificato con il nuovo nome e formato WAV
        try:
            print(f"  - Salvo il file modificato come '{output_filename}'{overwrite_message}...")
            modified_audio.export(output_filepath, format="wav")
            print(f"  - File '{output_filename}' creato/aggiornato con successo.")
            output_file_created = True # Imposta il flag

        except Exception as e:
            print(f"  Errore durante il salvataggio del file modificato '{output_filepath}': {e}")
            return False # Esce se il salvataggio fallisce

        # --- MODIFICA: Verifica la media del file appena creato ---
        if output_file_created:
            try:
                print(f"  - Verifico la media del file di output '{output_filename}'...")
                # Ricarica il file appena salvato
                output_audio = AudioSegment.from_file(output_filepath)
                # Calcola la sua media
                output_samples = np.array(output_audio.get_array_of_samples()).astype(np.float64)
                output_mean = np.mean(output_samples)
                print(f"  - Media verificata del file di output: {output_mean:.6f}") # Stampa la media effettiva

                # Controllo aggiuntivo (opzionale ma utile)
                if abs(output_mean) >= ZERO_MEAN_TOLERANCE:
                     print(f"  - ATTENZIONE: La media del file di output ({output_mean:.6f}) non è vicina a zero come atteso (tolleranza: {ZERO_MEAN_TOLERANCE}). Possibili problemi di precisione nel salvataggio/ricaricamento.")

            except CouldntDecodeError:
                print(f"  - Errore di verifica: Impossibile decodificare il file di output '{output_filename}' appena creato.")
            except Exception as verify_e:
                # Cattura altri errori durante la verifica
                print(f"  - Attenzione: Impossibile verificare la media del file di output '{output_filename}': {verify_e}")
        # --- FINE MODIFICA ---

        return True # Ritorna True perché il salvataggio (e la verifica opzionale) sono andati a buon fine

    except Exception as e:
        # Cattura errori generali durante l'elaborazione
        print(f"  Errore imprevisto durante l'elaborazione di '{filepath}': {e}")
        # Stampa la media originale se è stata calcolata, per debug
        if original_mean is not None:
             print(f"  (Media originale calcolata prima dell'errore: {original_mean:.6f})")
        return False


# --- La funzione main() rimane invariata rispetto alla versione precedente ---
def main():
    parser = argparse.ArgumentParser(
        description="Controlla la media dei file audio e crea una versione a media nulla "
                    "con suffisso '_zeromean.wav'. Accetta un file o una directory. "
                    "I file che contengono già '_zeromean' nel nome vengono ignorati. "
                    "I file di output esistenti vengono sovrascritti."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Percorso del file audio o della directory contenente i file audio da processare."
    )

    args = parser.parse_args()
    input_path = args.input_path

    success_count = 0
    fail_count = 0
    skipped_count = 0 # File non audio nella directory
    skipped_zeromean_count = 0 # File che contengono già _zeromean

    if not os.path.exists(input_path):
        print(f"Errore: Il percorso specificato non esiste: '{input_path}'")
        sys.exit(1)

    if os.path.isfile(input_path):
        print(f"Modalità file singolo: Controllo '{input_path}'")
        base_name = os.path.basename(input_path)

        if "_zeromean" in base_name:
            print(f"  - Il nome del file '{base_name}' contiene '_zeromean'. Esclusione dall'analisi.")
            skipped_zeromean_count += 1
        else:
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext in AUDIO_EXTENSIONS:
                print(f"Elaborazione file: '{base_name}'")
                if make_audio_zero_mean(input_path):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                 print(f"Errore: Il file '{input_path}' non sembra essere un file audio supportato (estensione non in {AUDIO_EXTENSIONS}).")
                 sys.exit(1) # Esce se il file singolo non è audio valido

    elif os.path.isdir(input_path):
        print(f"Modalità directory: Elaborazione dei file audio in '{input_path}' (non ricorsivo)")
        found_audio_files = False
        all_files = sorted(os.listdir(input_path)) # Ordina per un output più consistente

        for filename in all_files:
            filepath = os.path.join(input_path, filename)
            if os.path.isfile(filepath): # Assicurati che sia un file

                if "_zeromean" in filename:
                    # Non serve \n qui perché make_audio_zero_mean lo mette all'inizio
                    print(f"Ignoro file (contiene '_zeromean'): '{filename}'")
                    skipped_zeromean_count += 1
                    continue # Passa al prossimo file

                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in AUDIO_EXTENSIONS:
                    found_audio_files = True
                    # La funzione make_audio_zero_mean ora stampa tutto, inclusa la media di output
                    if make_audio_zero_mean(filepath):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    skipped_count += 1
            else:
                 skipped_count += 1 # Conta le subdir come skippate

        if not found_audio_files and skipped_zeromean_count == len([f for f in all_files if os.path.isfile(os.path.join(input_path, f))]):
             print(f"\nNessun file audio idoneo all'elaborazione trovato nella directory (solo file '_zeromean', non audio o sotto-directory).")
        elif not found_audio_files and skipped_zeromean_count == 0 and skipped_count > 0:
             print(f"\nNessun file audio trovato nella directory specificata (solo file non audio o sotto-directory).")
        elif not found_audio_files and skipped_zeromean_count == 0 and skipped_count == 0:
             print(f"\nLa directory specificata è vuota.")

        else:
             print(f"\nElaborazione directory completata.")

    else:
        print(f"Errore: Il percorso specificato '{input_path}' non è né un file né una directory valida.")
        sys.exit(1)

    print("\n--- Riepilogo ---")
    print(f"File processati con successo (o già a media nulla): {success_count}")
    print(f"File con errori durante l'elaborazione:             {fail_count}")
    if skipped_count > 0:
         print(f"Elementi ignorati (non audio o sotto-directory):  {skipped_count}")

    if fail_count > 0:
        print("\nAttenzione: Si sono verificati errori durante l'elaborazione di alcuni file.")
        sys.exit(1)


if __name__ == "__main__":
    main()