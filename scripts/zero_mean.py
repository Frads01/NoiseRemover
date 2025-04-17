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
ZERO_MEAN_TOLERANCE = 1e-9

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

    try:
        # Determina il formato originale per il caricamento
        # Non serve più per il salvataggio, ma utile per pydub
        file_ext = os.path.splitext(filepath)[1].lower()
        if not file_ext:
             print(f"  Errore: Impossibile determinare l'estensione del file da '{filepath}'")
             return False

        # Carica il file audio usando pydub
        # Specifica il formato se non è wav per sicurezza, altrimenti lascia che pydub provi
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

        print(f"  - Media attuale: {mean_value:.6f}")

        # Controlla se la media è già vicina a zero
        if abs(mean_value) < ZERO_MEAN_TOLERANCE:
            print(f"  - Il file ha già una media vicina a zero. Nessuna modifica necessaria.")
            return True # Operazione "riuscita" perché non c'era nulla da fare

        print(f"  - La media non è zero. Procedo con la creazione del file corretto...")

        # Rimuovi l'offset DC (rendi la media zero)
        centered_samples = samples - mean_value

        # --- Verifica e Clipping ---
        if audio.sample_width == 1:
            min_val, max_val = 0, 255
            dtype = np.uint8
            centered_samples += 128 # Sposta per unsigned
        elif audio.sample_width == 2:
            min_val, max_val = -32768, 32767
            dtype = np.int16
        elif audio.sample_width == 4: # pydub spesso usa 32bit anche per 24bit source
            min_val, max_val = -2147483648, 2147483647
            dtype = np.int32
        # Aggiungere sample_width == 3 se necessario, ma pydub potrebbe convertirlo a 4
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

        # Salva il file modificato con il nuovo nome e formato WAV
        try:
            print(f"  - Salvo il file modificato come '{output_filename}'...")
            # Esporta sempre come WAV
            modified_audio.export(output_filepath, format="wav")
            print(f"  - File '{output_filename}' creato con successo.")
            return True

        except Exception as e:
            print(f"  Errore durante il salvataggio del file modificato '{output_filepath}': {e}")
            return False

    except Exception as e:
        print(f"  Errore imprevisto durante l'elaborazione di '{filepath}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Controlla la media dei file audio e crea una versione a media nulla "
                    "con suffisso '_zeromean.wav'. Accetta un file o una directory."
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

    if not os.path.exists(input_path):
        print(f"Errore: Il percorso specificato non esiste: '{input_path}'")
        sys.exit(1)

    if os.path.isfile(input_path):
        print(f"Modalità file singolo: Elaborazione di '{input_path}'")
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in AUDIO_EXTENSIONS:
            if make_audio_zero_mean(input_path):
                success_count += 1
            else:
                fail_count += 1
        else:
             print(f"Errore: Il file '{input_path}' non sembra essere un file audio supportato (estensione non in {AUDIO_EXTENSIONS}).")
             sys.exit(1)

    elif os.path.isdir(input_path):
        print(f"Modalità directory: Elaborazione dei file audio in '{input_path}' (non ricorsivo)")
        found_audio_files = False
        for filename in os.listdir(input_path):
            filepath = os.path.join(input_path, filename)
            if os.path.isfile(filepath): # Assicurati che sia un file, non una sotto-directory
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in AUDIO_EXTENSIONS:
                    found_audio_files = True
                    if make_audio_zero_mean(filepath):
                        # La funzione stessa indica se ha avuto successo o se non era necessario
                        # Potremmo voler distinguere qui, ma per ora contiamo entrambi come "ok"
                        success_count +=1 # Conta successo anche se non ha modificato
                    else:
                        fail_count += 1
                else:
                    # print(f"  Ignoro file non audio: '{filename}'") # Opzionale: troppo verboso?
                    skipped_count += 1
            else:
                 # print(f"  Ignoro sotto-directory: '{filename}'") # Opzionale
                 skipped_count += 1 # Conta anche le subdir come skippate

        if not found_audio_files:
             print("\nNessun file audio trovato nella directory specificata.")
        else:
             print(f"\nElaborazione completata.")

    else:
        print(f"Errore: Il percorso specificato '{input_path}' non è né un file né una directory valida.")
        sys.exit(1)

    print("--- Riepilogo ---")
    print(f"File processati con successo (o già a media nulla): {success_count}")
    print(f"File con errori durante l'elaborazione:             {fail_count}")
    if skipped_count > 0:
         print(f"Elementi ignorati (non audio o sotto-directory):  {skipped_count}")

    if fail_count > 0:
        print("\nAttenzione: Si sono verificati errori durante l'elaborazione di alcuni file.")
        sys.exit(1) # Esce con codice di errore se qualcosa è andato storto


if __name__ == "__main__":
    main()