import argparse
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import os
import sys
import warnings
# import math # Non più necessario

# Sopprime gli avvisi di runtime di Pydub/Numpy (opzionale)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Soglia di media assoluta desiderata per terminare il ciclo
TARGET_MEAN_THRESHOLD = 0.1

# Estensioni file considerate audio (aggiungi se necessario)
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma'}

def make_audio_zero_mean(filepath: str):
    """
    Controlla se un file audio ha media assoluta < TARGET_MEAN_THRESHOLD.
    Se non ce l'ha, crea iterativamente un nuovo file .wav a media nulla
    con suffisso _zeromean finché la media assoluta del file di output
    non è strettamente minore di TARGET_MEAN_THRESHOLD.

    Args:
        filepath: Il percorso del file audio originale da processare.

    Returns:
        True se l'operazione ha avuto successo (file originale già ok o
             file _zeromean creato con successo entro la soglia),
        False se si è verificato un errore durante l'elaborazione.
    """
    print(f"\nAnalisi del file: '{os.path.basename(filepath)}'...")

    # --- Preparazione iniziale ---
    original_filepath = filepath # Conserva il percorso originale
    current_filepath_to_process = original_filepath
    base_name_orig, _ = os.path.splitext(os.path.basename(original_filepath))
    dir_name = os.path.dirname(original_filepath)
    output_filename = f"{base_name_orig}_zeromean.wav"
    output_filepath = os.path.join(dir_name, output_filename)

    iteration = 0
    # Ciclo potenzialmente infinito finché la condizione non è soddisfatta
    while True:
        iteration += 1
        print(f"--- Iterazione {iteration} per '{os.path.basename(original_filepath)}' ---")
        if iteration > 1:
            print(f"  - Input per questa iterazione: '{os.path.basename(current_filepath_to_process)}'")
        else:
            print(f"  - Input iniziale: '{os.path.basename(current_filepath_to_process)}'")


        # --- Caricamento e Calcolo Media Input Corrente ---
        try:
            file_ext = os.path.splitext(current_filepath_to_process)[1].lower()
            if not file_ext:
                 print(f"  Errore (Iter. {iteration}): Impossibile determinare l'estensione del file da '{current_filepath_to_process}'")
                 return False # Errore fatale

            # Caricamento audio
            audio = None
            try:
                # Dopo la prima iterazione, l'input sarà sempre il file .wav di output
                 if file_ext != '.wav' or iteration == 1:
                    # Se è la prima iterazione ma il file è .wav, non specificare format
                    if file_ext == '.wav':
                         audio = AudioSegment.from_file(current_filepath_to_process)
                    else:
                         audio = AudioSegment.from_file(current_filepath_to_process, format=file_ext[1:])
                 else: # Iterazioni successive, input è _zeromean.wav
                    audio = AudioSegment.from_file(current_filepath_to_process) # È sicuramente wav
            except CouldntDecodeError:
                 print(f"  Errore (Iter. {iteration}): Impossibile decodificare il file audio '{os.path.basename(current_filepath_to_process)}'.")
                 if iteration == 1: # Se fallisce al primo colpo, dare suggerimenti
                      print("      Verifica che FFmpeg sia installato e accessibile nel PATH,")
                      print("      e che il file non sia corrotto o in un formato non supportato.")
                 return False # Errore fatale
            except FileNotFoundError:
                print(f"  Errore (Iter. {iteration}): File non trovato '{current_filepath_to_process}' (Potrebbe accadere se il salvataggio precedente è fallito o il file è stato rimosso)")
                return False
            except Exception as e:
                print(f"  Errore (Iter. {iteration}): durante il caricamento del file '{os.path.basename(current_filepath_to_process)}': {e}")
                return False

            # Ottieni i campioni audio come array numpy
            samples = np.array(audio.get_array_of_samples()).astype(np.float64)

            # Calcola la media dei campioni dell'input corrente
            current_mean = np.mean(samples)
            print(f"  - Media input (Iter. {iteration}): {current_mean:.8f}")

            # --- Controllo Media Input Corrente (per eventuale uscita anticipata) ---
            # Se l'input CORRENTE soddisfa GIÀ la condizione, abbiamo finito.
            if abs(current_mean) < TARGET_MEAN_THRESHOLD:
                print(f"  - La media assoluta dell'input corrente ({abs(current_mean):.8f}) è già minore di {TARGET_MEAN_THRESHOLD}.")
                # Se è la prima iterazione, il file originale era già OK.
                if iteration == 1:
                    print(f"  - Il file originale '{os.path.basename(original_filepath)}' non richiede modifiche.")
                    # Rimuovi eventuali file _zeromean obsoleti SE ESISTONO
                    if os.path.exists(output_filepath):
                         try:
                             os.remove(output_filepath)
                             print(f"  - Rimosso file obsoleto esistente: '{os.path.basename(output_filepath)}'")
                         except Exception as e:
                             print(f"  - Attenzione: impossibile rimuovere file obsoleto '{os.path.basename(output_filepath)}': {e}")
                    return True # Successo, nessuna azione necessaria
                else:
                    # Se siamo in un'iterazione successiva, significa che l'output
                    # dell'iterazione PRECEDENTE (ora input) era già OK.
                    # Non dovrebbe succedere se la verifica post-salvataggio funziona,
                    # ma è una sicurezza. Il file output_filepath è quello corretto.
                    print(f"  - Il file '{os.path.basename(output_filepath)}' generato nell'iterazione precedente è confermato entro la soglia.")
                    return True # Successo, il file _zeromean.wav è valido

            # Se siamo qui, la media dell'input corrente NON soddisfa la soglia.
            # Dobbiamo (ri)generare il file.
            print(f"  - La media assoluta non è inferiore a {TARGET_MEAN_THRESHOLD}. Procedo con la (ri)creazione del file corretto...")

            # --- Centratura, Clipping e Creazione Nuovo AudioSegment ---
            centered_samples = samples - current_mean

            # Gestione sample width e clipping (invariata)
            sample_width = audio.sample_width
            dtype = None
            min_val, max_val = 0, 0
            if sample_width == 1: # 8-bit unsigned
                min_val, max_val = 0, 255
                dtype = np.uint8
                centered_samples += 128 # Shift per unsigned dopo la centratura
            elif sample_width == 2: # 16-bit signed
                min_val, max_val = -32768, 32767
                dtype = np.int16
            elif sample_width == 3: # 24-bit signed
                 print(f"  Attenzione: La gestione dei 24-bit potrebbe essere imprecisa. Si procede come 32-bit.")
                 min_val, max_val = -2147483648, 2147483647
                 dtype = np.int32
                 sample_width = 4 # Trattiamo come 32bit per AudioSegment
            elif sample_width == 4: # 32-bit signed
                min_val, max_val = -2147483648, 2147483647
                dtype = np.int32
            else:
                print(f"  Errore (Iter. {iteration}): Sample width non supportato: {audio.sample_width} ({audio.sample_width * 8}-bit) per {os.path.basename(current_filepath_to_process)}")
                return False

            clipped_samples = np.clip(centered_samples, min_val, max_val)
            num_clipped = np.sum((centered_samples < min_val) | (centered_samples > max_val))
            if num_clipped > 0:
                print(f"  - Attenzione (Iter. {iteration}): {num_clipped} campioni sono stati clippati durante la centratura.")

            # Assicurati che il dtype sia corretto prima di tobytes()
            if dtype is None: # Salvaguardia
                 print(f"  Errore interno: dtype non impostato per sample_width {audio.sample_width}")
                 return False
            corrected_samples_int = clipped_samples.astype(dtype)


            # Crea un nuovo oggetto AudioSegment
            try:
                modified_audio = AudioSegment(
                    data=corrected_samples_int.tobytes(),
                    # Usa la sample_width corretta (eventualmente aggiustata per 24bit)
                    sample_width=sample_width,
                    frame_rate=audio.frame_rate,
                    channels=audio.channels
                )
            except Exception as e:
                print(f"  Errore (Iter. {iteration}): durante la creazione del nuovo AudioSegment: {e}")
                return False

            # --- Salvataggio File Modificato ---
            overwrite_message = ""
            if os.path.exists(output_filepath):
                overwrite_message = " (sovrascrivendo file esistente)"

            try:
                print(f"  - Salvo il file modificato (tentativo {iteration}) come '{output_filename}'{overwrite_message}...")
                modified_audio.export(output_filepath, format="wav")
                print(f"  - File '{output_filename}' creato/aggiornato.")

            except Exception as e:
                print(f"  Errore (Iter. {iteration}): durante il salvataggio del file modificato '{output_filepath}': {e}")
                # Non continuare se il salvataggio fallisce
                return False

            # --- Verifica Media del File Appena Salvato ---
            try:
                print(f"  - Verifico la media del file di output '{output_filename}'...")
                # Pausa brevissima per dare tempo al FS (potrebbe non essere necessaria)
                # import time
                # time.sleep(0.05)
                output_audio = AudioSegment.from_file(output_filepath)
                output_samples = np.array(output_audio.get_array_of_samples()).astype(np.float64)
                output_mean = np.mean(output_samples)

                print(f"  - Media verificata del file di output (precisa): {output_mean:.8f}")
                print(f"  - Media assoluta: {abs(output_mean):.8f}")

                # --- CONDIZIONE DI USCITA DAL CICLO ---
                if abs(output_mean) < TARGET_MEAN_THRESHOLD:
                     print(f"  - SUCCESSO: La media assoluta del file di output è minore di {TARGET_MEAN_THRESHOLD} dopo {iteration} iterazioni.")
                     # Il file 'output_filepath' è quello corretto.
                     return True # Obiettivo raggiunto! Esce dal ciclo e dalla funzione.
                else:
                     print(f"  - La media assoluta del file di output ({abs(output_mean):.8f}) non è ancora minore di {TARGET_MEAN_THRESHOLD}.")
                     print(f"  - Procedo con l'iterazione successiva usando '{os.path.basename(output_filepath)}' come input.")
                     # Prepara per la prossima iterazione usando l'output come nuovo input
                     current_filepath_to_process = output_filepath
                     # Il ciclo while True continuerà automaticamente

            except FileNotFoundError:
                 print(f"  - Errore di verifica (Iter. {iteration}): Impossibile trovare il file '{output_filename}' appena salvato per la verifica.")
                 return False # Errore critico
            except CouldntDecodeError:
                print(f"  - Errore di verifica (Iter. {iteration}): Impossibile decodificare il file di output '{output_filename}' appena creato.")
                return False # Non possiamo verificare, errore
            except Exception as verify_e:
                print(f"  - Errore durante la verifica post-salvataggio (Iter. {iteration}): {verify_e}")
                return False # Non possiamo verificare, errore

        # --- Fine del blocco try principale per caricamento/elaborazione ---
        except Exception as e:
            # Cattura errori generali imprevisti durante l'elaborazione di una iterazione
            print(f"  Errore imprevisto durante l'iterazione {iteration} per '{os.path.basename(original_filepath)}': {e}")
            # Stampa stack trace per debug, se necessario
            # import traceback
            # traceback.print_exc()
            return False # Errore fatale per questo file

    # Questa parte non dovrebbe mai essere raggiunta a causa del `while True`
    # e dei `return` all'interno.
    # return False # In caso di uscita imprevista dal ciclo (non dovrebbe accadere)


# --- La funzione main() rimane invariata rispetto alla versione precedente ---
def main():
    parser = argparse.ArgumentParser(
        description=f"Controlla la media dei file audio e crea una versione a media nulla "
                    f"con suffisso '_zeromean.wav'. Tenta iterativamente finché la media "
                    f"assoluta dell'output non è < {TARGET_MEAN_THRESHOLD}. "
                    "Accetta un file o una directory. "
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
        all_files = []
        try:
            with os.scandir(input_path) as it:
                for entry in it:
                    all_files.append(entry.name)
            all_files.sort()
        except OSError as e:
             print(f"Errore durante la lettura della directory '{input_path}': {e}")
             sys.exit(1)


        for filename in all_files:
            filepath = os.path.join(input_path, filename)
            if os.path.isfile(filepath):

                if "_zeromean" in filename:
                    print(f"Ignoro file (contiene '_zeromean'): '{filename}'")
                    skipped_zeromean_count += 1
                    continue

                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in AUDIO_EXTENSIONS:
                    found_audio_files = True
                    if make_audio_zero_mean(filepath):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                     if not filename.startswith('.'):
                         print(f"Ignoro file (non audio): '{filename}'")
                     skipped_count += 1
            else:
                 if not filename.startswith('.'):
                    print(f"Ignoro elemento (directory o altro): '{filename}'")
                 skipped_count += 1

        # Messaggi riepilogativi per la directory (invariati)
        if not found_audio_files and skipped_zeromean_count == 0 and skipped_count == 0 and not any(os.path.isfile(os.path.join(input_path, f)) or os.path.isdir(os.path.join(input_path, f)) for f in all_files) :
             print(f"\nLa directory specificata '{input_path}' è vuota.")
        elif not found_audio_files and skipped_zeromean_count > 0 and skipped_count == 0 :
             print(f"\nNessun file audio idoneo all'elaborazione trovato nella directory (solo file '_zeromean' presenti).")
        elif not found_audio_files and skipped_zeromean_count == 0 and skipped_count > 0:
             print(f"\nNessun file audio idoneo all'elaborazione trovato nella directory (solo file non audio, directory o altro).")
        elif not found_audio_files and skipped_zeromean_count > 0 and skipped_count > 0:
             print(f"\nNessun file audio idoneo all'elaborazione trovato nella directory (presenti solo file '_zeromean', non audio, directory o altro).")
        elif found_audio_files:
             print(f"\nElaborazione directory completata.")


    else:
        print(f"Errore: Il percorso specificato '{input_path}' non è né un file né una directory valida.")
        sys.exit(1)

    # Riepilogo finale (aggiornato il testo per 'fail_count')
    print("\n--- Riepilogo ---")
    print(f"File processati con successo (raggiunta soglia < {TARGET_MEAN_THRESHOLD}): {success_count}")
    print(f"File falliti (errore durante l'elaborazione):             {fail_count}")
    if skipped_zeromean_count > 0:
        print(f"File ignorati (contenenti '_zeromean'):                 {skipped_zeromean_count}")
    if skipped_count > 0:
         print(f"Elementi ignorati (non audio, dir, altro):              {skipped_count}")


    if fail_count > 0:
        print("\nAttenzione: Si sono verificati errori durante l'elaborazione di alcuni file.")
        sys.exit(1)
    else:
        print("\nElaborazione completata senza errori critici.")
        sys.exit(0)


if __name__ == "__main__":
    main()