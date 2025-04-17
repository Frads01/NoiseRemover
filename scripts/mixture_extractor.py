import os
import subprocess
import argparse
import sys
from pathlib import Path

def check_ffmpeg():
    """Verifica se ffmpeg è installato e accessibile nel PATH."""
    try:
        # Esegui 'ffmpeg -version' e sopprimi l'output
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("INFO: ffmpeg trovato.")
        return True
    except FileNotFoundError:
        print("ERRORE: ffmpeg non trovato. Assicurati che sia installato e nel PATH di sistema.")
        return False
    except subprocess.CalledProcessError:
        # Questo non dovrebbe accadere con '-version', ma per sicurezza
        print("ERRORE: C'è stato un problema nell'eseguire ffmpeg.")
        return False

def extract_mixture(mp4_path: Path, output_dir: Path):
    """
    Estrae la traccia audio 0 (mixture) da un file MP4 e la salva come WAV.

    Args:
        mp4_path (Path): Percorso del file MP4 di input.
        output_dir (Path): Cartella dove salvare il file WAV di output.
    """
    if not mp4_path.is_file() or mp4_path.suffix.lower() != '.mp4':
        print(f"WARN: Il file '{mp4_path}' non è un file MP4 valido. Ignorato.")
        return

    # Costruisci il nome del file di output
    output_filename = f"{mp4_path.stem}_mixture.wav"
    output_wav_path = output_dir / output_filename

    print(f"INFO: Processando '{mp4_path.name}'...")

    # Comando FFmpeg:
    # -i input.mp4       : Specifica il file di input
    # -map 0:a:0         : Seleziona la traccia audio 0 (la prima) dal primo input (0)
    # -acodec pcm_s16le  : Specifica il codec audio per WAV (PCM 16-bit little-endian, standard)
    # -ar 44100          : Imposta la frequenza di campionamento (opzionale, puoi rimuoverla
    #                      per mantenere quella originale se diversa da 44100Hz)
    # -ac 2              : Forza l'output a stereo (le tracce MUSDB18 sono stereo)
    # -y                 : Sovrascrive il file di output se esiste senza chiedere
    # -loglevel error    : Mostra solo errori critici da ffmpeg
    # output.wav         : Specifica il file di output
    command = [
        'ffmpeg',
        '-i', str(mp4_path),
        '-map', '0:a:0',        # Seleziona la prima traccia audio
        '-acodec', 'pcm_s16le', # Codec WAV standard
        # '-ar', '44100',       # Descommenta se vuoi forzare 44.1kHz
        '-ac', '2',            # Forza stereo
        '-y',                  # Sovrascrivi output
        '-loglevel', 'error',
        str(output_wav_path)
    ]

    try:
        # Esegui il comando ffmpeg
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"INFO: Traccia mixture estratta con successo: '{output_wav_path.name}'")

        # --- Verifica Base ---
        # Verifica che il file esista e non sia vuoto
        if output_wav_path.exists() and output_wav_path.stat().st_size > 0:
            print(f"VERIFICA: File '{output_wav_path.name}' creato e non vuoto.")
            # Qui potresti aggiungere verifiche più specifiche usando librerie
            # come 'soundfile' o 'wave' se necessario (es. controllare sample rate, canali)
        else:
             print(f"ERRORE: Il file di output '{output_wav_path.name}' non è stato creato o è vuoto.")

    except subprocess.CalledProcessError as e:
        print(f"ERRORE: ffmpeg ha fallito durante l'elaborazione di '{mp4_path.name}'.")
        print(f"  Comando: {' '.join(command)}")
        print(f"  Errore ffmpeg: {e.stderr}")
    except Exception as e:
        print(f"ERRORE: Si è verificato un errore inaspettato durante l'elaborazione di '{mp4_path.name}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Estrae la traccia audio 'mixture' (traccia 0) da file MP4 MUSDB18 e la salva come WAV.")
    parser.add_argument("input_path", help="Percorso del file MP4 o della cartella contenente file MP4.")
    parser.add_argument("-o", "--output", help="Cartella di output per i file WAV (opzionale, default: stessa cartella dell'input).")

    args = parser.parse_args()

    # Verifica la disponibilità di ffmpeg prima di iniziare
    if not check_ffmpeg():
        sys.exit(1) # Esce dallo script se ffmpeg non è disponibile

    input_path = Path(args.input_path).resolve() # Ottieni il percorso assoluto
    output_dir = None

    if not input_path.exists():
        print(f"ERRORE: Il percorso specificato '{args.input_path}' non esiste.")
        sys.exit(1)

    # Determina la cartella di output
    if args.output:
        output_dir = Path(args.output).resolve()
        # Crea la cartella di output se non esiste
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"INFO: I file WAV verranno salvati in '{output_dir}'")
        except OSError as e:
             print(f"ERRORE: Impossibile creare la cartella di output '{output_dir}': {e}")
             sys.exit(1)
    else:
        # Se non specificata, usa la stessa cartella del file o la cartella input
        if input_path.is_file():
            output_dir = input_path.parent
        else: # è una cartella
            output_dir = input_path
        print(f"INFO: I file WAV verranno salvati nella cartella di input/origine '{output_dir}'")


    if input_path.is_file():
        # Caso: Input è un singolo file
        if input_path.suffix.lower() == '.mp4':
             extract_mixture(input_path, output_dir)
        else:
            print(f"ERRORE: Il file specificato '{input_path}' non è un file .mp4.")
            sys.exit(1)

    elif input_path.is_dir():
        # Caso: Input è una cartella
        print(f"INFO: Scansione della cartella '{input_path}' per file .mp4...")
        mp4_files_found = list(input_path.glob('*.mp4')) # Cerca file .mp4 (case-insensitive su alcuni OS)
        # Alternativa per case-insensitive su tutti gli OS:
        # mp4_files_found = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == '.mp4']

        if not mp4_files_found:
            print("INFO: Nessun file .mp4 trovato nella cartella specificata.")
        else:
            print(f"INFO: Trovati {len(mp4_files_found)} file .mp4. Inizio elaborazione...")
            count = 0
            for mp4_file in mp4_files_found:
                count += 1
                print(f"\n--- File {count}/{len(mp4_files_found)} ---")
                extract_mixture(mp4_file, output_dir)
            print("\nINFO: Elaborazione completata.")

    else:
        # Caso: Né file né cartella (improbabile dopo il check iniziale, ma per sicurezza)
        print(f"ERRORE: Il percorso specificato '{input_path}' non è né un file né una cartella valida.")
        sys.exit(1)

if __name__ == "__main__":
    main()