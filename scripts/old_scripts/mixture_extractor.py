import os
import subprocess
import argparse
import sys
from pathlib import Path
import shutil

def check_ffmpeg():
    """Verifica se ffmpeg è installato e accessibile nel PATH."""
    if shutil.which("ffmpeg"):
        print("INFO: ffmpeg trovato.")
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, OSError) as e:
             print(f"ERRORE: Problema nell'eseguire ffmpeg -version: {e}")
             return False
    else:
        print("ERRORE: ffmpeg non trovato. Assicurati che sia installato e nel PATH di sistema.")
        print("         Puoi installarlo tramite il gestore pacchetti (es. 'sudo apt install ffmpeg' o 'brew install ffmpeg')")
        print("         o scaricarlo da https://ffmpeg.org/download.html")
        return False


def extract_mixture(mp4_path: Path, output_dir: Path,
                    output_format: str = 'aac',  # Default è FLAC
                    bitrate: str = '320k',        # Default per MP3/AAC se scelti
                    sample_rate: int = None,
                    force_mono: bool = False,
                    flac_compression: int = 5):   # Default compression per FLAC
    """
    Estrae la traccia audio 0 da un file MP4 e la salva nel formato specificato.
    Default: FLAC (lossless compresso, livello 5), sample rate e canali originali.

    Args:
        mp4_path (Path): Percorso del file MP4 di input.
        output_dir (Path): Cartella dove salvare il file di output.
        output_format (str): Formato ('wav', 'mp3', 'aac', 'flac'). Default: 'flac'.
        bitrate (str): Bitrate per formati lossy (mp3, aac). IGNORATO per wav/flac. Default: '192k'.
        sample_rate (int, optional): Forza frequenza campionamento. Default: None (originale).
        force_mono (bool): Forza output mono. Default: False (originali).
        flac_compression (int): Livello compressione FLAC (0-8). Default: 5. Ignorato se non FLAC.
    """
    if not mp4_path.is_file() or mp4_path.suffix.lower() != '.mp4':
        print(f"WARN: Il file '{mp4_path}' non è un file MP4 valido o non esiste. Ignorato.")
        return False

    output_format = output_format.lower()
    allowed_formats = ['wav', 'mp3', 'aac', 'flac']
    if output_format not in allowed_formats:
        print(f"ERRORE: Formato di output '{output_format}' non supportato. Scegliere tra: {', '.join(allowed_formats)}")
        print(f"INFO: Utilizzo del formato di default 'flac'.")
        output_format = 'flac' # Usa il default FLAC in caso di errore


    # Costruisci il nome del file di output
    output_filename = f"{mp4_path.stem}_mixture.{output_format}"
    output_path = output_dir / output_filename

    print(f"INFO: Processando '{mp4_path.name}' -> '{output_path.name}' (Formato: {output_format.upper()})")

    # Base command
    command = [
        'ffmpeg',
        '-i', str(mp4_path),
        '-map', '0:a:0',        # Seleziona la prima traccia audio
    ]

    # Impostazioni specifiche del formato
    if output_format == 'wav':
        command.extend(['-acodec', 'pcm_s16le'])
        print("INFO: Output in formato WAV (lossless non compresso).")
    elif output_format == 'mp3':
        command.extend(['-acodec', 'libmp3lame'])
        command.extend(['-b:a', bitrate])
        print(f"INFO: Output in formato MP3 (lossy) con bitrate {bitrate}.")
    elif output_format == 'aac':
        command.extend(['-acodec', 'aac'])
        command.extend(['-b:a', bitrate])
        print(f"INFO: Output in formato AAC (lossy) con bitrate {bitrate}.")
    elif output_format == 'flac':
        command.extend(['-acodec', 'flac'])
        command.extend(['-compression_level', str(flac_compression)])
        print(f"INFO: Output in formato FLAC (lossless compresso) con livello {flac_compression}.")
        # Nota: Il bitrate specificato viene ignorato per FLAC. Il warning è in main().

    # Gestione Canali
    if force_mono:
        command.extend(['-ac', '1'])
        print("INFO: Conversione forzata a mono.")
    else:
        print("INFO: Mantenimento dei canali audio originali.")


    # Gestione Sample Rate
    if sample_rate:
        command.extend(['-ar', str(sample_rate)])
        print(f"INFO: Ricampionamento a {sample_rate} Hz.")
    else:
         print("INFO: Mantenimento del sample rate originale.")


    # Opzioni finali
    command.extend([
        '-y',
        '-loglevel', 'warning',
        str(output_path)
    ])

    try:
        print(f"DEBUG: Comando ffmpeg: {' '.join(command)}")
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"INFO: Traccia mixture estratta con successo: '{output_path.name}'")

        if output_path.exists() and output_path.stat().st_size > 0:
            original_size_mb = mp4_path.stat().st_size / (1024 * 1024)
            output_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"VERIFICA: File '{output_path.name}' creato ({output_size_mb:.2f} MB). (Originale MP4: {original_size_mb:.2f} MB)")
            return True
        else:
             print(f"ERRORE: Il file di output '{output_path.name}' non è stato creato o è vuoto.")
             return False

    except subprocess.CalledProcessError as e:
        print(f"ERRORE: ffmpeg ha fallito durante l'elaborazione di '{mp4_path.name}'.")
        print(f"  Comando: {' '.join(command)}")
        if e.stdout: print(f"  Output ffmpeg (stdout):\n{e.stdout}")
        if e.stderr: print(f"  Errore ffmpeg (stderr):\n{e.stderr}")
        return False
    except FileNotFoundError:
         print(f"ERRORE CRITICO: Il comando 'ffmpeg' non è stato trovato durante l'esecuzione.")
         return False
    except Exception as e:
        print(f"ERRORE: Errore inaspettato durante l'elaborazione di '{mp4_path.name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Usa ArgumentDefaultsHelpFormatter per mostrare i default nell'help
    # Usa RawTextHelpFormatter per permettere newline nella descrizione
    class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Estrae la traccia audio 'mixture' (traccia 0) da file MP4 e la salva in formato audio.\n"
            "Default: FLAC (lossless compresso, livello 5), mantenendo sample rate e canali originali.\n"
            "FLAC è lossless e IGNORA l'opzione --bitrate."
            ),
        formatter_class=CustomHelpFormatter
        )
    parser.add_argument("input_path", help="Percorso del file MP4 o della cartella contenente file MP4.")
    parser.add_argument("-o", "--output", help="Cartella di output per i file audio (default: stessa cartella dell'input).")

    # --- Impostazioni Formato (Default FLAC) ---
    parser.add_argument("-f", "--format", choices=['wav', 'mp3', 'aac', 'flac'], default='aac',
                        help="Formato del file audio di output.")
    # Default bitrate per quando si SCELGONO mp3/aac
    parser.add_argument("-b", "--bitrate", default='320k',
                        help="Bitrate per formati lossy (mp3, aac). Es: '96k', '128k', '192k', '320k'.\nIGNORATO per il formato default (FLAC) e per WAV.")

    # --- Opzioni Qualità/Dimensione ---
    parser.add_argument("-sr", "--sample-rate", type=int, default=None,
                        help="Forza una specifica frequenza di campionamento (es. 44100, 22050).\nDefault: mantiene l'originale.")
    parser.add_argument("-m", "--mono", action='store_true',
                        help="Forza l'output in mono (1 canale).\nDefault: mantiene i canali originali.")

    # --- Opzione Specifica per FLAC (Default 5) ---
    parser.add_argument("--flac-compression", type=int, default=5, choices=range(0, 9), # range 0-8
                        help="Livello di compressione FLAC (0=veloce/grande, 8=lento/piccolo).\nUsato solo se il formato è 'flac'.")

    args = parser.parse_args()

    # Verifica ffmpeg
    if not check_ffmpeg():
        sys.exit(1)

    input_path = Path(args.input_path).resolve()
    output_dir = None

    if not input_path.exists():
        print(f"ERRORE: Il percorso specificato '{args.input_path}' non esiste.")
        sys.exit(1)

    # Determina e crea cartella di output
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = input_path.parent if input_path.is_file() else input_path
        print(f"INFO: Nessuna cartella di output specificata (--output). I file verranno salvati in '{output_dir}'")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.output: print(f"INFO: I file audio verranno salvati in '{output_dir}'")
    except OSError as e:
         print(f"ERRORE: Impossibile creare la cartella di output '{output_dir}': {e}")
         sys.exit(1)

    # --- Validazioni Argomenti ---
    # Warning se si specifica bitrate ESPLICITAMENTE per formati non lossy
    # Controlla se l'argomento bitrate è stato effettivamente fornito dall'utente
    # (non basta confrontare col default se il default stesso viene modificato)
    # Un modo semplice è controllare se 'bitrate' è negli argomenti passati a sys.argv
    # Approccio più robusto: controllare se il valore di args.bitrate è diverso dal valore di default *definito nel parser*
    bitrate_arg_provided = args.bitrate != parser.get_default('bitrate')

    if args.format in ['wav', 'flac'] and bitrate_arg_provided:
         print(f"WARN: L'opzione --bitrate ('{args.bitrate}') è stata specificata ma verrà IGNORATA per il formato lossless '{args.format.upper()}'.")

    # Validazione bitrate per formati lossy
    if args.format in ['mp3', 'aac'] and not (args.bitrate.lower().endswith('k') and args.bitrate[:-1].isdigit()):
         print(f"WARN: Formato bitrate '{args.bitrate}' non sembra standard (es. '192k'). Ffmpeg potrebbe fallire.")

    # --- Elaborazione ---
    if input_path.is_file():
        if input_path.suffix.lower() == '.mp4':
             success = extract_mixture(
                 input_path, output_dir, args.format, args.bitrate,
                 args.sample_rate, args.mono, args.flac_compression
                 )
             if not success: sys.exit(1)
        else:
            print(f"ERRORE: Il file specificato '{input_path}' non è un file .mp4.")
            sys.exit(1)

    elif input_path.is_dir():
        print(f"INFO: Scansione della cartella '{input_path}' per file .mp4...")
        mp4_files_found = list(input_path.glob('*.[mM][pP]4'))

        if not mp4_files_found:
            print("INFO: Nessun file .mp4 trovato nella cartella specificata.")
        else:
            total_files = len(mp4_files_found)
            print(f"INFO: Trovati {total_files} file .mp4. Inizio elaborazione...")

            # Descrizione impostazioni aggiornata
            sr_desc = f"{args.sample_rate} Hz" if args.sample_rate else "Originale"
            ch_desc = "Mono" if args.mono else "Originali"
            extra_opts = ""
            if args.format == 'flac':
                 extra_opts = f", Compressione FLAC={args.flac_compression}"
                 # Non mostrare il bitrate per FLAC
                 print(f"INFO: Impostazioni: Formato={args.format.upper()}{extra_opts}, Sample Rate={sr_desc}, Canali={ch_desc}")
            elif args.format in ['mp3', 'aac']:
                extra_opts = f", Bitrate={args.bitrate}"
                print(f"INFO: Impostazioni: Formato={args.format.upper()}{extra_opts}, Sample Rate={sr_desc}, Canali={ch_desc}")
            else: # WAV
                print(f"INFO: Impostazioni: Formato={args.format.upper()}, Sample Rate={sr_desc}, Canali={ch_desc}")


            success_count = 0
            fail_count = 0
            for i, mp4_file in enumerate(mp4_files_found):
                print(f"\n--- File {i+1}/{total_files} ---")
                if extract_mixture(mp4_file, output_dir, args.format, args.bitrate,
                                   args.sample_rate, args.mono, args.flac_compression):
                    success_count += 1
                else:
                    fail_count += 1

            print("\n--- Riepilogo Elaborazione ---")
            print(f"File totali trovati: {total_files}")
            print(f"File processati con successo: {success_count}")
            print(f"File falliti: {fail_count}")
            print("INFO: Elaborazione completata.")
            if fail_count > 0: sys.exit(1)

    else:
        print(f"ERRORE: Il percorso specificato '{input_path}' non è né un file né una cartella valida.")
        sys.exit(1)

if __name__ == "__main__":
    main()