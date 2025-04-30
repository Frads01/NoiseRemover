import argparse
import os
import sys
import warnings
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import math
import re
import time

# Ignora avvisi specifici di torchaudio se necessario (es. backend)
warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')

# --- Costanti ---
OUTPUT_DIR_NAME = "db_merged"
SONG_SUFFIX = "_mixture"
NOISE_SUFFIX = "_zeromean"
DEFAULT_OUTPUT_FORMAT = "aac"
DEFAULT_AAC_BITRATE = "256k" # Stringa per ffmpeg/torchaudio backend se necessario

# Mappatura per formati e qualità "massima ragionevole"
# Nota: 'bits_per_sample' è rilevante per WAV/FLAC, 'compression' per FLAC, 'encoding'/'bitrate' per lossy.
# torchaudio.save usa backend (soundfile, sox), i parametri esatti possono dipendere da esso.
# Per MP3, 'V0' è spesso considerato alta qualità VBR. torchaudio potrebbe non esporlo direttamente.
# Usiamo None dove il parametro non è applicabile o per lasciare il default del backend.
MAX_QUALITY_PARAMS = {
    'wav': {'format': 'wav', 'encoding': 'PCM_S', 'bits_per_sample': 24}, # WAV 24-bit
    'flac': {'format': 'flac', 'compression': 8}, # FLAC max compression (lossless)
    'mp3': {'format': 'mp3', 'compression': -9.4}, # Equivalente a -V0 LAME (alta qualità VBR) - Potrebbe richiedere sox backend
    'aac': {'format': 'aac', 'compression': -1} # torchaudio non sembra avere controllo bitrate diretto per AAC, usiamo default
}

# --- Funzioni Helper ---

def check_cuda_device():
    """Verifica la disponibilità di CUDA e imposta il dispositivo."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"INFO: CUDA disponibile. Utilizzo GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("INFO: CUDA non disponibile. Utilizzo CPU.")
    return device

def find_audio_files(input_path: Path, required_suffix: str) -> list[Path]:
    """
    Cerca file audio in un percorso (file o directory di primo livello)
    filtrando per suffisso del nome base.
    """
    valid_files = []
    if not input_path.exists():
        print(f"ERRORE: Il percorso '{input_path}' non esiste.", file=sys.stderr)
        return []

    # Estensioni audio comuni supportate da torchaudio (dipende dal backend)
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.aiff', '.aif'}

    if input_path.is_file():
        if input_path.stem.endswith(required_suffix) and input_path.suffix.lower() in audio_extensions:
            valid_files.append(input_path)
        else:
            print(f"INFO: File '{input_path.name}' ignorato (nome non termina con '{required_suffix}' o estensione non audio).")
    elif input_path.is_dir():
        print(f"INFO: Scansione directory '{input_path}'...")
        for item in input_path.iterdir():
            # Solo primo livello, controlla se è un file
            if item.is_file():
                if item.stem.endswith(required_suffix) and item.suffix.lower() in audio_extensions:
                    valid_files.append(item)
                # else: # Commentato per ridurre verbosità
                #     print(f"INFO: File '{item.name}' ignorato (nome/estensione non corrispondente).")
    else:
         print(f"ERRORE: Il percorso '{input_path}' non è né un file né una directory valida.", file=sys.stderr)

    print(f"INFO: Trovati {len(valid_files)} file validi con suffisso '{required_suffix}' in '{input_path}'.")
    return valid_files

def load_audio(file_path: Path, device: torch.device) -> tuple[torch.Tensor | None, int | None]:
    """Carica un file audio usando torchaudio e lo sposta sul dispositivo specificato."""
    try:
        # waveform: (channels, time)
        # sample_rate: int
        waveform, sample_rate = torchaudio.load(file_path, format=None) # format=None per autodetect
        return waveform.to(device), sample_rate
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il file audio '{file_path}': {e}", file=sys.stderr)
        return None, None

def save_audio(waveform: torch.Tensor, sample_rate: int, output_path: Path,
               output_format: str, song_channels: int, song_sample_rate: int):
    """Salva il tensore audio nel formato specificato."""
    try:
        # Prepara i parametri di salvataggio
        save_kwargs = {}
        target_channels = song_channels
        target_sample_rate = song_sample_rate

        if output_format != DEFAULT_OUTPUT_FORMAT:
            # Formati opzionali: forza stereo e usa SR canzone (o max ragionevole se richiesto)
            target_channels = 2
            # Potremmo forzare SR più alto, ma usiamo quello della canzone per coerenza
            target_sample_rate = song_sample_rate

            if output_format in MAX_QUALITY_PARAMS:
                 save_kwargs = MAX_QUALITY_PARAMS[output_format].copy()
                 # Rimuovi 'format' perché è il primo argomento posizionale di save
                 if 'format' in save_kwargs: del save_kwargs['format']
            else:
                print(f"ATTENZIONE: Nessun parametro di qualità massima definito per '{output_format}'. Uso default.")

        else:
            # Formato di default (AAC): usa SR e canali della canzone originale
            # Torchaudio potrebbe non permettere di impostare bitrate AAC direttamente via save()
            # Dipende molto dal backend (sox, soundfile+libsndfile/ffmpeg)
            # Proviamo a salvare come AAC, potrebbe usare un default ragionevole.
            # Se serve controllo bitrate preciso, bisognerebbe salvare in WAV e poi usare ffmpeg esternamente.
            # Per semplicità, qui ci affidiamo a torchaudio.save()
            save_kwargs = {'format': 'aac'} # Potrebbe richiedere compression=-1 o altri parametri specifici del backend
            print(f"INFO: Tentativo di salvataggio in AAC (bitrate {DEFAULT_AAC_BITRATE} desiderato, controllo effettivo dipende dal backend torchaudio).")


        # --- Gestione Canali e Sample Rate prima del salvataggio ---
        # 1. Resample se necessario (normalmente non dovrebbe servire se abbiamo già allineato SR noise a song)
        if waveform.shape[0] != 0 and sample_rate != target_sample_rate:
             print(f"INFO: Ricampionamento output da {sample_rate} Hz a {target_sample_rate} Hz.")
             resampler = T.Resample(sample_rate, target_sample_rate, dtype=waveform.dtype).to(waveform.device)
             waveform = resampler(waveform)
             sample_rate = target_sample_rate # Aggiorna sample rate per il salvataggio

        # 2. Adatta i canali se necessario
        current_channels = waveform.shape[0]
        if current_channels != target_channels:
            print(f"INFO: Adattamento canali output da {current_channels} a {target_channels}.")
            if target_channels == 1 and current_channels > 1:
                # Mixdown a mono (media)
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif target_channels == 2 and current_channels == 1:
                # Converti mono a stereo duplicando il canale
                waveform = waveform.repeat(2, 1)
            elif target_channels > 2 and current_channels == 1:
                 # Converti mono a multi-canale duplicando
                 waveform = waveform.repeat(target_channels, 1)
            elif target_channels == 2 and current_channels > 2:
                 # Mixdown multi-canale > 2 a stereo (es. prendi i primi 2)
                 waveform = waveform[:2, :]
            else:
                 # Altri casi (es. 6 canali -> 1) potrebbero richiedere logiche specifiche
                 print(f"ATTENZIONE: Conversione canali da {current_channels} a {target_channels} non gestita esplicitamente, potrebbero esserci risultati inattesi. Prendo i primi {target_channels} canali.")
                 if current_channels > target_channels:
                    waveform = waveform[:target_channels, :]
                 else: # target > current
                     # Potremmo duplicare l'ultimo canale? O lasciare come è e sperare che save gestisca?
                     # Per sicurezza, non facciamo nulla qui e lasciamo che save() fallisca se non può gestire
                     pass


        # Assicurati che il tensore sia sulla CPU prima di salvare (alcuni backend potrebbero richiederlo)
        waveform_cpu = waveform.cpu()

        # Esegui il salvataggio
        torchaudio.save(
            output_path,
            waveform_cpu,
            sample_rate, # Usiamo il sample_rate (potenzialmente ricampionato)
            format=output_format,
            **save_kwargs
        )
        # print(f"DEBUG: Salvataggio con parametri: format={output_format}, sample_rate={sample_rate}, kwargs={save_kwargs}")
        return True

    except Exception as e:
        print(f"ERRORE: Impossibile salvare il file audio '{output_path}' come {output_format}: {e}", file=sys.stderr)
        # Considera di rimuovere il file parzialmente scritto se l'errore avviene durante la scrittura
        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass # Ignora se non si può rimuovere
        return False

def normalize_audio(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalizza il waveform (peak normalization per canale)."""
    # Normalizza per canale per evitare sbilanciamenti stereo
    max_val_per_channel = torch.max(torch.abs(waveform), dim=1, keepdim=True)[0]
    # Evita divisione per zero se un canale è silenzioso
    max_val_per_channel = torch.clamp(max_val_per_channel, min=eps)
    return waveform / max_val_per_channel

def adjust_volume_db(waveform: torch.Tensor, db_change: float) -> torch.Tensor:
    """Applica un cambiamento di volume in dB."""
    if db_change == 0.0:
        return waveform
    gain = 10.0**(db_change / 20.0)
    return waveform * gain

def overlay_noise(song_waveform: torch.Tensor, noise_waveform: torch.Tensor,
                  song_sr: int, noise_sr: int,
                  device: torch.device) -> torch.Tensor:
    """
    Sovrappone il rumore alla canzone, gestendo durata, sample rate e canali.
    Restituisce il waveform risultante.
    """
    song_channels, song_samples = song_waveform.shape
    noise_channels, noise_samples = noise_waveform.shape

    # --- 1. Allinea Sample Rate (Resample Noise se necessario) ---
    if noise_sr != song_sr:
        print(f"INFO: Ricampionamento rumore da {noise_sr} Hz a {song_sr} Hz per sovrapposizione.")
        resampler = T.Resample(noise_sr, song_sr, dtype=noise_waveform.dtype).to(device)
        noise_waveform = resampler(noise_waveform)
        # Aggiorna numero campioni rumore dopo resampling
        noise_samples = noise_waveform.shape[1]
        noise_sr = song_sr # Ora sono uguali

    # --- 2. Allinea Canali (Noise -> Song Channels) ---
    if noise_channels != song_channels:
        print(f"INFO: Adattamento canali rumore da {noise_channels} a {song_channels} per sovrapposizione.")
        if song_channels == 1 and noise_channels > 1:
            # Mixdown noise a mono
            noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)
        elif song_channels == 2 and noise_channels == 1:
            # Duplica canale mono noise a stereo
            noise_waveform = noise_waveform.repeat(2, 1)
        elif song_channels > 1 and noise_channels == 1:
             # Duplica canale mono noise a multi-canale
            noise_waveform = noise_waveform.repeat(song_channels, 1)
        elif song_channels == 2 and noise_channels > 2:
             # Prendi i primi 2 canali del noise multi-canale
             noise_waveform = noise_waveform[:2, :]
        else:
             # Caso generico: se noise ha più canali della song, prendi i primi 'song_channels'
             if noise_channels > song_channels:
                 noise_waveform = noise_waveform[:song_channels, :]
             else:
                 # Se noise ha meno canali (e non è mono vs stereo/multi), cosa fare?
                 # Potrebbe essere un errore, ma proviamo a duplicare l'ultimo canale esistente
                 print(f"ATTENZIONE: Gestione canali non standard (Noise {noise_channels}ch vs Song {song_channels}ch). Duplico ultimo canale noise.")
                 last_channel = noise_waveform[-1:, :]
                 repeats_needed = song_channels - noise_channels
                 noise_waveform = torch.cat([noise_waveform] + [last_channel] * repeats_needed, dim=0)

        noise_channels = noise_waveform.shape[0] # Aggiorna numero canali
        if noise_channels != song_channels:
             # Se ancora non corrispondono, qualcosa è andato storto
             print(f"ERRORE: Impossibile allineare i canali del rumore ({noise_waveform.shape[0]}) a quelli della canzone ({song_channels}). Salto sovrapposizione.", file=sys.stderr)
             # Restituisce la canzone originale in caso di errore grave di allineamento
             # O potremmo sollevare un'eccezione
             return song_waveform


    # --- 3. Gestisci Durata Rumore (Loop/Truncate) ---
    if noise_samples == 0:
        print("ATTENZIONE: Il file di rumore sembra vuoto dopo il caricamento/resampling. Salto sovrapposizione.", file=sys.stderr)
        return song_waveform

    adjusted_noise_waveform = torch.zeros_like(song_waveform, device=device)

    if noise_samples < song_samples:
        # Loop noise
        num_repeats = song_samples // noise_samples
        remainder = song_samples % noise_samples
        looped_noise_parts = [noise_waveform] * num_repeats
        if remainder > 0:
            looped_noise_parts.append(noise_waveform[:, :remainder])
        # Controlla se `looped_noise_parts` non è vuoto prima di `torch.cat`
        if looped_noise_parts:
             adjusted_noise_waveform = torch.cat(looped_noise_parts, dim=1)
        else:
             # Caso strano: song_samples=0 o noise_samples=0 (già gestito sopra)
             # O num_repeats=0 e remainder=0, il che implica song_samples=0
             # Se song_samples > 0 e noise_samples > 0, questo non dovrebbe accadere
             print("ATTENZIONE: Nessuna parte di rumore da concatenare durante il loop. Verificare le durate.")
             # adjusted_noise_waveform rimane a zero


    elif noise_samples >= song_samples:
        # Truncate noise
         if song_samples > 0: # Evita slicing negativo se la canzone è vuota
            adjusted_noise_waveform = noise_waveform[:, :song_samples]
         # else: adjusted_noise_waveform rimane a zero


    # Verifica finale dimensioni prima dell'addizione
    if adjusted_noise_waveform.shape != song_waveform.shape:
        print(f"ERRORE: Disallineamento shape prima della sovrapposizione! Song: {song_waveform.shape}, Noise: {adjusted_noise_waveform.shape}. Salto sovrapposizione.", file=sys.stderr)
        # Tenta di aggiustare troncando/paddando il rumore alla dimensione esatta della canzone
        # Questo non dovrebbe succedere con la logica sopra, ma è una salvaguardia
        if adjusted_noise_waveform.shape[1] > song_samples:
            adjusted_noise_waveform = adjusted_noise_waveform[:, :song_samples]
        elif adjusted_noise_waveform.shape[1] < song_samples:
             padding_needed = song_samples - adjusted_noise_waveform.shape[1]
             padding = torch.zeros((song_channels, padding_needed), dtype=adjusted_noise_waveform.dtype, device=device)
             adjusted_noise_waveform = torch.cat([adjusted_noise_waveform, padding], dim=1)

        # Se ancora non combacia (problema di canali?), ritorna la canzone originale
        if adjusted_noise_waveform.shape != song_waveform.shape:
             print(f"ERRORE: Impossibile correggere disallineamento shape. Salto sovrapposizione.", file=sys.stderr)
             return song_waveform


    # --- 4. Sovrapponi (Addizione) ---
    # L'addizione potrebbe portare a clipping (>1.0 o <-1.0).
    # Potremmo normalizzare il risultato, ma per ora facciamo semplice somma.
    merged_waveform = song_waveform + adjusted_noise_waveform

    # --- Opzionale: Clipping ---
    # merged_waveform = torch.clamp(merged_waveform, -1.0, 1.0)
    # print("INFO: Applicato clipping a [-1.0, 1.0] dopo la sovrapposizione.")

    return merged_waveform

# --- Funzione Principale ---

def main():
    parser = argparse.ArgumentParser(description="Sovrappone file audio di rumore a file audio di canzoni con opzioni avanzate.")

    parser.add_argument("path_canzoni", type=str,
                        help="Percorso al file canzone o directory contenente le canzoni (file *_mixture).")
    parser.add_argument("path_rumori", type=str,
                        help="Percorso al file rumore o directory contenente i rumori (file *_zeromean).")
    parser.add_argument("--formato_output", type=str, choices=['aac', 'flac', 'wav', 'mp3'], default=DEFAULT_OUTPUT_FORMAT,
                        help=f"Formato del file di output (default: {DEFAULT_OUTPUT_FORMAT}).")
    parser.add_argument("--volume_rumore_db", type=float, default=0.0,
                        help="Aggiustamento del volume (in dB) da applicare ai file di rumore prima della sovrapposizione (default: 0.0).")
    # Potremmo aggiungere un flag per controllare il clipping post-sovrapposizione
    # parser.add_argument("--clip_output", action="store_true", help="Applica clipping a [-1, 1] all'output.")

    args = parser.parse_args()

    # --- Setup Iniziale ---
    start_time = time.time()
    device = check_cuda_device()

    path_canzoni = Path(args.path_canzoni)
    path_rumori = Path(args.path_rumori)
    output_dir = Path.cwd() / OUTPUT_DIR_NAME

    # Crea directory di output se non esiste
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Directory di output: '{output_dir}'")
    except OSError as e:
        print(f"ERRORE: Impossibile creare la directory di output '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Trova File di Input ---
    song_files = find_audio_files(path_canzoni, SONG_SUFFIX)
    noise_files = find_audio_files(path_rumori, NOISE_SUFFIX)

    if not song_files:
        print("ERRORE: Nessun file canzone valido trovato. Termino.", file=sys.stderr)
        sys.exit(1)
    if not noise_files:
        print("ERRORE: Nessun file rumore valido trovato. Termino.", file=sys.stderr)
        sys.exit(1)

    # --- Processo di Elaborazione ---
    total_files_to_process = len(song_files) * len(noise_files)
    processed_count = 0
    error_count = 0

    print(f"\nINFO: Inizio elaborazione di {len(song_files)} canzoni e {len(noise_files)} rumori ({total_files_to_process} combinazioni totali).")

    for song_path in song_files:
        print(f"\n--- Elaborazione Canzone: {song_path.name} ---")

        # Carica la canzone
        song_waveform, song_sr = load_audio(song_path, device)
        if song_waveform is None or song_sr is None:
            print(f"ERRORE: Salto canzone '{song_path.name}' a causa di errore di caricamento.", file=sys.stderr)
            error_count += len(noise_files) # Incrementa errore per tutte le combinazioni con questa canzone
            continue

        song_channels = song_waveform.shape[0]
        song_duration_sec = song_waveform.shape[1] / song_sr if song_sr > 0 else 0
        print(f"  - Caricata canzone: {song_channels} canali, {song_sr} Hz, {song_duration_sec:.2f} sec")

        for noise_path in noise_files:
            processed_count += 1
            print(f"\n  -> Sovrapposizione con Rumore ({processed_count}/{total_files_to_process}): {noise_path.name}")

            # Carica il rumore
            noise_waveform, noise_sr = load_audio(noise_path, device)
            if noise_waveform is None or noise_sr is None:
                print(f"ERRORE: Salto rumore '{noise_path.name}' per questa canzone.", file=sys.stderr)
                error_count += 1
                continue

            # Normalizza e aggiusta volume rumore
            print("     - Normalizzazione rumore...")
            noise_normalized = normalize_audio(noise_waveform)
            if args.volume_rumore_db != 0.0:
                 print(f"     - Applicazione volume: {args.volume_rumore_db} dB...")
                 noise_adjusted = adjust_volume_db(noise_normalized, args.volume_rumore_db)
            else:
                 noise_adjusted = noise_normalized

            # Esegui sovrapposizione (include resampling, channel/duration adjustment)
            print("     - Sovrapposizione (gestione durata, canali, SR)...")
            merged_waveform = overlay_noise(song_waveform, noise_adjusted, song_sr, noise_sr, device)

            # Costruisci nome file output
            song_base_name = song_path.stem.removesuffix(SONG_SUFFIX)
            noise_base_name = noise_path.stem.removesuffix(NOISE_SUFFIX)
            output_filename = f"[{noise_base_name}] {song_base_name}.{args.formato_output}"
            output_path = output_dir / output_filename

            # Salva il file risultante
            print(f"     - Salvataggio in '{output_path}'...")
            save_success = save_audio(
                merged_waveform,
                song_sr, # Il sample rate finale dovrebbe essere quello della canzone
                output_path,
                args.formato_output,
                song_channels=song_waveform.shape[0], # Passa i canali originali della canzone per riferimento
                song_sample_rate=song_sr # Passa SR originale canzone per riferimento
            )

            if not save_success:
                error_count += 1
            else:
                print(f"     - File salvato con successo.")

            # Opzionale: Pulisci memoria GPU se necessario (utile per processi lunghi)
            del noise_waveform, noise_normalized, noise_adjusted, merged_waveform
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Pulisci memoria canzone dopo aver processato tutti i rumori per essa
        del song_waveform
        if device.type == 'cuda':
             torch.cuda.empty_cache()


    # --- Riepilogo Finale ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n--- Elaborazione Completata ---")
    print(f"Tempo totale impiegato: {total_time:.2f} secondi")
    print(f"File totali processati (combinazioni): {processed_count}")
    print(f"File salvati con successo: {processed_count - error_count}")
    print(f"Errori incontrati: {error_count}")
    print(f"Tutti i file di output si trovano in: '{output_dir}'")

if __name__ == "__main__":
    # Aggiunge un piccolo controllo per torchaudio e torch
    try:
        import torch
        import torchaudio
        print(f"INFO: Utilizzo torch v{torch.__version__} e torchaudio v{torchaudio.__version__}")
    except ImportError:
        print("ERRORE: Le librerie 'torch' e 'torchaudio' sono necessarie. Installale con 'pip install torch torchaudio'.", file=sys.stderr)
        sys.exit(1)

    # Verifica backend torchaudio (informativo)
    try:
        backends = torchaudio.list_audio_backends()
        print(f"INFO: Backend audio disponibili per torchaudio: {backends}")
        current_backend = torchaudio.get_audio_backend()
        print(f"INFO: Backend audio corrente: {current_backend}")
        if current_backend is None and sys.platform == "win32":
             print("ATTENZIONE: Nessun backend audio trovato (potrebbe mancare soundfile o sox). Il caricamento/salvataggio potrebbe non funzionare.")
        elif current_backend is None:
             print("ATTENZIONE: Nessun backend audio trovato. Installare 'sox' o 'soundfile' (pip install soundfile) potrebbe essere necessario.")

    except Exception as e:
        print(f"ATTENZIONE: Impossibile verificare i backend torchaudio: {e}")


    main()