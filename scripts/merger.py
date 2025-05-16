#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
import wave

import numpy as np
import soundfile as sf
import torch  # Make sure torch is imported globally so it's accessible in all functions
import unicodedata
from pydub import AudioSegment

# Costanti globali

### MODIFICARE QUESTI PER TRAIN/TEST ###
IS_TRAINING = False

ITER_SONGS_MIN = 1  # Numero MINIMO di canzoni da processare
ITER_SONGS_MAX = 3  # Numero MASSIMO di canzoni da processare

ITER_NOISE_MIN = 1  # Numero MINIMO di coppie di rumori per canzone
ITER_NOISE_MAX = 10 # Numero MASSIMO di coppie di rumori per canzone
### -------------------------------- ###

# INPUT_DIR = "I:\\Il mio Drive\\dataset\\train\\input" if IS_TRAINING else "I:\\Il mio Drive\\dataset\\test\\input"
# TARGET_DIR = "I:\\Il mio Drive\\dataset\\train\\target" if IS_TRAINING else "I:\\Il mio Drive\\dataset\\test\\target"
INPUT_DIR = ".\\dataset\\train\\input" if IS_TRAINING else ".\\dataset\\test\\input"
TARGET_DIR = ".\\dataset\\train\\target" if IS_TRAINING else ".\\dataset\\test\\target"
SONGS_DIR = ".\\musdb18\\train" if IS_TRAINING else ".\\musdb18\\test"
NOISE_DIR = ".\\UrbanSound8K\\audio"


def slugify(text):
    """
    Converte un testo in uno slug URL-friendly.
    """
    # Converte in unicode se necessario
    if not isinstance(text, str):
        text = str(text)

    # Converte in ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    # Converte in minuscolo e rimuove caratteri non desiderati
    text = re.sub(r'[^\w\s-]', '', text.lower())

    # Sostituisce gli spazi con trattini
    text = re.sub(r'[-\s]+', '-', text).strip('-_')

    return text


def log_generation_stats(is_training, num_pairs, parent_dir):
    """
    Registra le statistiche di generazione in un file di log.

    Args:
        is_training (bool): True se è una sessione di TRAIN, False se è TEST
        num_pairs (int): Numero di coppie generate
        parent_dir (str): Directory principale del dataset
    """
    import os
    import datetime
    from pathlib import Path

    # Determina la directory padre che contiene le cartelle train e test
    log_file = os.path.join(parent_dir, "log.txt")

    # Calcola lo spazio occupato
    input_size = 0
    target_size = 0
    input_files = 0
    target_files = 0

    if is_training:
        session_type = "TRAIN"
        input_dir = os.path.join(parent_dir, "train", "input")
        target_dir = os.path.join(parent_dir, "train", "target")
    else:
        session_type = "TEST"
        input_dir = os.path.join(parent_dir, "test", "input")
        target_dir = os.path.join(parent_dir, "test", "target")

    # Calcola la dimensione totale delle cartelle e conta i file
    if os.path.exists(input_dir):
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                input_size += os.path.getsize(fp)
                input_files += 1

    if os.path.exists(target_dir):
        for dirpath, dirnames, filenames in os.walk(target_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                target_size += os.path.getsize(fp)
                target_files += 1

    total_size = input_size + target_size
    total_size_mb = total_size / (1024 * 1024)  # Conversione in MB
    total_files = input_files + target_files

    # Ottieni la data odierna
    current_date = datetime.datetime.now().strftime("%d/%m/%Y")

    # Crea il messaggio di log
    log_message = f"""--- DATASET for {session_type} - {current_date} ---

    Generated pairs: {num_pairs}
    Generated files: {total_files}
    Output size: {total_size_mb:.2f} MB
---

"""

    # Scrivi nel file di log (append)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message)

    print(f"Log scritto in: {log_file}")


def clean_directory(dir_path):
    """Pulisce o crea la directory di output se non esiste."""
    try:
        if os.path.exists(dir_path):
            # Rimuove il contenuto della directory se esiste
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Pulita directory esistente: {dir_path}")
        else:
            # Crea la directory se non esiste
            os.makedirs(dir_path)
            print(f"Creata directory: {dir_path}")
        return True
    except Exception as e:
        print(f"Errore durante la creazione/pulizia della directory {dir_path}: {e}")
        return False


def verify_ffmpeg():
    """Verifica se ffmpeg è installato e accessibile nel PATH."""
    if shutil.which("ffmpeg"):
        print("INFO: ffmpeg trovato.")
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"ERRORE: Problema nell'eseguire ffmpeg: {e}")
            return False
    else:
        print("ERRORE: ffmpeg non trovato. Assicurati che sia installato e nel PATH di sistema.")
        return False


def is_audio_file(file_path):
    """Verifica se il file è un file audio supportato."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.mp4', '.aac', '.ogg', '.m4a'}
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in audio_extensions


def load_audio(file_path, use_cuda=False):
    """
    Carica un file audio utilizzando librerie di supporto appropriate.
    Prova prima soundfile, poi pydub, gestendo diversi formati e codifiche.
    """
    try:
        # Prova prima con soundfile che supporta molti formati WAV
        try:
            data, sample_rate = sf.read(file_path, always_2d=True)
            # Trasforma in (canali, campioni) se è in (campioni, canali)
            if data.shape[1] <= 2:  # Se la seconda dimensione è 1 o 2 (mono o stereo)
                data = data.T

            channels = data.shape[0]
            sample_width = 2  # Assumiamo 16-bit per default

            # Se richiesto e disponibile, sposta su CUDA
            if use_cuda and torch.cuda.is_available():
                data_tensor = torch.tensor(data, dtype=torch.float32).to('cuda')
                return {
                    'tensor': data_tensor,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'sample_width': sample_width,
                    'format': os.path.splitext(file_path)[1][1:],
                    'cuda': True
                }
            else:
                return {
                    'array': data,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'sample_width': sample_width,
                    'format': os.path.splitext(file_path)[1][1:],
                    'cuda': False
                }
        except Exception as sf_error:
            # Se soundfile fallisce, prova con pydub
            try:
                audio = AudioSegment.from_file(file_path)
                
                # Normalizza volume dell'audio
                # audio = effects.normalize(audio)

                # Conversione in numpy array
                samples = np.array(audio.get_array_of_samples())

                # Se è mono, reshappiamo l'array per avere la dimensione del canale
                if audio.channels == 1:
                    samples = samples.reshape(1, -1)
                else:
                    # Converte un array stereo in formato (canali, samples)
                    samples = samples.reshape(-1, audio.channels).T

                # Se richiesto e disponibile, sposta su CUDA
                if use_cuda and torch.cuda.is_available():
                    samples_tensor = torch.tensor(samples, dtype=torch.float32).to('cuda')
                    return {
                        'tensor': samples_tensor,
                        'sample_rate': audio.frame_rate,
                        'channels': audio.channels,
                        'sample_width': audio.sample_width,
                        'format': os.path.splitext(file_path)[1][1:],
                        'cuda': True
                    }
                else:
                    return {
                        'array': samples,
                        'sample_rate': audio.frame_rate,
                        'channels': audio.channels,
                        'sample_width': audio.sample_width,
                        'format': os.path.splitext(file_path)[1][1:],
                        'cuda': False
                    }
            except Exception as pydub_error:
                # Se anche pydub fallisce, usa direttamente ffmpeg per convertire il file
                try:
                    temp_dir = os.path.join(os.getcwd(), "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_wav = os.path.join(temp_dir, f"temp_{os.path.basename(file_path)}.wav")

                    # Converti il file problematico in WAV usando ffmpeg
                    command = [
                        'ffmpeg',
                        '-i', file_path,
                        '-acodec', 'pcm_s16le',
                        '-y',
                        temp_wav
                    ]

                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Ora carica il file WAV temporaneo con wave
                    with wave.open(temp_wav, 'rb') as wf:
                        # Ottieni le proprietà
                        channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                        sample_rate = wf.getframerate()
                        frames = wf.getnframes()

                        # Leggi i dati
                        buffer = wf.readframes(frames)
                        samples = np.frombuffer(buffer, dtype=np.int16)

                        # Riorganizza in (canali, campioni)
                        if channels > 1:
                            samples = samples.reshape(-1, channels).T
                        else:
                            samples = samples.reshape(1, -1)

                    # Rimuovi il file temporaneo
                    os.remove(temp_wav)

                    # Se richiesto e disponibile, sposta su CUDA
                    if use_cuda and torch.cuda.is_available():
                        samples_tensor = torch.tensor(samples, dtype=torch.float32).to('cuda')
                        return {
                            'tensor': samples_tensor,
                            'sample_rate': sample_rate,
                            'channels': channels,
                            'sample_width': sample_width,
                            'format': 'wav',
                            'cuda': True
                        }
                    else:
                        return {
                            'array': samples,
                            'sample_rate': sample_rate,
                            'channels': channels,
                            'sample_width': sample_width,
                            'format': 'wav',
                            'cuda': False
                        }
                except Exception as ffmpeg_error:
                    print(f"Errore durante la conversione con ffmpeg: {ffmpeg_error}")
                    print(f"Errore originale con soundfile: {sf_error}")
                    print(f"Errore con pydub: {pydub_error}")
                    return None

    except Exception as e:
        print(f"Errore durante il caricamento del file audio {file_path}: {e}")
        traceback.print_exc()
        return None


def normalize_db(noise_data, clean_data):
    """
    Normalizes noise data relative to clean data with a random SNR between 0 and 10 dB.
    Works with both numpy arrays and PyTorch tensors.
    Returns the adjusted noise_data.
    """
    try:
        # Ensure noise_data is not None
        if noise_data is None:
            print("  Error: noise_data is None in normalize_db function")
            return None
            
        # Ensure clean_data is not None
        if clean_data is None:
            print("  Error: clean_data is None in normalize_db function")
            return None
            
        # Extract data arrays or tensors based on whether they're on CUDA or not
        if noise_data.get('cuda', False):
            # CUDA version with PyTorch
            noise_samples = noise_data['tensor']
            clean_samples = clean_data['tensor']
            
            # Calculate power of clean signal
            clean_power = torch.mean(clean_samples.float() ** 2)
            
            # Calculate power of noise
            noise_power = torch.mean(noise_samples.float() ** 2)
            
            # Generate random SNR between 0 and 10 dB (inclusive)
            target_snr = torch.randint(0, 11, (1,))
            
            # Calculate the scaling factor for noise based on the desired SNR
            # SNR = 10 * log10(clean_power / noise_power)
            # So: noise_power_new = clean_power / (10^(SNR/10))
            import math
            scaling_factor = torch.sqrt(clean_power / (noise_power * (10 ** (target_snr.float() / 10))))
            
            # Scale the noise
            noise_data['tensor'] = noise_samples * scaling_factor
            
        else:
            # CPU version with numpy
            noise_samples = noise_data['array']
            clean_samples = clean_data['array']
            
            # Calculate power of clean signal
            clean_power = np.mean(clean_samples ** 2)
            
            # Calculate power of noise
            noise_power = np.mean(noise_samples ** 2)
            
            # Generate random SNR between 0 and 10 dB (inclusive)
            target_snr = np.random.randint(0, 11)
            
            # Calculate the scaling factor for noise based on the desired SNR
            scaling_factor = np.sqrt(clean_power / (noise_power * (10 ** (target_snr / 10))))
            
            # Scale the noise
            noise_data['array'] = noise_samples * scaling_factor
        
        print(f"  Noise normalized with target SNR: {target_snr} dB")
        return noise_data
    
    except Exception as e:
        print(f"  Error during noise normalization: {e}")
        traceback.print_exc()  # Add traceback for debugging
        return None


def save_audio(audio_data, output_path, format_='wav'):
    """Salva i dati audio in un file, supportando sia dati numpy che tensori PyTorch."""
    try:
        # Estrai i dati in formato numpy se sono su CUDA
        if audio_data.get('cuda', False):
            samples = audio_data['tensor'].cpu().numpy()
        else:
            samples = audio_data['array']

        # Assicurati che i dati siano in int16 per la compatibilità con la maggior parte dei formati audio
        if samples.dtype != np.int16:
            # Scala e converti in int16 se necessario
            if samples.dtype == np.float32 or samples.dtype == np.float64:
                # Assumiamo che i float siano in range [-1, 1]
                samples = (samples * 32767).astype(np.int16)
            else:
                samples = samples.astype(np.int16)

        # Crea la directory di output se non esiste
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Prova a salvare con soundfile per prima scelta
        try:
            # Reshaping se necessario per soundfile (campioni, canali)
            sf_samples = samples.T
            sf.write(output_path, sf_samples, audio_data['sample_rate'])
            return True
        except Exception as sf_error:
            # Fallback a pydub
            try:
                # Converti in formato adatto per pydub (canali, samples) -> (samples, canali)
                if samples.ndim > 1:
                    pydub_samples = samples.T.reshape(-1)
                else:
                    pydub_samples = samples

                # Crea un nuovo AudioSegment
                segment = AudioSegment(
                    pydub_samples.tobytes(),
                    frame_rate=audio_data['sample_rate'],
                    sample_width=audio_data['sample_width'],
                    channels=audio_data['channels']
                )

                # Salva il file
                segment.export(output_path, format=format_)
                return True
            except Exception as pydub_error:
                # Se anche pydub fallisce, usa ffmpeg direttamente
                try:
                    # Salva temporaneamente come raw PCM
                    temp_dir = os.path.join(os.getcwd(), "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_raw = os.path.join(temp_dir, "temp_raw.pcm")

                    # Salva i dati raw
                    with open(temp_raw, 'wb') as f:
                        if samples.ndim > 1:
                            # Converti da (canali, campioni) a (campioni, canali)
                            samples_interleaved = samples.T.reshape(-1)
                        else:
                            samples_interleaved = samples
                        f.write(samples_interleaved.tobytes())

                    # Usa ffmpeg per convertire da raw PCM al formato desiderato
                    command = [
                        'ffmpeg',
                        '-f', 's16le',  # formato di input
                        '-ar', str(audio_data['sample_rate']),  # sample rate
                        '-ac', str(audio_data['channels']),  # canali
                        '-i', temp_raw,  # file di input
                        '-y',  # sovrascrivi
                        output_path  # file di output
                    ]

                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Rimuovi il file temporaneo
                    os.remove(temp_raw)
                    return True
                except Exception as ffmpeg_error:
                    print(f"Errore durante il salvataggio con ffmpeg: {ffmpeg_error}")
                    print(f"Errore originale con soundfile: {sf_error}")
                    print(f"Errore con pydub: {pydub_error}")
                    return False
    except Exception as e:
        print(f"Errore durante il salvataggio del file audio {output_path}: {e}")
        traceback.print_exc()
        return False


def make_audio_zero_mean(audio_data):
    """
    Rende l'audio a media zero. Funziona sia con array numpy che con tensori PyTorch.
    Versione semplificata rispetto a zero_mean.py
    """
    try:
        # Check if audio_data is None
        if audio_data is None:
            print("  Error: audio_data is None in make_audio_zero_mean")
            return None
            
        if audio_data.get('cuda', False):
            # Versione CUDA con PyTorch
            samples = audio_data['tensor']
            mean = torch.mean(samples.float(), dim=1, keepdim=True)
            samples = samples - mean
            audio_data['tensor'] = samples
        else:
            # Versione CPU con numpy
            samples = audio_data['array']
            mean = np.mean(samples, axis=1, keepdims=True)
            samples = samples - mean
            audio_data['array'] = samples

        return audio_data
    except Exception as e:
        print(f"Errore durante l'azzeramento della media: {e}")
        traceback.print_exc()  # Add traceback for debugging
        return None


def loop_or_truncate(audio_data, target_length):
    """
    Loop o tronca l'audio per raggiungere la lunghezza target.
    Funziona sia con array numpy che con tensori PyTorch.
    """
    try:
        # Check if audio_data is None
        if audio_data is None:
            print("  Error: audio_data is None in loop_or_truncate")
            return None
            
        if not audio_data.get('cuda', False) and 'array' not in audio_data:
            print("  Error: audio_data does not contain 'array' key")
            return None
            
        if audio_data.get('cuda', False) and 'tensor' not in audio_data:
            print("  Error: audio_data does not contain 'tensor' key")
            return None
            
        if audio_data.get('cuda', False):
            # Versione CUDA con PyTorch
            samples = audio_data['tensor']
            current_length = samples.shape[1]

            if current_length >= target_length:
                # Tronca se più lungo
                audio_data['tensor'] = samples[:, :target_length]
            else:
                # Loop se più corto
                num_repeats = target_length // current_length
                remainder = target_length % current_length

                repeated = samples.repeat(1, num_repeats)
                if remainder > 0:
                    remainder_part = samples[:, :remainder]
                    audio_data['tensor'] = torch.cat([repeated, remainder_part], dim=1)
                else:
                    audio_data['tensor'] = repeated
        else:
            # Versione CPU con numpy
            samples = audio_data['array']
            current_length = samples.shape[1]

            if current_length >= target_length:
                # Tronca se più lungo
                audio_data['array'] = samples[:, :target_length]
            else:
                # Loop se più corto
                num_repeats = target_length // current_length
                remainder = target_length % current_length

                repeated = np.tile(samples, (1, num_repeats))
                if remainder > 0:
                    remainder_part = samples[:, :remainder]
                    audio_data['array'] = np.concatenate([repeated, remainder_part], axis=1)
                else:
                    audio_data['array'] = repeated

        return audio_data
    except Exception as e:
        print(f"Errore durante il loop/troncamento dell'audio: {e}")
        traceback.print_exc()  # Add traceback for debugging
        return None


def convert_to_stereo(audio_data):
    """Converte l'audio in stereo se è mono."""
    try:
        # Check if audio_data is None
        if audio_data is None:
            print("  Error: audio_data is None in convert_to_stereo")
            return None
            
        if audio_data.get('cuda', False):
            # Versione CUDA con PyTorch
            samples = audio_data['tensor']
            if samples.shape[0] == 1:  # È mono
                audio_data['tensor'] = samples.repeat(2, 1)
                audio_data['channels'] = 2
        else:
            # Versione CPU con numpy
            samples = audio_data['array']
            if samples.shape[0] == 1:  # È mono
                audio_data['array'] = np.repeat(samples, 2, axis=0)
                audio_data['channels'] = 2

        return audio_data
    except Exception as e:
        print(f"Errore durante la conversione in stereo: {e}")
        traceback.print_exc()  # Add traceback for debugging
        return None


def extract_mixture(mp4_path, temp_dir):
    """Estrae la traccia mixture da un file MP4 usando ffmpeg."""
    try:
        output_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(mp4_path))[0]}_mixture.wav")

        # Comando ffmpeg per estrarre la prima traccia audio
        command = [
            'ffmpeg',
            '-i', mp4_path,
            '-map', '0:a:0',  # Prima traccia audio
            '-acodec', 'pcm_s16le',  # Output WAV
            '-y',  # Sovrascrivi se esiste
            output_path
        ]

        subprocess.run(command, check=True, capture_output=True)

        if os.path.exists(output_path):
            return output_path
        return None
    except Exception as e:
        print(f"Errore durante l'estrazione della traccia mixture: {e}")
        traceback.print_exc()  # Add traceback for debugging
        return None


def fix_path_separators(path):
    """Corregge i separatori di percorso in base al sistema operativo."""
    if os.name == 'nt':  # Windows
        return path.replace('/', '\\')
    else:  # Unix/Linux/Mac
        return path.replace('\\', '/')


def main():
    # Parsing degli argomenti
    parser = argparse.ArgumentParser(description='Script per sovrapporre file audio di rumore a file audio di canzoni.')
    parser.add_argument('--path-canzoni', type=str, default=SONGS_DIR,
                        help='Percorso alla directory contenente i file audio delle canzoni (MP4).')
    parser.add_argument('--path-rumori', type=str, default=NOISE_DIR,
                        help='Percorso alla directory contenente i file audio di rumore (WAV).')
    parser.add_argument('--iter-songs', type=int, default=ITER_SONGS_MAX,
                        help=f'Numero di canzoni da processare (default: {ITER_SONGS_MAX}).')
    parser.add_argument('--iter-noise', type=int, default=ITER_NOISE_MAX,
                        help=f'Numero di coppie di rumori per canzone (default: {ITER_NOISE_MAX}).')
    parser.add_argument('--use-cuda', action='store_true', help='Utilizza CUDA/GPU se disponibile.')
    parser.add_argument('--input-dir', type=str, default=INPUT_DIR, help='Directory di output per i file INPUT.')
    parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help='Directory di output per i file TARGET.')
    args = parser.parse_args()

    # Correggi i percorsi per il sistema operativo corrente
    args.path_canzoni = fix_path_separators(args.path_canzoni)
    args.path_rumori = fix_path_separators(args.path_rumori)
    args.input_dir = fix_path_separators(args.input_dir)
    args.target_dir = fix_path_separators(args.target_dir)

    # Verifica dell'esistenza delle directory
    if not os.path.isdir(args.path_canzoni):
        print(f"Errore: La directory delle canzoni '{args.path_canzoni}' non esiste.")
        sys.exit(1)

    if not os.path.isdir(args.path_rumori):
        print(f"Errore: La directory dei rumori '{args.path_rumori}' non esiste.")
        sys.exit(1)

    # Verifica della presenza di ffmpeg
    if not verify_ffmpeg():
        print("Errore: FFmpeg non disponibile. Impossibile continuare.")
        sys.exit(1)

    # Verifica dell'uso di CUDA
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        if use_cuda:
            print("INFO: CUDA è disponibile e verrà utilizzato per l'elaborazione.")
        else:
            print("AVVISO: CUDA richiesto ma non disponibile. Verrà utilizzata la CPU.")

    # Creazione delle directory di output
    input_dir = args.input_dir
    target_dir = args.target_dir
    temp_dir = os.path.join(os.getcwd(), "temp")

    for dir_path in [input_dir, target_dir, temp_dir]:
        if not clean_directory(dir_path):
            print(f"Errore: Impossibile creare/pulire la directory '{dir_path}'.")
            sys.exit(1)

    # Trova i file nelle directory
    print("Ricerca dei file di canzoni e rumori...")

    # Lista dei file delle canzoni
    canzoni_files = [os.path.join(args.path_canzoni, f) for f in os.listdir(args.path_canzoni)
                     if is_audio_file(os.path.join(args.path_canzoni, f))]
    if not canzoni_files:
        print(f"Errore: Nessun file audio trovato nella directory delle canzoni '{args.path_canzoni}'.")
        sys.exit(1)

    # Liste dei file di rumore per ogni fold
    folds = [f"fold{i}" for i in range(1, 11)]
    rumori_per_fold = {}

    for fold in folds:
        fold_path = os.path.join(args.path_rumori, fold)
        if os.path.isdir(fold_path):
            rumori_per_fold[fold] = [os.path.join(fold_path, f) for f in os.listdir(fold_path)
                                     if is_audio_file(os.path.join(fold_path, f))]
            print(f"Trovati {len(rumori_per_fold[fold])} file di rumore in {fold}")

    # Verifica la presenza di file di rumore
    if not any(rumori_per_fold.values()):
        print(f"Errore: Nessun file audio di rumore trovato nelle sottocartelle fold* di '{args.path_rumori}'.")
        sys.exit(1)
        
    # Counter delle coppie generate
    generated_pairs = 0

    # Ciclo principale di elaborazione
    songs_processed = 1
    last_song_path = None

    # Ciclo esterno (Canzoni)
    rand_iter_songs = np.random.randint(ITER_SONGS_MIN, ITER_SONGS_MAX + 1) if 1 <= ITER_SONGS_MIN < ITER_SONGS_MAX else 1
    rand_iter_noises = np.zeros(rand_iter_songs)
    
    for i in range(rand_iter_songs):
        rand_iter_noises[i] = np.random.randint(ITER_NOISE_MIN, ITER_NOISE_MIN + 1) if 1 <= ITER_SONGS_MIN < ITER_NOISE_MAX else 1
        
    print(f"\nAvvio elaborazione di {rand_iter_songs} canzoni...")

    while songs_processed <= rand_iter_songs:
        # Seleziona una canzone casuale diversa dalla precedente
        while True:
            song_path = random.choice(canzoni_files)
            if song_path != last_song_path or len(canzoni_files) == 1:
                break

        last_song_path = song_path
        song_name = os.path.splitext(os.path.basename(song_path))[0]
        print(f"\n--- Elaborazione canzone {songs_processed}/{rand_iter_songs}: {song_name} ---")

        # Estrai la traccia mixture per le canzoni MP4
        if song_path.lower().endswith('.mp4'):
            print(f"Estrazione della traccia mixture da {song_name}...")
            extracted_path = extract_mixture(song_path, temp_dir)
            if not extracted_path:
                print(
                    f"Errore: Impossibile estrarre la traccia mixture da {song_path}. Passaggio alla canzone successiva.")
                continue
            song_path = extracted_path
            song_name = os.path.splitext(os.path.basename(song_path))[0].replace('.stem_mixture', '')

        # Carica la canzone
        print(f"Caricamento della canzone: {song_name}...")
        song_data = load_audio(song_path, use_cuda)
        if song_data is None:
            print(f"Errore: Impossibile caricare il file {song_path}. Passaggio alla canzone successiva.")
            continue

        # Verifica che la canzone sia in stereo o convertila
        song_data = convert_to_stereo(song_data)
        if song_data is None:
            print(f"Errore: Impossibile convertire la canzone in stereo. Passaggio alla canzone successiva.")
            continue

        # Ottieni la lunghezza della canzone
        song_length = song_data['tensor'].shape[1] if use_cuda else song_data['array'].shape[1]

        # Ciclo interno (Rumori)
        noise_pairs_processed = 1
        last_noise_pair = None
        
        rand_iter_noise = int(rand_iter_noises[songs_processed-1])

        print(f"Generazione di {rand_iter_noise} coppie di rumori per questa canzone...")
        generated_pairs += rand_iter_noise

        while noise_pairs_processed <= rand_iter_noise:
            # Per la modalità test, basta un solo rumore
            noise2_path = noise2_data = noise2_name = None
            if IS_TRAINING:
                # Modalità train: seleziona due fold casuali diversi
                available_folds = [fold for fold in rumori_per_fold.keys() if rumori_per_fold[fold]]
                if len(available_folds) < 2:
                    print("Errore: Non ci sono abbastanza fold con file di rumore. Impossibile continuare.")
                    sys.exit(1)

                fold1, fold2 = random.sample(available_folds, 2)
                
                # Seleziona un rumore casuale da ciascun fold
                noise1_path = random.choice(rumori_per_fold[fold1])
                noise2_path = random.choice(rumori_per_fold[fold2])

                # Verifica che la coppia di rumori sia diversa dalla precedente
                current_noise_pair = (noise1_path, noise2_path)
                if current_noise_pair == last_noise_pair and noise_pairs_processed > 1:
                    continue

                last_noise_pair = current_noise_pair
                
                noise1_name = os.path.splitext(os.path.basename(noise1_path))[0]
                noise2_name = os.path.splitext(os.path.basename(noise2_path))[0]

                print(f"\n  Coppia di rumori {noise_pairs_processed}/{rand_iter_noise} "
                    f"(canzone {songs_processed}/{rand_iter_songs}):")
                print(f"  - Rumore 1: {noise1_name} (da {fold1})")
                print(f"  - Rumore 2: {noise2_name} (da {fold2})")
                
            else:
                # Modalità test: seleziona un solo fold casuale
                available_folds = [fold for fold in rumori_per_fold.keys() if rumori_per_fold[fold]]
                if not available_folds:
                    print("Errore: Non ci sono fold con file di rumore. Impossibile continuare.")
                    sys.exit(1)

                fold1 = random.choice(available_folds)
                
                # Seleziona un rumore casuale dal fold
                noise1_path = random.choice(rumori_per_fold[fold1])
                
                # Verifica che il rumore sia diverso dal precedente
                if noise1_path == last_noise_pair and noise_pairs_processed > 1:
                    continue

                last_noise_pair = noise1_path
                
                noise1_name = os.path.splitext(os.path.basename(noise1_path))[0]

                print(f"\n  Rumore {noise_pairs_processed}/{rand_iter_noise} "
                    f"(canzone {songs_processed}/{rand_iter_songs}):")
                print(f"  - Rumore: {noise1_name} (da {fold1})")

            # Carica il primo rumore
            print(f"  Caricamento del rumore 1: {noise1_path}")
            noise1_data = load_audio(noise1_path, use_cuda)
            if noise1_data is None:
                print(f"  Errore: Impossibile caricare il rumore 1. Tentativo con un altro rumore.")
                continue

            # Verifica che il rumore sia in stereo o convertilo
            noise1_data = convert_to_stereo(noise1_data)

            # Azzera la media del rumore
            noise1_data = make_audio_zero_mean(noise1_data)
            if noise1_data is None:
                print("  Errore: Impossibile azzerare la media del rumore 1. Passaggio alla coppia successiva.")
                continue
            
            # Normalizza i dati di rumore rispetto ai dati puliti con un SNR casuale tra 0 e 10 dB.
            noise1_data = normalize_db(noise1_data, song_data)

            # Adatta la lunghezza del rumore alla canzone
            noise1_data = loop_or_truncate(noise1_data, song_length)
            if noise1_data is None:
                print("  Errore: Impossibile adattare la lunghezza del rumore 1. Passaggio alla coppia successiva.")
                continue

            # Per la modalità train, processa anche il secondo rumore
            if IS_TRAINING:
                print(f"  Caricamento del rumore 2: {noise2_path}")
                noise2_data = load_audio(noise2_path, use_cuda)
                if noise2_data is None:
                    print(f"  Errore: Impossibile caricare il rumore 2. Tentativo con un altro rumore.")
                    continue

                # Verifica che il rumore sia in stereo o convertilo
                noise2_data = convert_to_stereo(noise2_data)
                
                # Normalizza i dati di rumore rispetto ai dati puliti con un SNR casuale tra 0 e 10 dB.
                noise2_data = normalize_db(noise2_data, song_data)

                # Azzera la media del rumore
                noise2_data = make_audio_zero_mean(noise2_data)
                if noise2_data is None:
                    print("  Errore: Impossibile azzerare la media del rumore 2. Passaggio alla coppia successiva.")
                    continue

                # Adatta la lunghezza del rumore alla canzone
                noise2_data = loop_or_truncate(noise2_data, song_length)
                if noise2_data is None:
                    print("  Errore: Impossibile adattare la lunghezza del rumore 2. Passaggio alla coppia successiva.")
                    continue

            # Sovrapponi i rumori alla canzone
            if use_cuda:
                # Versione CUDA
                input_samples = song_data['tensor'] + noise1_data['tensor']
                
                if IS_TRAINING:
                    target_samples = song_data['tensor'] + noise2_data['tensor']
                else:
                    # In modalità test, il target è la canzone originale senza rumore
                    target_samples = song_data['tensor']

                input_audio = {
                    'tensor': input_samples,
                    'sample_rate': song_data['sample_rate'],
                    'channels': song_data['channels'],
                    'sample_width': song_data['sample_width'],
                    'cuda': True
                }

                target_audio = {
                    'tensor': target_samples,
                    'sample_rate': song_data['sample_rate'],
                    'channels': song_data['channels'],
                    'sample_width': song_data['sample_width'],
                    'cuda': True
                }
            else:
                # Versione CPU
                input_samples = song_data['array'] + noise1_data['array']
                
                if IS_TRAINING:
                    target_samples = song_data['array'] + noise2_data['array']
                else:
                    # In modalità test, il target è la canzone originale senza rumore
                    target_samples = song_data['array']

                input_audio = {
                    'array': input_samples,
                    'sample_rate': song_data['sample_rate'],
                    'channels': song_data['channels'],
                    'sample_width': song_data['sample_width'],
                    'cuda': False
                }

                target_audio = {
                    'array': target_samples,
                    'sample_rate': song_data['sample_rate'],
                    'channels': song_data['channels'],
                    'sample_width': song_data['sample_width'],
                    'cuda': False
                }

            # Crea i nomi dei file di output
            noise1_base_name = slugify(noise1_name)
            song_base_name = slugify(song_name)

            if IS_TRAINING:
                noise2_base_name = slugify(noise2_name)
                input_filename = f"INPUT-S{songs_processed}N{noise_pairs_processed}-[{noise1_base_name}]-[{song_base_name}].wav"
                target_filename = f"TARGET-S{songs_processed}N{noise_pairs_processed}-[{noise2_base_name}]-[{song_base_name}].wav"
            else:
                input_filename = f"INPUT-S{songs_processed}N{noise_pairs_processed}-[{noise1_base_name}]-[{song_base_name}].wav"
                target_filename = f"TARGET-S{songs_processed}N{noise_pairs_processed}-[CLEAN]-[{song_base_name}].wav"

            input_path = os.path.join(input_dir, input_filename)
            target_path = os.path.join(target_dir, target_filename)

            # Salva i file di output
            print(f"  Salvataggio dei file di output...")

            if save_audio(input_audio, input_path) and save_audio(target_audio, target_path):
                print(f"  File salvati con successo:\n    - {input_filename}\n    - {target_filename}")
                noise_pairs_processed += 1
            else:
                print("  Errore durante il salvataggio dei file. Passaggio alla coppia successiva.")

        # Incrementa il contatore delle canzoni processate
        songs_processed += 1

    # Pulizia della directory temporanea
    try:
        shutil.rmtree(temp_dir)
        print(f"\nDirectory temporanea '{temp_dir}' rimossa.")
    except Exception as e:
        print(f"\nAvviso: Impossibile rimuovere la directory temporanea '{temp_dir}': {e}")

    print("\nElaborazione completata con successo!")
    parent_dir = os.path.dirname(os.path.dirname(INPUT_DIR))  # Ottiene la directory padre che contiene train/test
    log_generation_stats(IS_TRAINING, generated_pairs, parent_dir)


if __name__ == "__main__":
    main()