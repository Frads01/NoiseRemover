#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zipfile
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
import torch
import unicodedata
from pydub import AudioSegment
from pydub import effects

# === PARAMETRI GLOBALI ===

IS_TRAINING = True
DEF_CLASS_NUM = 9
CLASS_NUM = DEF_CLASS_NUM
# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music

ITER_SONGS_MIN = 5 if IS_TRAINING else 3
ITER_SONGS_MAX = 10 if IS_TRAINING else 5

ITER_NOISE_MIN = 10 if IS_TRAINING else 5
ITER_NOISE_MAX = 20 if IS_TRAINING else 8

DATASET_DIR = f".\\dataset_class\\dataset_n{CLASS_NUM}"
INPUT_DIR = os.path.join(DATASET_DIR, "train", "input") if IS_TRAINING else os.path.join(DATASET_DIR, "test", "input")
TARGET_DIR = os.path.join(DATASET_DIR, "train", "target") if IS_TRAINING else os.path.join(DATASET_DIR, "test", "target")
SONGS_DIR = ".\\musdb18\\train" if IS_TRAINING else ".\\musdb18\\test"
NOISE_DIR = ".\\UrbanSound8K\\audio"

used_songs_file = f".\\dataset_class\\dataset_n{CLASS_NUM}\\used_songs_{CLASS_NUM}.txt"
used_noises_file = f".\\dataset_class\\dataset_n{CLASS_NUM}\\used_noises_{CLASS_NUM}.txt"

# === FUNZIONI DI UTILITÀ ===


def slugify(text):
    """Converte un testo in uno slug URL-friendly."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text).strip('-_')
    return text


def get_noise_category(noise_filename):
    """Estrae la categoria del rumore dal nome file UrbanSound8K."""
    parts = os.path.splitext(noise_filename)[0].split('-')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def log_generation_stats(is_training, num_pairs, parent_dir):
    """Registra le statistiche di generazione in un file di log."""
    import datetime
    
    log_file = os.path.join(parent_dir, "log.txt")
    
    input_size = 0
    target_size = 0
    input_files = 0
    target_files = 0
    attempt = 1

    if is_training:
        session_type = "TRAIN"
        input_dir = os.path.join(parent_dir, "train", "input")
        target_dir = os.path.join(parent_dir, "train", "target")
    else:
        session_type = "TEST"
        input_dir = os.path.join(parent_dir, "test", "input")
        target_dir = os.path.join(parent_dir, "test", "target")

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
    total_size_mb = total_size / (1024 * 1024)
    total_files = input_files + target_files

    current_date = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            pattern = fr"--- DATASET for {session_type} #(\d+)- {current_date}"
            attempts = re.findall(pattern, log_content)
            if attempts:
                attempt = max(map(int, attempts)) + 1

    title = f"--- DATASET for {session_type} #{attempt}- {current_date} ---"
    separator = "-" * len(title)

    log_message = f"""{title}

    Generated pairs: {num_pairs}
    Generated files: {total_files}
    Output size: {total_size_mb:.2f} MB
{separator}

"""

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message)

    print(f"Log scritto in: {log_file}")
    return [num_pairs, total_files, total_size_mb]
    

def clean_directory(dir_path):
    """Pulisce o crea la directory di output se non esiste."""
    try:
        if os.path.exists(dir_path):
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Pulita directory esistente: {dir_path}")
        else:
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
    """Carica un file audio utilizzando librerie di supporto appropriate."""
    try:
        try:
            data, sample_rate = sf.read(file_path, always_2d=True)
            if data.shape[1] <= 2:
                data = data.T

            channels = data.shape[0]
            sample_width = 2

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
            try:
                audio = AudioSegment.from_file(file_path)
                audio = effects.normalize(audio)
                samples = np.array(audio.get_array_of_samples())

                if audio.channels == 1:
                    samples = samples.reshape(1, -1)
                else:
                    samples = samples.reshape(-1, audio.channels).T

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
                try:
                    temp_dir = os.path.join(os.getcwd(), "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_wav = os.path.join(temp_dir, f"temp_{os.path.basename(file_path)}.wav")

                    command = [
                        'ffmpeg',
                        '-i', file_path,
                        '-acodec', 'pcm_s16le',
                        '-y',
                        temp_wav
                    ]

                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    with wave.open(temp_wav, 'rb') as wf:
                        channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                        sample_rate = wf.getframerate()
                        frames = wf.getnframes()

                        buffer = wf.readframes(frames)
                        samples = np.frombuffer(buffer, dtype=np.int16)

                        if channels > 1:
                            samples = samples.reshape(-1, channels).T
                        else:
                            samples = samples.reshape(1, -1)

                    os.remove(temp_wav)

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
    """Normalizza il rumore rispetto al segnale pulito con approccio SNR intelligente."""
    try:
        if noise_data is None:
            print("  Error: noise_data is None in normalize_db function")
            return None

        if clean_data is None:
            print("  Error: clean_data is None in normalize_db function")
            return None

        min_snr = 2
        max_snr = 27

        if noise_data.get('cuda', False):
            noise_samples = noise_data['tensor']
            clean_samples = clean_data['tensor']

            clean_rms = torch.sqrt(torch.mean(clean_samples.float() ** 2))
            noise_rms = torch.sqrt(torch.mean(noise_samples.float() ** 2))

            if clean_samples.shape[1] > 0:
                chunk_size = min(44100, clean_samples.shape[1])
                num_chunks = max(1, clean_samples.shape[1] // chunk_size)
                chunks_rms = []

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, clean_samples.shape[1])
                    chunk = clean_samples[:, start:end]
                    chunk_rms = torch.sqrt(torch.mean(chunk.float() ** 2))
                    chunks_rms.append(chunk_rms.item())

                dynamic_range = max(chunks_rms) / (min(chunks_rms) + 1e-10)

                if dynamic_range > 10:
                    min_snr = 4
                    max_snr = 30
                elif dynamic_range < 3:
                    min_snr = 0
                    max_snr = 25
            else:
                min_snr = 2
                max_snr = 27

            target_snr = min_snr + torch.rand(1).item() * (max_snr - min_snr)
            scaling_factor = clean_rms / (noise_rms * (10 ** (target_snr / 20)))
            noise_data['tensor'] = noise_samples * scaling_factor

        else:
            noise_samples = noise_data['array']
            clean_samples = clean_data['array']

            clean_rms = np.sqrt(np.mean(clean_samples ** 2))
            noise_rms = np.sqrt(np.mean(noise_samples ** 2))

            if clean_samples.shape[1] > 0:
                chunk_size = min(44100, clean_samples.shape[1])
                num_chunks = max(1, clean_samples.shape[1] // chunk_size)
                chunks_rms = []

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, clean_samples.shape[1])
                    chunk = clean_samples[:, start:end]
                    chunk_rms = np.sqrt(np.mean(chunk ** 2))
                    chunks_rms.append(chunk_rms)

                dynamic_range = max(chunks_rms) / (min(chunks_rms) + 1e-10)

                if dynamic_range > 10:
                    min_snr = 4
                    max_snr = 30
                elif dynamic_range < 3:
                    min_snr = 0
                    max_snr = 25
            else:
                min_snr = 2
                max_snr = 27

            target_snr = min_snr + np.random.random() * (max_snr - min_snr)
            scaling_factor = clean_rms / (noise_rms * (10 ** (target_snr / 20)))
            noise_data['array'] = noise_samples * scaling_factor

        print(f"  Noise normalized with target SNR: {target_snr:.2f} dB (dynamic range-adjusted)")
        return noise_data

    except Exception as e:
        print(f"  Error during noise normalization: {e}")
        traceback.print_exc()
        return None

def save_audio(audio_data, output_path, format_='wav'):
    """Salva i dati audio in un file, supportando sia dati numpy che tensori PyTorch."""
    try:
        if audio_data.get('cuda', False):
            samples = audio_data['tensor'].cpu().numpy()
        else:
            samples = audio_data['array']

        if samples.dtype != np.int16:
            if samples.dtype == np.float32 or samples.dtype == np.float64:
                samples = np.clip(samples, -1.0, 1.0)
                samples = (samples * 32767).astype(np.int16)
            else:
                samples = np.clip(samples, -1.0, 1.0)
                samples = samples.astype(np.int16)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            sf_samples = samples.T
            sf.write(output_path, sf_samples, audio_data['sample_rate'])
            return True
        except Exception as sf_error:
            try:
                if samples.ndim > 1:
                    pydub_samples = samples.T.reshape(-1)
                else:
                    pydub_samples = samples

                segment = AudioSegment(
                    pydub_samples.tobytes(),
                    frame_rate=audio_data['sample_rate'],
                    sample_width=audio_data['sample_width'],
                    channels=audio_data['channels']
                )

                segment.export(output_path, format=format_)
                return True
            except Exception as pydub_error:
                try:
                    temp_dir = os.path.join(os.getcwd(), "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_raw = os.path.join(temp_dir, "temp_raw.pcm")

                    with open(temp_raw, 'wb') as f:
                        if samples.ndim > 1:
                            samples_interleaved = samples.T.reshape(-1)
                        else:
                            samples_interleaved = samples
                        f.write(samples_interleaved.tobytes())

                    command = [
                        'ffmpeg',
                        '-f', 's16le',
                        '-ar', str(audio_data['sample_rate']),
                        '-ac', str(audio_data['channels']),
                        '-i', temp_raw,
                        '-y',
                        output_path
                    ]

                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    """Rende l'audio a media zero."""
    try:
        if audio_data is None:
            print("  Error: audio_data is None in make_audio_zero_mean")
            return None

        if audio_data.get('cuda', False):
            samples = audio_data['tensor']
            mean = torch.mean(samples.float(), dim=1, keepdim=True)
            samples = samples - mean
            audio_data['tensor'] = samples
        else:
            samples = audio_data['array']
            mean = np.mean(samples, axis=1, keepdims=True)
            samples = samples - mean
            audio_data['array'] = samples

        return audio_data
    except Exception as e:
        print(f"Errore durante l'azzeramento della media: {e}")
        traceback.print_exc()
        return None

def loop_or_truncate(audio_data, target_length):
    """Loop o tronca l'audio per raggiungere la lunghezza target."""
    try:
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
            samples = audio_data['tensor']
            current_length = samples.shape[1]

            if current_length >= target_length:
                audio_data['tensor'] = samples[:, :target_length]
            else:
                num_repeats = target_length // current_length
                remainder = target_length % current_length

                repeated = samples.repeat(1, num_repeats)
                if remainder > 0:
                    remainder_part = samples[:, :remainder]
                    audio_data['tensor'] = torch.cat([repeated, remainder_part], dim=1)
                else:
                    audio_data['tensor'] = repeated
        else:
            samples = audio_data['array']
            current_length = samples.shape[1]

            if current_length >= target_length:
                audio_data['array'] = samples[:, :target_length]
            else:
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
        traceback.print_exc()
        return None

def convert_to_stereo(audio_data):
    """Converte l'audio in stereo se è mono."""
    try:
        if audio_data is None:
            print("  Error: audio_data is None in convert_to_stereo")
            return None

        if audio_data.get('cuda', False):
            samples = audio_data['tensor']
            if samples.shape[0] == 1:
                audio_data['tensor'] = samples.repeat(2, 1)
                audio_data['channels'] = 2
        else:
            samples = audio_data['array']
            if samples.shape[0] == 1:
                audio_data['array'] = np.repeat(samples, 2, axis=0)
                audio_data['channels'] = 2

        return audio_data
    except Exception as e:
        print(f"Errore durante la conversione in stereo: {e}")
        traceback.print_exc()
        return None

def extract_mixture(mp4_path, temp_dir):
    """Estrae la traccia mixture da un file MP4 usando ffmpeg."""
    try:
        output_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(mp4_path))[0]}_mixture.wav")

        command = [
            'ffmpeg',
            '-i', mp4_path,
            '-map', '0:a:0',
            '-acodec', 'pcm_s16le',
            '-y',
            output_path
        ]

        subprocess.run(command, check=True, capture_output=True)

        if os.path.exists(output_path):
            return output_path
        return None
    except Exception as e:
        print(f"Errore durante l'estrazione della traccia mixture: {e}")
        traceback.print_exc()
        return None

def fix_path_separators(path):
    """Corregge i separatori di percorso in base al sistema operativo."""
    if os.name == 'nt':
        return path.replace('/', '\\')
    else:
        return path.replace('\\', '/')

def main():
    def backup_file(path):
        if os.path.exists(path):
            shutil.copy2(path, path + '.bak')

    def restore_file(path):
        bak = path + '.bak'
        if os.path.exists(bak):
            shutil.copy2(bak, path)
            os.remove(bak)

    def remove_backup(path):
        bak = path + '.bak'
        if os.path.exists(bak):
            os.remove(bak)

    try:
        parser = argparse.ArgumentParser(description='Script per sovrapporre file audio di rumore a file audio di canzoni.')
        parser.add_argument('--class-num', type=int, default=DEF_CLASS_NUM, choices=range(0, 10))
        parser.add_argument('--path-songs', type=str, default=SONGS_DIR)
        parser.add_argument('--path-noises', type=str, default=NOISE_DIR)
        parser.add_argument('--iter-songs', type=int, default=ITER_SONGS_MAX)
        parser.add_argument('--iter-noise', type=int, default=ITER_NOISE_MAX)
        parser.add_argument('--use-cuda', action='store_true')
        parser.add_argument('--input-dir', type=str, default=INPUT_DIR)
        parser.add_argument('--target-dir', type=str, default=TARGET_DIR)
        args = parser.parse_args()

        args.path_songs = fix_path_separators(args.path_songs)
        args.path_noises = fix_path_separators(args.path_noises)
        args.input_dir = fix_path_separators(args.input_dir)
        args.target_dir = fix_path_separators(args.target_dir)
        
        global CLASS_NUM
        CLASS_NUM = args.class_num

        if not os.path.isdir(args.path_songs):
            print(f"Errore: La directory delle canzoni '{args.path_songs}' non esiste.")
            sys.exit(1)

        if not os.path.isdir(args.path_noises):
            print(f"Errore: La directory dei rumori '{args.path_noises}' non esiste.")
            sys.exit(1)

        if not verify_ffmpeg():
            print("Errore: FFmpeg non disponibile. Impossibile continuare.")
            sys.exit(1)

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if args.use_cuda:
            if use_cuda:
                print("INFO: CUDA è disponibile e verrà utilizzato per l'elaborazione.")
            else:
                print("AVVISO: CUDA richiesto ma non disponibile. Verrà utilizzata la CPU.")

        input_dir = args.input_dir
        target_dir = args.target_dir
        temp_dir = os.path.join(os.getcwd(), "temp")

        for dir_path in [input_dir, target_dir, temp_dir]:
            if not clean_directory(dir_path):
                print(f"Errore: Impossibile creare/pulire la directory '{dir_path}'.")
                sys.exit(1)

        print("Ricerca dei file di canzoni e rumori...")

        canzoni_files = [os.path.join(args.path_songs, f) for f in os.listdir(args.path_songs)
                         if is_audio_file(os.path.join(args.path_songs, f))]
        if not canzoni_files:
            print(f"Errore: Nessun file audio trovato nella directory delle canzoni '{args.path_songs}'.")
            sys.exit(1)
            
        os.makedirs(os.path.dirname(used_songs_file), exist_ok=True)
        os.makedirs(os.path.dirname(used_noises_file), exist_ok=True)
        
        for file_path in [used_songs_file, used_noises_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    pass

        # CARICAMENTO RUMORI SOLO DELLA CLASSE SPECIFICATA
        folds = [f"fold{i}" for i in range(1, 11)]
        rumori_per_fold = {}
        
        rumori_altre_categorie = []
        for fold in folds:
            fold_path = os.path.join(args.path_noises, fold)
            if os.path.isdir(fold_path):
                rumori_altre_categorie += [
                    os.path.join(fold_path, f)
                    for f in os.listdir(fold_path)
                    if is_audio_file(os.path.join(fold_path, f)) and get_noise_category(f) != CLASS_NUM
                ]
        print(f"Trovati {len(rumori_altre_categorie)} file di rumore NON della categoria {CLASS_NUM}")

        for fold in folds:
            fold_path = os.path.join(args.path_noises, fold)
            if os.path.isdir(fold_path):
                rumori_per_fold[fold] = [
                    os.path.join(fold_path, f)
                    for f in os.listdir(fold_path)
                    if is_audio_file(os.path.join(fold_path, f)) and get_noise_category(f) == CLASS_NUM
                ]
                print(f"Trovati {len(rumori_per_fold[fold])} file di rumore della categoria {CLASS_NUM} in {fold}")

        if not any(rumori_per_fold.values()):
            print(f"Errore: Nessun file audio di rumore della categoria {CLASS_NUM} trovato.")
            sys.exit(1)

        generated_pairs = 0
        songs_processed = 1

        backup_file(used_songs_file)
        backup_file(used_noises_file)

        used_songs = []
        if os.path.exists(used_songs_file):
            with open(used_songs_file, 'r') as f:
                used_songs = [line.strip() for line in f.readlines()]

        rand_iter_songs = np.random.randint(ITER_SONGS_MIN, ITER_SONGS_MAX + 1) if 1 <= ITER_SONGS_MIN < ITER_SONGS_MAX else 1
        rand_iter_noises = np.zeros(rand_iter_songs)

        for i in range(rand_iter_songs):
            rand_iter_noises[i] = np.random.randint(ITER_NOISE_MIN, ITER_NOISE_MAX + 1) if 1 <= ITER_SONGS_MIN < ITER_NOISE_MAX else 1

        print(f"\nAvvio elaborazione di {rand_iter_songs} canzoni...")

        while songs_processed <= rand_iter_songs:
            available_songs = [song for song in canzoni_files if os.path.basename(song) not in used_songs]

            if not available_songs and canzoni_files:
                song_path = random.choice(canzoni_files)
            elif available_songs:
                song_path = random.choice(available_songs)
            else:
                print("Errore: Nessuna canzone disponibile per l'elaborazione.")
                sys.exit(1)

            song_basename = os.path.basename(song_path)

            with open(used_songs_file, 'r') as f:
                existing_songs = f.readlines()

            songs_count = len(existing_songs)

            if songs_count >= 75:
                existing_songs = existing_songs[1:]
                with open(used_songs_file, 'w') as f:
                    f.writelines(existing_songs)
                    f.write(f"{song_basename}\n")
                print(f"Rimossa la prima riga dal file. Mantenute {len(existing_songs) + 1} righe totali.")
            else:
                with open(used_songs_file, 'a') as f:
                    f.write(f"{song_basename}\n")

            used_songs.append(song_basename)

            song_name = os.path.splitext(os.path.basename(song_path))[0]
            print(f"\n--- Elaborazione canzone {songs_processed}/{rand_iter_songs}: {song_name} ---")

            if song_path.lower().endswith('.mp4'):
                print(f"Estrazione della traccia mixture da {song_name}...")
                extracted_path = extract_mixture(song_path, temp_dir)
                if not extracted_path:
                    print(f"Errore: Impossibile estrarre la traccia mixture da {song_path}. Passaggio alla canzone successiva.")
                    continue
                song_path = extracted_path
                song_name = os.path.splitext(os.path.basename(song_path))[0].replace('.stem_mixture', '')

            print(f"Caricamento della canzone: {song_name}...")
            song_data = load_audio(song_path, use_cuda)
            if song_data is None:
                print(f"Errore: Impossibile caricare il file {song_path}. Passaggio alla canzone successiva.")
                continue

            song_data = convert_to_stereo(song_data)
            if song_data is None:
                print(f"Errore: Impossibile convertire la canzone in stereo. Passaggio alla canzone successiva.")
                continue

            song_length = song_data['tensor'].shape[1] if use_cuda else song_data['array'].shape[1]

            noise_pairs_processed = 1
            used_noises = []
            rand_iter_noise = int(rand_iter_noises[songs_processed - 1])

            print(f"Generazione di {rand_iter_noise} coppie di rumori per questa canzone...")
            generated_pairs += rand_iter_noise

            while noise_pairs_processed <= rand_iter_noise:
                noise1_data = None
                noise2_data = None
                noise1_name = ""
                noise2_name = ""

                print(f"\n  Coppia di rumori {noise_pairs_processed}/{rand_iter_noise} "
                      f"(canzone {songs_processed}/{rand_iter_songs}):")

                # PRIMO RUMORE - SOLO DELLA CATEGORIA CLASS_NUM
                available_folds = [fold for fold in rumori_per_fold.keys() if rumori_per_fold[fold]]
                if not available_folds:
                    print("Errore: Non ci sono fold con file di rumore. Impossibile continuare.")
                    sys.exit(1)

                fold1 = random.choice(available_folds)

                fold1_available = [noise for noise in rumori_per_fold[fold1]
                                   if os.path.basename(noise) not in used_noises]
                if not fold1_available and rumori_per_fold[fold1]:
                    noise1_path = random.choice(rumori_per_fold[fold1])
                elif fold1_available:
                    noise1_path = random.choice(fold1_available)
                else:
                    print(f"Errore: Nessun rumore disponibile nel fold {fold1}.")
                    continue

                noise1_basename = os.path.basename(noise1_path)

                existing_noises = []
                if os.path.exists(used_noises_file):
                    with open(used_noises_file, 'r') as f:
                        existing_noises = f.readlines()

                noises_count = len(existing_noises)

                if noises_count >= 4000:
                    existing_noises = existing_noises[1:]
                    with open(used_noises_file, 'w') as f:
                        f.writelines(existing_noises)
                        f.write(f"{noise1_basename}\n")
                    print(f"Rimossa la prima riga dal file rumori. Mantenute {len(existing_noises) + 1} righe totali.")
                else:
                    with open(used_noises_file, 'a') as f:
                        f.write(f"{noise1_basename}\n")

                used_noises.append(noise1_basename)
                noise1_name = os.path.splitext(os.path.basename(noise1_path))[0]
                print(f"  - Rumore 1: {noise1_name} (da {fold1})")

                print(f"  Caricamento del rumore 1: {noise1_path}")
                noise1_data = load_audio(noise1_path, use_cuda)
                if noise1_data is None:
                    print(f"  Errore: Impossibile caricare il rumore 1. Tentativo con un altro rumore.")
                    continue

                noise1_data = convert_to_stereo(noise1_data)
                noise1_data = make_audio_zero_mean(noise1_data)
                if noise1_data is None:
                    print("  Errore: Impossibile azzerare la media del rumore 1. Passaggio alla coppia successiva.")
                    continue

                noise1_data = normalize_db(noise1_data, song_data)
                noise1_data = loop_or_truncate(noise1_data, song_length)
                if noise1_data is None:
                    print("  Errore: Impossibile adattare la lunghezza del rumore 1. Passaggio alla coppia successiva.")
                    continue

                # SECONDO RUMORE - NON DELLA CATEGORIA CLASS_NUM (per training)
                if IS_TRAINING:
                    # Raccogli tutti i rumori NON della categoria CLASS_NUM da tutti i fold
                    rumori_altre_categorie = []
                    for fold in rumori_per_fold.keys():
                        fold_path = os.path.join(args.path_noises, fold)
                        if os.path.isdir(fold_path):
                            rumori_altre_categorie += [
                                os.path.join(fold_path, f)
                                for f in os.listdir(fold_path)
                                if is_audio_file(os.path.join(fold_path, f)) and get_noise_category(f) != CLASS_NUM
                            ]

                    # Escludi i rumori già usati in questa canzone
                    available_noises2 = [noise for noise in rumori_altre_categorie if os.path.basename(noise) not in used_noises]
                    if not available_noises2 and rumori_altre_categorie:
                        noise2_path = random.choice(rumori_altre_categorie)
                    elif available_noises2:
                        noise2_path = random.choice(available_noises2)
                    else:
                        print("Errore: Nessun rumore disponibile NON della categoria CLASS_NUM.")
                        continue

                    noise2_basename = os.path.basename(noise2_path)

                    existing_noises = []
                    if os.path.exists(used_noises_file):
                        with open(used_noises_file, 'r') as f:
                            existing_noises = f.readlines()

                    noises_count = len(existing_noises)

                    if noises_count >= 4000:
                        existing_noises = existing_noises[1:]
                        with open(used_noises_file, 'w') as f:
                            f.writelines(existing_noises)
                            f.write(f"{noise2_basename}\n")
                        print(f"Rimossa la prima riga dal file rumori. Mantenute {len(existing_noises) + 1} righe totali.")
                    else:
                        with open(used_noises_file, 'a') as f:
                            f.write(f"{noise2_basename}\n")

                    used_noises.append(noise2_basename)
                    noise2_name = os.path.splitext(os.path.basename(noise2_path))[0]
                    print(f"  - Rumore 2: {noise2_name} (categoria diversa da {CLASS_NUM})")

                    print(f"  Caricamento del rumore 2: {noise2_path}")
                    noise2_data = load_audio(noise2_path, use_cuda)
                    if noise2_data is None:
                        print(f"  Errore: Impossibile caricare il rumore 2. Tentativo con un altro rumore.")
                        continue

                    noise2_data = convert_to_stereo(noise2_data)
                    noise2_data = normalize_db(noise2_data, song_data)
                    noise2_data = make_audio_zero_mean(noise2_data)
                    if noise2_data is None:
                        print("  Errore: Impossibile azzerare la media del rumore 2. Passaggio alla coppia successiva.")
                        continue

                    noise2_data = loop_or_truncate(noise2_data, song_length)
                    if noise2_data is None:
                        print("  Errore: Impossibile adattare la lunghezza del rumore 2. Passaggio alla coppia successiva.")
                        continue
                    
                if os.path.exists(used_noises_file):
                    with open(used_noises_file, 'r') as f:
                        noises_count = len(f.readlines())

                    if noises_count >= 4000:
                        open(used_noises_file, 'w').close()
                        used_noises = []

                # SOVRAPPOSIZIONE
                if use_cuda:
                    input_samples = song_data['tensor'] + noise1_data['tensor']

                    if IS_TRAINING:
                        target_samples = song_data['tensor'] + noise2_data['tensor']
                    else:
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
                    input_samples = song_data['array'] + noise1_data['array']

                    if IS_TRAINING:
                        target_samples = song_data['array'] + noise2_data['array']
                    else:
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

                # NOMI FILE
                noise1_base_name = slugify(noise1_name)
                song_base_name = slugify(song_name)

                if IS_TRAINING:
                    noise2_base_name = slugify(noise2_name)
                    input_filename = f"INPUT-S{songs_processed}N{noise_pairs_processed}-({noise1_base_name})-({song_base_name}).wav"
                    target_filename = f"TARGET-S{songs_processed}N{noise_pairs_processed}-({noise2_base_name})-({song_base_name}).wav"
                else:
                    input_filename = f"INPUT-S{songs_processed}N{noise_pairs_processed}-({noise1_base_name})-({song_base_name}).wav"
                    target_filename = f"TARGET-S{songs_processed}N{noise_pairs_processed}-(CLEAN)-({song_base_name}).wav"

                input_path = os.path.join(input_dir, input_filename)
                target_path = os.path.join(target_dir, target_filename)

                print(f"  Salvataggio dei file di output...")

                if save_audio(input_audio, input_path) and save_audio(target_audio, target_path):
                    print(f"  File salvati con successo:\n    - {input_filename}\n    - {target_filename}")
                    noise_pairs_processed += 1
                else:
                    print("  Errore durante il salvataggio dei file. Passaggio alla coppia successiva.")

            songs_processed += 1

        try:
            shutil.rmtree(temp_dir)
            print(f"\nDirectory temporanea '{temp_dir}' rimossa.")
        except Exception as e:
            print(f"\nAvviso: Impossibile rimuovere la directory temporanea '{temp_dir}': {e}")

        print("\nElaborazione completata con successo!")
        parent_dir = os.path.dirname(os.path.dirname(INPUT_DIR))
        stats = log_generation_stats(IS_TRAINING, generated_pairs, parent_dir)
        print(stats[1], stats[2])
    except (Exception, KeyboardInterrupt):
        print("\nERRORE o interruzione! Ripristino dei file di uso...")
        traceback.print_exc()
        restore_file(used_songs_file)
        restore_file(used_noises_file)
        sys.exit(1)
    finally:
        remove_backup(used_songs_file)
        remove_backup(used_noises_file)

if __name__ == "__main__":
    main()
