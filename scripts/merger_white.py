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
import torch  # Make sure torch is imported globally so it's accessible in all functions
import unicodedata
from pydub import AudioSegment
from pydub import effects

# Costanti globali
### MODIFICARE QUESTI PER TRAIN/TEST ###
IS_TRAINING = False
ITER_SONGS_MIN = 14 if IS_TRAINING else 7  # Numero MINIMO di canzoni da processare
ITER_SONGS_MAX = 28 if IS_TRAINING else 14  # Numero MASSIMO di canzoni da processare

ITER_NOISE_MIN = 5 if IS_TRAINING else 2  # Numero MINIMO di coppie di rumore bianco per canzone
ITER_NOISE_MAX = 7 if IS_TRAINING else 4  # Numero MASSIMO di coppie di rumore bianco per canzone

### --- PATHS --- ###
INPUT_DIR = ".\\dataset_white\\train\\input" if IS_TRAINING else ".\\dataset_white\\test\\input"
TARGET_DIR = ".\\dataset_white\\train\\target" if IS_TRAINING else ".\\dataset_white\\test\\target"
SONGS_DIR = ".\\musdb18\\train" if IS_TRAINING else ".\\musdb18\\test"


### --- FUNZIONI --- ###

def generate_white_noise(sample_rate, channels, length_samples, use_cuda=False):
    """
    Genera rumore bianco con le caratteristiche specificate.

    Args:
        sample_rate (int): Frequenza di campionamento
        channels (int): Numero di canali (1 per mono, 2 per stereo)
        length_samples (int): Lunghezza in campioni
        use_cuda (bool): Se utilizzare CUDA per la generazione

    Returns:
        dict: Dizionario con i dati del rumore bianco
    """
    try:
        if use_cuda and torch.cuda.is_available():
            # Genera rumore bianco usando PyTorch su CUDA
            white_noise = torch.randn(channels, length_samples, device='cuda', dtype=torch.float32)

            return {
                'tensor': white_noise,
                'sample_rate': sample_rate,
                'channels': channels,
                'sample_width': 2,  # 16-bit
                'format': 'generated',
                'cuda': True
            }
        else:
            # Genera rumore bianco usando numpy
            white_noise = np.random.randn(channels, length_samples).astype(np.float32)

            return {
                'array': white_noise,
                'sample_rate': sample_rate,
                'channels': channels,
                'sample_width': 2,  # 16-bit
                'format': 'generated',
                'cuda': False
            }

    except Exception as e:
        print(f"Errore durante la generazione del rumore bianco: {e}")
        traceback.print_exc()
        return None


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
    import re
    from pathlib import Path

    # Determina la directory padre che contiene le cartelle train e test
    log_file = os.path.join(parent_dir, "log.txt")

    # Calcola lo spazio occupato
    input_size = 0
    target_size = 0
    input_files = 0
    target_files = 0
    attempt = 1  # Valore predefinito se non ci sono tentativi precedenti

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
    current_date = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    # Determina il numero del tentativo controllando il file di log
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Cerca tutti i tentativi per la stessa sessione e la stessa data
        pattern = fr"--- DATASET for {session_type} #(\d+)- {current_date}"
        attempts = re.findall(pattern, log_content)

        if attempts:
            # Prendi l'ultimo tentativo e incrementalo di 1
            attempt = max(map(int, attempts)) + 1

    # Crea il titolo
    title = f"--- DATASET for {session_type} #{attempt}- {current_date} ---"

    # Crea una linea di separazione con la stessa lunghezza del titolo
    separator = "-" * len(title)

    # Crea il messaggio di log
    log_message = f"""{title}
Generated pairs: {num_pairs}
Generated files: {total_files}
Output size: {total_size_mb:.2f} MB
{separator}

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
                audio = effects.normalize(audio)

                # Conversione in numpy array
                samples = np.array(audio.get_array_of_samples())

                # Se è mono, reshappiamo l'array per avere la dimensione del canale
                if audio.channels == 1:
                    samples = samples.reshape(1, -1)
                else:
                    # Converte un array stereo in formato (canali, samples)
                    samples = samples.reshape(-1, audio.channels).T

                # Normalizza i campioni in float32
                samples = samples.astype(np.float32) / 32768.0

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


def normalize_db(white_noise_data, clean_data):
    """
    Normalizza i dati del rumore bianco relativamente ai dati puliti con un approccio SNR intelligente.
    Usa un range di valori SNR (6-15 dB) per assicurarsi che il rumore non sovrasti
    la musica ma rimanga udibile.
    Restituisce i white_noise_data modificati.
    """
    try:
        # Assicurati che white_noise_data non sia None
        if white_noise_data is None:
            print(" Errore: white_noise_data è None nella funzione normalize_db")
            return None

        # Assicurati che clean_data non sia None
        if clean_data is None:
            print(" Errore: clean_data è None nella funzione normalize_db")
            return None

        # Se la canzone ha un'alta gamma dinamica, usa SNR più alto per rendere il rumore meno intrusivo
        # nei passaggi silenziosi, pur rimanendo udibile nelle parti più forti
        min_snr = 2
        max_snr = 27

        # Estrai array di dati o tensori basandoti su se sono su CUDA o no
        if white_noise_data.get('cuda', False):
            # Versione CUDA con PyTorch
            white_noise_samples = white_noise_data['tensor']
            clean_samples = clean_data['tensor']

            # Calcola RMS del segnale pulito (più rilevante percettivamente della potenza)
            clean_rms = torch.sqrt(torch.mean(clean_samples.float() ** 2))

            # Calcola RMS del rumore bianco
            white_noise_rms = torch.sqrt(torch.mean(white_noise_samples.float() ** 2))

            # Analizza la gamma dinamica del segnale pulito
            if clean_samples.shape[1] > 0:  # Assicurati che abbiamo campioni
                # Dividi il segnale in chunk e ottieni RMS per chunk
                chunk_size = min(44100, clean_samples.shape[1])  # chunk di 1 secondo a 44.1 kHz
                num_chunks = max(1, clean_samples.shape[1] // chunk_size)
                chunks_rms = []

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, clean_samples.shape[1])
                    chunk = clean_samples[:, start:end]
                    chunk_rms = torch.sqrt(torch.mean(chunk.float() ** 2))
                    chunks_rms.append(chunk_rms.item())

                # Calcola gamma dinamica - usala per regolare SNR intelligentemente
                dynamic_range = max(chunks_rms) / (min(chunks_rms) + 1e-10)

                # Regola range SNR basandoti sulla gamma dinamica
                if dynamic_range > 10:  # Canzone molto dinamica
                    min_snr = 4
                    max_snr = 30
                elif dynamic_range < 3:  # Canzone non molto dinamica
                    min_snr = 0
                    max_snr = 25
            else:
                # Valori predefiniti se non possiamo analizzare i chunk
                min_snr = 2
                max_snr = 27

            # Genera SNR casuale nel nostro range calcolato
            target_snr = min_snr + torch.rand(1).item() * (max_snr - min_snr)

            # Calcola il fattore di scala per il rumore basato sull'SNR desiderato
            # SNR = 20 * log10(clean_rms / white_noise_rms) # Usando RMS per risultati percettivi migliori
            # Quindi: white_noise_rms_new = clean_rms / (10^(SNR/20))
            scaling_factor = clean_rms / (white_noise_rms * (10 ** (target_snr / 20)))

            # Scala il rumore bianco
            white_noise_data['tensor'] = white_noise_samples * scaling_factor

        else:
            # Versione CPU con numpy
            white_noise_samples = white_noise_data['array']
            clean_samples = clean_data['array']

            # Calcola RMS del segnale pulito (più rilevante percettivamente della potenza)
            clean_rms = np.sqrt(np.mean(clean_samples ** 2))

            # Calcola RMS del rumore bianco
            white_noise_rms = np.sqrt(np.mean(white_noise_samples ** 2))

            # Analizza la gamma dinamica del segnale pulito
            if clean_samples.shape[1] > 0:  # Assicurati che abbiamo campioni
                # Dividi il segnale in chunk e ottieni RMS per chunk
                chunk_size = min(44100, clean_samples.shape[1])  # chunk di 1 secondo a 44.1 kHz
                num_chunks = max(1, clean_samples.shape[1] // chunk_size)
                chunks_rms = []

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, clean_samples.shape[1])
                    chunk = clean_samples[:, start:end]
                    chunk_rms = np.sqrt(np.mean(chunk ** 2))
                    chunks_rms.append(chunk_rms)

                # Calcola gamma dinamica - usala per regolare SNR intelligentemente
                dynamic_range = max(chunks_rms) / (min(chunks_rms) + 1e-10)

                # Regola range SNR basandoti sulla gamma dinamica
                if dynamic_range > 10:  # Canzone molto dinamica
                    min_snr = 4
                    max_snr = 30
                elif dynamic_range < 3:  # Canzone non molto dinamica
                    min_snr = 0
                    max_snr = 25
            else:
                # Valori predefiniti se non possiamo analizzare i chunk
                min_snr = 2
                max_snr = 27

            # Genera SNR casuale nel nostro range calcolato
            target_snr = min_snr + np.random.random() * (max_snr - min_snr)

            # Calcola il fattore di scala per il rumore basato sull'SNR desiderato
            # SNR = 20 * log10(clean_rms / white_noise_rms) # Usando RMS per risultati percettivi migliori
            # Quindi: white_noise_rms_new = clean_rms / (10^(SNR/20))
            scaling_factor = clean_rms / (white_noise_rms * (10 ** (target_snr / 20)))

            # Scala il rumore bianco
            white_noise_data['array'] = white_noise_samples * scaling_factor

        print(f" Rumore bianco normalizzato con SNR target: {target_snr:.2f} dB (regolato per gamma dinamica)")
        return white_noise_data

    except Exception as e:
        print(f" Errore durante la normalizzazione del rumore bianco: {e}")
        traceback.print_exc()  # Aggiungi traceback per debugging
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
                samples = np.clip(samples, -1.0, 1.0)
                # Assumiamo che i float siano in range [-1, 1]
                samples = (samples * 32767).astype(np.int16)
            else:
                samples = np.clip(samples, -1.0, 1.0)
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
            print(" Errore: audio_data è None in make_audio_zero_mean")
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
            print(" Errore: audio_data è None in loop_or_truncate")
            return None

        if not audio_data.get('cuda', False) and 'array' not in audio_data:
            print(" Errore: audio_data non contiene la chiave 'array'")
            return None

        if audio_data.get('cuda', False) and 'tensor' not in audio_data:
            print(" Errore: audio_data non contiene la chiave 'tensor'")
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
            print(" Errore: audio_data è None in convert_to_stereo")
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
    # File per tracciare le canzoni usate (i rumori ora sono generati dinamicamente)
    used_songs_file = ".\\dataset\\used_songs.txt"

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
        # Parsing degli argomenti
        parser = argparse.ArgumentParser(
            description='Script per sovrapporre rumore bianco generato a file audio di canzoni.')
        parser.add_argument('--path-songs', type=str, default=SONGS_DIR,
                            help='Percorso alla directory contenente i file audio delle canzoni (MP4).')
        parser.add_argument('--iter-songs', type=int, default=ITER_SONGS_MAX,
                            help=f'Numero di canzoni da processare (default: {ITER_SONGS_MAX}).')
        parser.add_argument('--iter-white-noise', type=int, default=ITER_NOISE_MAX,
                            help=f'Numero di coppie di rumore bianco per canzone (default: {ITER_NOISE_MAX}).')
        parser.add_argument('--use-cuda', action='store_true', help='Utilizza CUDA/GPU se disponibile.')
        parser.add_argument('--input-dir', type=str, default=INPUT_DIR, help='Directory di output per i file INPUT.')
        parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help='Directory di output per i file TARGET.')

        args = parser.parse_args()

        # Correggi i percorsi per il sistema operativo corrente
        args.path_songs = fix_path_separators(args.path_songs)
        args.input_dir = fix_path_separators(args.input_dir)
        args.target_dir = fix_path_separators(args.target_dir)

        # Verifica dell'esistenza delle directory
        if not os.path.isdir(args.path_songs):
            print(f"Errore: La directory delle canzoni '{args.path_songs}' non esiste.")
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
        print("Ricerca dei file di canzoni...")

        # Lista dei file delle canzoni
        canzoni_files = [os.path.join(args.path_songs, f) for f in os.listdir(args.path_songs)
                         if is_audio_file(os.path.join(args.path_songs, f))]

        if not canzoni_files:
            print(f"Errore: Nessun file audio trovato nella directory delle canzoni '{args.path_songs}'.")
            sys.exit(1)

        # Counter delle coppie generate
        generated_pairs = 0

        # Ciclo principale di elaborazione
        songs_processed = 1
        backup_file(used_songs_file)

        # Carica i file audio già utilizzati, se esistono
        used_songs = []
        if os.path.exists(used_songs_file):
            with open(used_songs_file, 'r') as f:
                used_songs = [line.strip() for line in f.readlines()]

        # Ciclo esterno (Canzoni)
        rand_iter_songs = np.random.randint(ITER_SONGS_MIN,
                                            ITER_SONGS_MAX + 1) if 1 <= ITER_SONGS_MIN < ITER_SONGS_MAX else 1
        rand_iter_white_noises = np.zeros(rand_iter_songs)

        for i in range(rand_iter_songs):
            rand_iter_white_noises[i] = np.random.randint(ITER_NOISE_MIN,
                                                          ITER_NOISE_MAX + 1) if 1 <= ITER_NOISE_MIN < ITER_NOISE_MAX else 1

        print(f"\nAvvio elaborazione di {rand_iter_songs} canzoni...")

        while songs_processed <= rand_iter_songs:
            # Seleziona una canzone casuale non utilizzata in precedenza
            available_songs = [song for song in canzoni_files if os.path.basename(song) not in used_songs]

            # Se tutte le canzoni sono state utilizzate o non ci sono canzoni disponibili, usa una casuale
            if not available_songs and canzoni_files:
                song_path = random.choice(canzoni_files)
            elif available_songs:
                song_path = random.choice(available_songs)
            else:
                print("Errore: Nessuna canzone disponibile per l'elaborazione.")
                sys.exit(1)

            song_basename = os.path.basename(song_path)

            # Prima leggi il contenuto esistente
            with open(used_songs_file, 'r') as f:
                existing_songs = f.readlines()

            songs_count = len(existing_songs)
            if songs_count >= 75:
                # Rimuovi la prima riga e mantieni le altre
                existing_songs = existing_songs[1:]

                # Riscrivi il file con le righe rimanenti + la nuova riga
                with open(used_songs_file, 'w') as f:
                    f.writelines(existing_songs)
                    f.write(f"{song_basename}\n")

                print(f"Rimossa la prima riga dal file. Mantenute {len(existing_songs) + 1} righe totali.")
            else:
                # Se siamo sotto le 75 righe, aggiungi normalmente
                with open(used_songs_file, 'a') as f:
                    f.write(f"{song_basename}\n")

            # Aggiorna la lista in memoria con i file appena caricati
            used_songs.append(song_basename)

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

            # Ciclo interno (Rumore Bianco)
            white_noise_pairs_processed = 1

            rand_iter_white_noise = int(rand_iter_white_noises[songs_processed - 1])
            print(f"Generazione di {rand_iter_white_noise} coppie di rumore bianco per questa canzone...")

            generated_pairs += rand_iter_white_noise

            while white_noise_pairs_processed <= rand_iter_white_noise:
                # Inizializza variabili per rumori bianchi
                white_noise1_data = None
                white_noise2_data = None

                print(f"\n Coppia di rumore bianco {white_noise_pairs_processed}/{rand_iter_white_noise} "
                      f"(canzone {songs_processed}/{rand_iter_songs}):")

                # Genera il primo rumore bianco
                print(f" - Rumore Bianco 1: Generazione in corso...")
                white_noise1_data = generate_white_noise(
                    song_data['sample_rate'],
                    song_data['channels'],
                    song_length,
                    use_cuda
                )

                if white_noise1_data is None:
                    print(" Errore: Impossibile generare il rumore bianco 1. Passaggio alla coppia successiva.")
                    continue

                # Processa il rumore bianco come i rumori originali
                white_noise1_data = make_audio_zero_mean(white_noise1_data)
                white_noise1_data = normalize_db(white_noise1_data, song_data)

                # Per la modalità training, genera anche il secondo rumore bianco
                if IS_TRAINING:
                    print(f" - Rumore Bianco 2: Generazione in corso...")
                    white_noise2_data = generate_white_noise(
                        song_data['sample_rate'],
                        song_data['channels'],
                        song_length,
                        use_cuda
                    )

                    if white_noise2_data is None:
                        print(" Errore: Impossibile generare il rumore bianco 2. Passaggio alla coppia successiva.")
                        continue

                    white_noise2_data = make_audio_zero_mean(white_noise2_data)
                    white_noise2_data = normalize_db(white_noise2_data, song_data)

                # Sovrapponi i rumori bianchi alla canzone
                if use_cuda:
                    # Versione CUDA
                    input_samples = song_data['tensor'] + white_noise1_data['tensor']

                    if IS_TRAINING:
                        target_samples = song_data['tensor'] + white_noise2_data['tensor']
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
                    input_samples = song_data['array'] + white_noise1_data['array']

                    if IS_TRAINING:
                        target_samples = song_data['array'] + white_noise2_data['array']
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
                song_base_name = slugify(song_name)

                if IS_TRAINING:
                    input_filename = f"INPUT-S{songs_processed}N{white_noise_pairs_processed}-(WHITE_NOISE_1)-({song_base_name}).wav"
                    target_filename = f"TARGET-S{songs_processed}N{white_noise_pairs_processed}-(WHITE_NOISE_2)-({song_base_name}).wav"
                else:
                    input_filename = f"INPUT-S{songs_processed}N{white_noise_pairs_processed}-(WHITE_NOISE)-({song_base_name}).wav"
                    target_filename = f"TARGET-S{songs_processed}N{white_noise_pairs_processed}-(CLEAN)-({song_base_name}).wav"

                input_path = os.path.join(input_dir, input_filename)
                target_path = os.path.join(target_dir, target_filename)

                # Salva i file di output
                print(f" Salvataggio dei file di output...")
                if save_audio(input_audio, input_path) and save_audio(target_audio, target_path):
                    print(f" File salvati con successo:\n - {input_filename}\n - {target_filename}")
                    white_noise_pairs_processed += 1
                else:
                    print(" Errore durante il salvataggio dei file. Passaggio alla coppia successiva.")

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

    except (Exception, KeyboardInterrupt, InterruptedError):
        print("\nERRORE o interruzione! Ripristino dei file di uso...")
        restore_file(used_songs_file)
        sys.exit(1)

    finally:
        # Se il backup esiste e non c'è stato errore, lo elimina
        remove_backup(used_songs_file)


if __name__ == "__main__":
    main()
