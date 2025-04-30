#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Overlay Script (CUDA Optimized)
-------------------
Sovrappone file audio di rumore a file audio di canzoni
con opzioni specifiche di filtraggio, elaborazione e output.
Utilizza CUDA/GPU quando disponibile per massimizzare le prestazioni.
Supporta vari formati audio attraverso pydub e torchaudio.
"""

import os
import argparse
import sys
import torch
import torchaudio
import numpy as np
import warnings
import time
import tempfile
import shutil
from pathlib import Path
from pydub import AudioSegment
import subprocess

# Per ignorare i warning non critici
warnings.filterwarnings("ignore")

def clean_output_directory(output_dir):
    """Cancella tutti i file nella directory di output se esiste."""
    print(f"\nPulizia directory di output: {output_dir}")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"  - Rimosso: {file}")
            except Exception as e:
                print(f"  ! Errore rimuovendo {file}: {e}")
        print(f"✓ Directory pulita: {len(os.listdir(output_dir))} file rimasti")
    else:
        os.makedirs(output_dir)
        print(f"✓ Creata nuova directory: {output_dir}")

def debug_file_existence(file_path):
    """Verifica e stampa informazioni dettagliate sul file."""
    print(f"\nDEBUG - Verifico esistenza file: {file_path}")
    
    if os.path.exists(file_path):
        print(f"✓ Il file esiste")
        print(f"  - Dimensione: {os.path.getsize(file_path)} bytes")
        print(f"  - Ultima modifica: {time.ctime(os.path.getmtime(file_path))}")
        
        # Controlla se il file è leggibile
        try:
            with open(file_path, 'rb') as f:
                f.read(1)
            print("  - Il file è leggibile")
        except Exception as e:
            print(f"  - ERRORE: Il file non è leggibile: {str(e)}")
    else:
        # Se il file non esiste, controlliamo la directory
        dir_path = os.path.dirname(file_path)
        print(f"✗ Il file NON esiste")
        
        if os.path.exists(dir_path):
            print(f"  - La directory {dir_path} esiste")
            print("  - Files nella directory:")
            for f in os.listdir(dir_path):
                print(f"      {f}")
        else:
            print(f"  - La directory {dir_path} NON esiste")
            parent_dir = os.path.dirname(dir_path)
            if os.path.exists(parent_dir):
                print(f"  - La directory padre {parent_dir} esiste")
                print("  - Directories presenti:")
                for d in os.listdir(parent_dir):
                    if os.path.isdir(os.path.join(parent_dir, d)):
                        print(f"      {d}/")

def verify_ffmpeg():
    """Verifica che ffmpeg sia installato e funzionante."""
    print("\nVerifica di FFmpeg:")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ FFmpeg installato e funzionante")
            print(f"  Versione: {result.stdout.splitlines()[0]}")
            return True
        else:
            print(f"✗ FFmpeg non funzionante (codice errore {result.returncode})")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg non trovato nel PATH")
        return False
    except Exception as e:
        print(f"✗ Errore verificando FFmpeg: {str(e)}")
        return False

def db_to_amplitude(db):
    """Converte un valore in dB a un fattore di amplificazione."""
    return 10 ** (db / 20)

def is_audio_file(filename):
    """Verifica se un file è un formato audio supportato."""
    audio_extensions = ['.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg', '.mp4']
    return any(filename.lower().endswith(ext) for ext in audio_extensions)

@torch.no_grad()  # Ottimizzazione: disabilita il calcolo dei gradienti
def load_audio(file_path, device):
    """
    Carica un file audio utilizzando torchaudio o pydub con supporto CUDA.
    """
    if not os.path.exists(file_path):
        debug_file_existence(file_path)
        raise FileNotFoundError(f"Il file {file_path} non esiste")
    
    print(f"Caricamento di {os.path.basename(file_path)}")
    
    try:
        # Primo tentativo: torchaudio (più efficiente con GPU)
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            # Sposta immediatamente su GPU se disponibile
            waveform = waveform.to(device)
            return waveform, sample_rate
        except Exception as e:
            print(f"✗ torchaudio fallito: {str(e)}")
        
        # Secondo tentativo: pydub con ffmpeg
        audio = AudioSegment.from_file(file_path)
        sample_rate = audio.frame_rate
        channels = audio.channels
        
        # Converti a numpy array e poi a tensor PyTorch
        samples = np.array(audio.get_array_of_samples())
        
        # Reshape per canali corretti
        if channels == 2:
            samples = samples.reshape((-1, 2)).T
        else:
            samples = samples.reshape((1, -1))
        
        # Normalizza in range [-1.0, 1.0]
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            samples = samples.astype(np.float32) / 2147483648.0
        
        # Converti a tensor PyTorch e sposta su device
        waveform = torch.from_numpy(samples.astype(np.float32)).to(device)
        
        return waveform, sample_rate
    
    except Exception as e:
        raise Exception(f"Impossibile caricare {file_path}: {str(e)}")

@torch.no_grad()  # Ottimizzazione: disabilita il calcolo dei gradienti
def save_audio(waveform, sample_rate, file_path, format_settings):
    """
    Salva un tensor PyTorch come file audio AAC ad alta qualità.
    """
    print(f"Salvataggio in {file_path}...")
    
    try:
        # Ottimizzazione: sposta sempre su CPU prima del salvataggio
        waveform_cpu = waveform.cpu()
        
        # Per AAC ad alta qualità (impostazioni specificate)
        if file_path.lower().endswith('.m4a'):
            # Usa torchaudio se possibile per formato AAC
            try:
                torchaudio.save(
                    file_path,
                    waveform_cpu,
                    sample_rate,
                    format="mp4",
                    compression=-2,  # Alta qualità
                    bits_per_sample=16
                )
                return
            except Exception:
                pass
            
            # Fallback a pydub per AAC
            # Normalizza e converti a int16
            waveform_np = waveform_cpu.numpy()
            max_val = np.max(np.abs(waveform_np))
            if max_val > 0:
                waveform_np = waveform_np / max_val
            waveform_np = (waveform_np * 32767).astype(np.int16)
            
            # Formatta per AudioSegment
            if waveform_np.shape[0] > 1:  # stereo
                waveform_np = waveform_np.T
            else:  # mono
                waveform_np = waveform_np.T.reshape(-1)
            
            # Crea AudioSegment
            if waveform_np.ndim == 1 or waveform_np.shape[1] == 1:  # mono
                audio = AudioSegment(
                    waveform_np.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # 16 bit
                    channels=1
                )
            else:  # stereo
                audio = AudioSegment(
                    waveform_np.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,  # 16 bit
                    channels=2
                )
            
            # Esporta con massima qualità
            audio.export(
                file_path, 
                format="ipod",  # pydub usa "ipod" per AAC
                bitrate="320k",  # Massimo bitrate
                parameters=["-q:a", "0"]  # Massima qualità
            )
    except Exception as e:
        raise Exception(f"Impossibile salvare {file_path}: {str(e)}")

@torch.no_grad()  # Ottimizzazione: disabilita il calcolo dei gradienti
def normalize_audio(waveform):
    """Normalizza l'audio per avere picco massimo a 0 dB."""
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform

@torch.no_grad()  # Ottimizzazione: disabilita il calcolo dei gradienti
def adjust_volume(waveform, db_change):
    """Applica un aggiustamento di volume in dB al waveform."""
    if db_change == 0:
        return waveform
    return waveform * db_to_amplitude(db_change)

@torch.no_grad()  # Ottimizzazione: disabilita il calcolo dei gradienti
def loop_or_truncate(noise_waveform, noise_sample_rate, song_length, song_sample_rate):
    """
    Adatta il rumore alla lunghezza della canzone, loopandolo se necessario
    o troncandolo se è più lungo.
    """
    target_length = int(song_length * noise_sample_rate / song_sample_rate)
    noise_length = noise_waveform.shape[1]
    
    if noise_length >= target_length:
        # Tronca il rumore se è più lungo (operazione efficiente)
        return noise_waveform[:, :target_length]
    else:
        # Loop il rumore se è più corto
        device = noise_waveform.device
        repeats = target_length // noise_length
        remainder = target_length % noise_length
        
        # Ottimizzazione: allocazione unica in memoria
        looped_noise = torch.zeros((noise_waveform.shape[0], target_length), 
                                  device=device, dtype=noise_waveform.dtype)
        
        # Ottimizzazione: operazione batch invece di cicli
        for i in range(repeats):
            start_pos = i * noise_length
            looped_noise[:, start_pos:start_pos + noise_length] = noise_waveform
        
        if remainder > 0:
            start_pos = repeats * noise_length
            looped_noise[:, start_pos:] = noise_waveform[:, :remainder]
            
        return looped_noise

@torch.no_grad()  # Ottimizzazione: disabilita il calcolo dei gradienti
def resample_if_needed(waveform, orig_sample_rate, target_sample_rate):
    """Ricampiona il waveform se necessario."""
    if orig_sample_rate != target_sample_rate:
        # Sposta il resampler sul device del waveform per efficienza
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=target_sample_rate
        ).to(waveform.device)
        return resampler(waveform)
    return waveform

@torch.no_grad()  # Ottimizzazione: disabilita il calcolo dei gradienti
def convert_to_stereo(waveform):
    """Converte l'audio a stereo se è mono."""
    if waveform.shape[0] == 1:  # Se è mono
        # Ottimizzazione: usa clone anziché cat quando possibile
        return torch.cat([waveform, waveform], dim=0)
    return waveform[:2]  # Se ha più di 2 canali, prendi solo i primi 2

def main():
    # Configurazione argparse
    parser = argparse.ArgumentParser(description="Sovrapponi file audio di rumore a file audio di canzoni")
    parser.add_argument("path_canzoni", help="Percorso del file o directory delle canzoni")
    parser.add_argument("path_rumori", help="Percorso del file o directory dei rumori")
    parser.add_argument("--formato_output", choices=["aac"], default="aac",
                      help="Formato del file di output (solo AAC supportato)")
    parser.add_argument("--volume_rumore_db", type=float, default=0.0,
                      help="Aggiustamento del volume dei rumori in dB (default: 0.0)")
    parser.add_argument("--debug", action="store_true", help="Attiva modalità debug")
    
    args = parser.parse_args()
    
    # Ottimizzazione PyTorch
    torch.backends.cudnn.benchmark = True  # Aumenta performance per operazioni ripetitive
    
    # Output debugging iniziale
    print("\n=========== INFO SISTEMA ===========")
    print(f"Directory corrente: {os.getcwd()}")
    print(f"PyTorch: {torch.__version__}")
    
    # Determina se usare CUDA o CPU e configura per massime prestazioni
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        # Ottimizzazione: aumenta memoria cache
        torch.cuda.empty_cache()
    print("=====================================\n")
    
    # Verifica che FFmpeg sia disponibile
    if not verify_ffmpeg():
        print("ERRORE: FFmpeg non disponibile. L'applicazione non può continuare.")
        sys.exit(1)
    
    # Directory di output
    output_dir = os.path.join(os.getcwd(), "db_merged")
    
    # Pulisci o crea la directory di output
    clean_output_directory(output_dir)
    
    # Processa il percorso delle canzoni
    print(f"\n=========== FILE CANZONI ===========")
    print(f"Percorso canzoni: {args.path_canzoni}")
    
    song_files = []
    if os.path.isdir(args.path_canzoni):
        for filename in os.listdir(args.path_canzoni):
            filepath = os.path.join(args.path_canzoni, filename)
            if os.path.isfile(filepath) and is_audio_file(filepath):
                basename = os.path.splitext(os.path.basename(filepath))[0]
                if "_mixture" in basename:
                    song_files.append(filepath)
    elif os.path.isfile(args.path_canzoni) and is_audio_file(args.path_canzoni):
        basename = os.path.splitext(os.path.basename(args.path_canzoni))[0]
        if "_mixture" in basename:
            song_files.append(args.path_canzoni)
    
    print(f"Trovati {len(song_files)} file canzone validi")
    
    # Processa il percorso dei rumori
    print(f"\n=========== FILE RUMORI ===========")
    print(f"Percorso rumori: {args.path_rumori}")
    
    noise_files = []
    if os.path.isdir(args.path_rumori):
        for filename in os.listdir(args.path_rumori):
            filepath = os.path.join(args.path_rumori, filename)
            if os.path.isfile(filepath) and is_audio_file(filepath):
                basename = os.path.splitext(os.path.basename(filepath))[0]
                if "_zeromean" in basename:
                    noise_files.append(filepath)
    elif os.path.isfile(args.path_rumori) and is_audio_file(args.path_rumori):
        basename = os.path.splitext(os.path.basename(args.path_rumori))[0]
        if "_zeromean" in basename:
            noise_files.append(args.path_rumori)
    
    print(f"Trovati {len(noise_files)} file rumore validi")
    print("=====================================\n")
    
    if not song_files:
        print("Nessun file canzone valido trovato. Uscita.")
        return
    
    if not noise_files:
        print("Nessun file rumore valido trovato. Uscita.")
        return
    
    # Utilizziamo sempre AAC come formato di output
    format_settings = {
        "format": "mp4", 
        "codec": "aac", 
        "bits_per_sample": 16, 
        "compression": -2
    }
    
    # Processa ogni canzone con ogni rumore
    processed_count = 0
    error_count = 0
    
    # Precarica i rumori per efficienza se possibile
    noise_cache = {}
    if len(noise_files) < 5:  # Cache solo se abbiamo pochi file di rumore
        print("Precaricamento rumori per ottimizzazione...")
        for noise_path in noise_files:
            try:
                noise_waveform, noise_sample_rate = load_audio(noise_path, device)
                noise_basename = os.path.splitext(os.path.basename(noise_path))[0]
                noise_clean_name = noise_basename.replace("_zeromean", "")
                noise_cache[noise_path] = (noise_waveform, noise_sample_rate, noise_clean_name)
                print(f"  ✓ Caricato {noise_clean_name}")
            except Exception as e:
                print(f"  ✗ Impossibile precaricare {os.path.basename(noise_path)}: {e}")
    
    # Utilizzo batch per sfruttare parallelismo GPU
    batch_size = 1  # Aumentalo se hai abbastanza VRAM
    
    for song_idx in range(0, len(song_files), batch_size):
        song_batch = song_files[song_idx:song_idx + batch_size]
        
        for song_path in song_batch:
            song_basename = os.path.splitext(os.path.basename(song_path))[0]
            song_clean_name = song_basename.replace(".stem_mixture", "")
            
            print(f"\n=========== PROCESSING {song_clean_name} ===========")
            
            try:
                # Carica la canzone con supporto GPU
                song_waveform, song_sample_rate = load_audio(song_path, device)
                
                # Converti a stereo per AAC
                if song_waveform.shape[0] == 1:
                    song_waveform = convert_to_stereo(song_waveform)
                elif song_waveform.shape[0] > 2:
                    song_waveform = song_waveform[:2]  # Limita a stereo
                
                song_length = song_waveform.shape[1]
                
                for noise_path in noise_files:
                    start_time = time.time()
                    
                    # Usa cache se disponibile
                    if noise_path in noise_cache:
                        noise_waveform, noise_sample_rate, noise_clean_name = noise_cache[noise_path]
                        print(f"\nElaborazione di '{song_clean_name}' con rumore '{noise_clean_name}' (cached)...")
                    else:
                        noise_basename = os.path.splitext(os.path.basename(noise_path))[0]
                        noise_clean_name = noise_basename.replace("_zeromean", "")
                        print(f"\nElaborazione di '{song_clean_name}' con rumore '{noise_clean_name}'...")
                        
                        # Carica il rumore con supporto GPU
                        noise_waveform, noise_sample_rate = load_audio(noise_path, device)
                    
                    try:
                        # Clone per evitare modifiche ai dati in cache
                        if noise_path in noise_cache:
                            noise_waveform = noise_waveform.clone()
                        
                        # Normalizza il rumore
                        noise_waveform = normalize_audio(noise_waveform)
                        
                        # Applica volume
                        if args.volume_rumore_db != 0:
                            noise_waveform = adjust_volume(noise_waveform, args.volume_rumore_db)
                        
                        # Ricampiona
                        if noise_sample_rate != song_sample_rate:
                            noise_waveform = resample_if_needed(noise_waveform, noise_sample_rate, song_sample_rate)
                        
                        # Adatta durata
                        noise_waveform = loop_or_truncate(noise_waveform, song_sample_rate, song_length, song_sample_rate)
                        
                        # Garantisci compatibilità canali
                        if noise_waveform.shape[0] == 1 and song_waveform.shape[0] == 2:
                            noise_waveform = convert_to_stereo(noise_waveform)
                        elif noise_waveform.shape[0] == 2 and song_waveform.shape[0] == 1:
                            # Converti entrambi a stereo per AAC
                            song_waveform = convert_to_stereo(song_waveform)
                        
                        # Sovrapponi audio (in parallelo su GPU se disponibile)
                        merged_waveform = song_waveform + noise_waveform
                        
                        # Normalizza
                        merged_waveform = normalize_audio(merged_waveform)
                        
                        # Nome file output
                        output_filename = f"[{noise_clean_name}] {song_clean_name}.m4a"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Salva AAC ad alta qualità
                        save_audio(
                            merged_waveform,
                            song_sample_rate,
                            output_path,
                            format_settings
                        )
                        
                        processing_time = time.time() - start_time
                        print(f"✓ File salvato: {output_filename}")
                        print(f"  Tempo: {processing_time:.2f}s")
                        processed_count += 1
                        
                        # Libera memoria GPU se necessario
                        if device.type == "cuda" and not noise_path in noise_cache:
                            del noise_waveform
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"✗ Errore con rumore '{noise_path}': {str(e)}")
                        error_count += 1
                        continue
                
                # Libera memoria GPU
                if device.type == "cuda":
                    del song_waveform
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"✗ Errore con canzone '{song_path}': {str(e)}")
                error_count += 1
                continue
    
    print("\n=========== RIEPILOGO ===========")
    print(f"File elaborati con successo: {processed_count}")
    print(f"Errori incontrati: {error_count}")
    print(f"Directory di output: {output_dir}")
    print("=================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERRORE CRITICO: {str(e)}")
        print("\nTraceback completo:")
        import traceback
        traceback.print_exc()
        sys.exit(1)