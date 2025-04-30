#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Overlay Script (Improved Version)
-------------------
Sovrappone file audio di rumore a file audio di canzoni
con opzioni specifiche di filtraggio, elaborazione e output.
Utilizza CUDA/GPU quando disponibile, altrimenti usa la CPU.
Supporta vari formati audio attraverso pydub e ffmpeg.
"""

import os
import argparse
import glob
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
import warnings
import time
import tempfile
from pydub import AudioSegment
import soundfile as sf
import librosa
import io
import subprocess

# Per ignorare i warning non critici
warnings.filterwarnings("ignore")

# Specifica il percorso a ffmpeg se non è nel PATH di sistema
# Decommentare e modificare la riga seguente con il percorso corretto a ffmpeg.exe
# ffmpeg_path = r'C:\percorso\a\ffmpeg\bin'  # Usa il percorso corretto!
# os.environ["PATH"] += os.pathsep + ffmpeg_path

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
    print("")  # Riga vuota per leggibilità

def verify_ffmpeg():
    """Verifica che ffmpeg sia installato e funzionante in modo dettagliato."""

    print("\nVerifica dettagliata di FFmpeg:")

    # Controlla nelle variabili d'ambiente
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    ffmpeg_in_path = False
    for directory in path_dirs:
        ffmpeg_exe = os.path.join(directory, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if os.path.exists(ffmpeg_exe) and os.access(ffmpeg_exe, os.X_OK):
            print(f"✓ FFmpeg trovato in: {directory}")
            ffmpeg_in_path = True
            break

    if not ffmpeg_in_path:
        print("✗ FFmpeg non trovato nel PATH di sistema")

    # Prova a eseguire ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ FFmpeg eseguibile e funzionante")
            print(f"  Versione: {result.stdout.splitlines()[0]}")
        else:
            print(f"✗ FFmpeg non funzionante (codice di errore {result.returncode})")
            print(f"  Errore: {result.stderr}")
    except FileNotFoundError:
        print("✗ FFmpeg non trovato o non eseguibile")
    except Exception as e:
        print(f"✗ Errore nel test di FFmpeg: {str(e)}")

    # Test con pydub
    print("\nTest di FFmpeg tramite pydub:")
    try:
        # Crea un breve segmento audio in memoria
        seg = AudioSegment.silent(duration=100)  # 100ms di silenzio
        print("✓ Pydub ha creato correttamente un segmento audio")

        # Prova a convertirlo in MP3
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            temp_path = tmp.name

        try:
            seg.export(temp_path, format="mp3")
            print("✓ Conversione a MP3 riuscita - FFmpeg funziona correttamente con pydub")
        except Exception as e:
            print(f"✗ Errore nella conversione a MP3: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"✗ Errore nel test di pydub: {str(e)}")

def db_to_amplitude(db):
    """Converte un valore in dB a un fattore di amplificazione."""
    return 10 ** (db / 20)

def is_audio_file(filename):
    """Verifica se un file è un formato audio supportato."""
    audio_extensions = ['.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg', '.mp4']
    return any(filename.lower().endswith(ext) for ext in audio_extensions)

def load_audio(file_path):
    """
    Carica un file audio usando pydub o torchaudio in base al formato.
    Ritorna un tensor PyTorch, la frequenza di campionamento e il numero di canali.
    """
    # Verifica l'esistenza del file prima di procedere
    if not os.path.exists(file_path):
        debug_file_existence(file_path)
        raise FileNotFoundError(f"Il file {file_path} non esiste")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    print(f"Caricamento di {os.path.basename(file_path)} - Dimensione: {os.path.getsize(file_path)} bytes")
    
    try:
        # Primo tentativo: torchaudio (più veloce ed efficiente se supporta il formato)
        try:
            print(f"Tentativo con torchaudio...")
            waveform, sample_rate = torchaudio.load(file_path)
            print(f"✓ torchaudio ha caricato il file: {waveform.shape} @ {sample_rate}Hz")
            return waveform, sample_rate
        except Exception as e:
            print(f"✗ torchaudio non può caricare {file_path}: {str(e)}")
        
        # Secondo tentativo: pydub con ffmpeg
        print(f"Tentativo con pydub...")
        audio = AudioSegment.from_file(file_path)
        sample_rate = audio.frame_rate
        channels = audio.channels
        
        print(f"✓ pydub ha caricato il file: {len(audio)}ms, {channels} canali, {sample_rate}Hz")
        
        # Converti a numpy array e poi a tensor PyTorch
        samples = np.array(audio.get_array_of_samples())
        
        # Reshape per ottenere i canali corretti (mono o stereo)
        if channels == 2:
            samples = samples.reshape((-1, 2)).T
        else:
            samples = samples.reshape((1, -1))
        
        # Normalizza in range [-1.0, 1.0]
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        elif samples.dtype == np.int32:
            samples = samples.astype(np.float32) / 2147483648.0
        
        # Converti a tensor PyTorch
        waveform = torch.from_numpy(samples.astype(np.float32))
        
        print(f"✓ Convertito in tensor: {waveform.shape}")
        
        return waveform, sample_rate
    
    except Exception as e:
        raise Exception(f"Impossibile caricare il file audio {file_path}: {str(e)}")

def save_audio(waveform, sample_rate, file_path, format_settings):
    """
    Salva un tensor PyTorch come file audio usando torchaudio o pydub.
    """
    print(f"Salvataggio in {file_path}...")
    
    try:
        # Prima prova con torchaudio
        try:
            print("Tentativo di salvataggio con torchaudio...")
            torchaudio.save(
                file_path,
                waveform.cpu(),
                sample_rate,
                **format_settings
            )
            print("✓ Salvataggio con torchaudio riuscito")
            return
        except Exception as e:
            print(f"✗ torchaudio non può salvare in {file_path}: {str(e)}")
        
        # Se torchaudio fallisce, usa pydub
        print("Tentativo di salvataggio con pydub...")
        
        # Converti il tensor PyTorch in numpy array
        waveform_np = waveform.cpu().numpy()
        
        # Normalizza a [-1, 1] se non lo è già
        max_val = np.max(np.abs(waveform_np))
        if max_val > 0:
            waveform_np = waveform_np / max_val
        
        # Converti da float32 a int16 o int32 in base al formato
        bit_depth = format_settings.get("bits_per_sample", 16)
        
        if bit_depth <= 16:
            waveform_np = (waveform_np * 32767).astype(np.int16)
        else:
            waveform_np = (waveform_np * 2147483647).astype(np.int32)
        
        # Trasforma l'array da [channels, samples] a [samples, channels]
        if waveform_np.shape[0] > 1:  # stereo
            waveform_np = waveform_np.T
        else:  # mono
            waveform_np = waveform_np.T.reshape(-1)
        
        # Crea un oggetto AudioSegment
        if waveform_np.ndim == 1 or waveform_np.shape[1] == 1:  # mono
            audio = AudioSegment(
                waveform_np.tobytes(),
                frame_rate=sample_rate,
                sample_width=bit_depth // 8,
                channels=1
            )
        else:  # stereo
            audio = AudioSegment(
                waveform_np.tobytes(),
                frame_rate=sample_rate,
                sample_width=bit_depth // 8,
                channels=2
            )
        
        # Determina il formato di export
        export_format = os.path.splitext(file_path)[1][1:]
        if export_format == "m4a":
            export_format = "ipod"  # pydub usa "ipod" per AAC
        
        # Imposta il bitrate
        bitrate = None
        if export_format == "ipod" or export_format == "mp3":
            bitrate = "256k"
        
        # Esporta il file
        audio.export(
            file_path, 
            format=export_format,
            bitrate=bitrate,
            parameters=["-q:a", "0"] if export_format == "mp3" else []
        )
        print(f"✓ Salvataggio con pydub riuscito ({export_format})")
    
    except Exception as e:
        raise Exception(f"Impossibile salvare il file audio {file_path}: {str(e)}")

def normalize_audio(waveform):
    """Normalizza l'audio per avere picco massimo a 0 dB."""
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform

def adjust_volume(waveform, db_change):
    """Applica un aggiustamento di volume in dB al waveform."""
    if db_change == 0:
        return waveform
    return waveform * db_to_amplitude(db_change)

def loop_or_truncate(noise_waveform, noise_sample_rate, song_length, song_sample_rate):
    """
    Adatta il rumore alla lunghezza della canzone, loopandolo se necessario
    o troncandolo se è più lungo.
    """
    # Converti la lunghezza della canzone alla frequenza di campionamento del rumore
    target_length = int(song_length * noise_sample_rate / song_sample_rate)
    
    noise_length = noise_waveform.shape[1]
    
    if noise_length >= target_length:
        # Tronca il rumore se è più lungo
        return noise_waveform[:, :target_length]
    else:
        # Loop il rumore se è più corto
        repeats = target_length // noise_length
        remainder = target_length % noise_length
        
        # Creiamo un nuovo tensor con la dimensione corretta
        device = noise_waveform.device
        looped_noise = torch.zeros((noise_waveform.shape[0], target_length), device=device)
        
        # Riempiamo con ripetizioni complete
        for i in range(repeats):
            start_pos = i * noise_length
            looped_noise[:, start_pos:start_pos + noise_length] = noise_waveform
        
        # Aggiungiamo la parte rimanente se necessario
        if remainder > 0:
            start_pos = repeats * noise_length
            looped_noise[:, start_pos:] = noise_waveform[:, :remainder]
            
        return looped_noise

def resample_if_needed(waveform, orig_sample_rate, target_sample_rate):
    """Ricampiona il waveform se necessario."""
    if orig_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=target_sample_rate
        ).to(waveform.device)
        return resampler(waveform)
    return waveform

def convert_to_stereo(waveform):
    """Converte l'audio a stereo se è mono."""
    if waveform.shape[0] == 1:  # Se è mono
        return torch.cat([waveform, waveform], dim=0)
    return waveform[:2]  # Se ha più di 2 canali, prendi solo i primi 2

def main():
    # Configurazione argparse
    parser = argparse.ArgumentParser(description="Sovrapponi file audio di rumore a file audio di canzoni")
    parser.add_argument("path_canzoni", help="Percorso del file o directory delle canzoni")
    parser.add_argument("path_rumori", help="Percorso del file o directory dei rumori")
    parser.add_argument("--formato_output", choices=["aac", "flac", "wav", "mp3"], default="aac",
                      help="Formato del file di output (default: aac)")
    parser.add_argument("--volume_rumore_db", type=float, default=0.0,
                      help="Aggiustamento del volume dei rumori in dB (default: 0.0)")
    parser.add_argument("--debug", action="store_true", help="Attiva modalità debug con informazioni aggiuntive")
    
    args = parser.parse_args()
    
    # Output debugging iniziale
    print("\n=========== INFO SISTEMA ===========")
    print(f"Directory corrente: {os.getcwd()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchaudio: {torchaudio.__version__}")
    print(f"NumPy: {np.__version__}")
    
    # Determina se usare CUDA o CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print("=====================================\n")
    
    # Verifica che FFmpeg sia disponibile in modo più dettagliato
    verify_ffmpeg()
    
    # Crea la directory di output se non esiste
    output_dir = os.path.join(os.getcwd(), "db_merged")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creata directory di output: {output_dir}")
    else:
        print(f"Directory di output esistente: {output_dir}")
    
    # Processa il percorso delle canzoni con maggiori dettagli
    print(f"\n=========== FILE CANZONI ===========")
    print(f"Percorso canzoni: {args.path_canzoni}")
    print(f"Percorso assoluto: {os.path.abspath(args.path_canzoni)}")
    
    # Debug del percorso canzoni
    if os.path.exists(args.path_canzoni):
        if os.path.isdir(args.path_canzoni):
            print(f"La directory esiste e contiene {len(os.listdir(args.path_canzoni))} elementi")
            if args.debug:
                print("Contenuto directory:")
                for item in os.listdir(args.path_canzoni):
                    print(f"  - {item}")
        else:
            print(f"Il percorso esiste ed è un file")
    else:
        print(f"ATTENZIONE: Il percorso non esiste!")
    
    song_files = []
    if os.path.isdir(args.path_canzoni):
        print(f"Cercando file di canzoni che terminano con '_mixture'...")
        for filename in os.listdir(args.path_canzoni):
            filepath = os.path.join(args.path_canzoni, filename)
            if os.path.isfile(filepath) and is_audio_file(filepath):
                basename = os.path.splitext(os.path.basename(filepath))[0]
                if "_mixture" in basename:
                    song_files.append(filepath)
                    print(f"  + Trovato: {filename}")
                else:
                    print(f"  - Ignorato: {filename} (non contiene '_mixture')")
    elif os.path.isfile(args.path_canzoni) and is_audio_file(args.path_canzoni):
        basename = os.path.splitext(os.path.basename(args.path_canzoni))[0]
        if "_mixture" in basename:
            song_files.append(args.path_canzoni)
            print(f"  + Aggiunto file singolo: {os.path.basename(args.path_canzoni)}")
        else:
            print(f"  - File ignorato: {os.path.basename(args.path_canzoni)} (non contiene '_mixture')")
    
    print(f"Trovati {len(song_files)} file canzone validi")
    
    # Processa il percorso dei rumori
    print(f"\n=========== FILE RUMORI ===========")
    print(f"Percorso rumori: {args.path_rumori}")
    print(f"Percorso assoluto: {os.path.abspath(args.path_rumori)}")
    
    # Debug del percorso rumori
    if os.path.exists(args.path_rumori):
        if os.path.isdir(args.path_rumori):
            print(f"La directory esiste e contiene {len(os.listdir(args.path_rumori))} elementi")
            if args.debug:
                print("Contenuto directory:")
                for item in os.listdir(args.path_rumori):
                    print(f"  - {item}")
        else:
            print(f"Il percorso esiste ed è un file")
    else:
        print(f"ATTENZIONE: Il percorso non esiste!")
    
    noise_files = []
    if os.path.isdir(args.path_rumori):
        print(f"Cercando file di rumore che terminano con '_zeromean'...")
        for filename in os.listdir(args.path_rumori):
            filepath = os.path.join(args.path_rumori, filename)
            if os.path.isfile(filepath) and is_audio_file(filepath):
                basename = os.path.splitext(os.path.basename(filepath))[0]
                if "_zeromean" in basename:
                    noise_files.append(filepath)
                    print(f"  + Trovato: {filename}")
                else:
                    print(f"  - Ignorato: {filename} (non contiene '_zeromean')")
    elif os.path.isfile(args.path_rumori) and is_audio_file(args.path_rumori):
        basename = os.path.splitext(os.path.basename(args.path_rumori))[0]
        if "_zeromean" in basename:
            noise_files.append(args.path_rumori)
            print(f"  + Aggiunto file singolo: {os.path.basename(args.path_rumori)}")
        else:
            print(f"  - File ignorato: {os.path.basename(args.path_rumori)} (non contiene '_zeromean')")
    
    print(f"Trovati {len(noise_files)} file rumore validi")
    print("=====================================\n")
    
    if not song_files:
        print("Nessun file canzone valido trovato. Uscita.")
        return
    
    if not noise_files:
        print("Nessun file rumore valido trovato. Uscita.")
        return
    
    # Definisci le impostazioni di output in base al formato scelto
    format_settings = {
        "aac": {"format": "mp4", "codec": "aac", "bits_per_sample": 16, "compression": -2},  # AAC a 256kbps
        "flac": {"format": "flac", "bits_per_sample": 24, "compression": 12},  # FLAC a massima qualità
        "wav": {"format": "wav", "bits_per_sample": 32, "compression": None},  # WAV a 32-bit float
        "mp3": {"format": "mp3", "codec": "libmp3lame", "bits_per_sample": 16, "compression": 0}  # MP3 a massima qualità
    }
    
    chosen_format = format_settings[args.formato_output]
    print(f"\nImpostazioni formato di output ({args.formato_output}):")
    for k, v in chosen_format.items():
        print(f"  - {k}: {v}")
    
    # Processa ogni canzone con ogni rumore
    processed_count = 0
    error_count = 0
    
    for song_path in song_files:
        song_basename = os.path.splitext(os.path.basename(song_path))[0]
        # Rimuovi il suffisso "_mixture" dal nome della canzone
        song_clean_name = song_basename.replace("_mixture", "")
        
        print(f"\n=========== PROCESSING {song_clean_name} ===========")
        
        try:
            # Carica la canzone usando la funzione migliorata
            song_waveform, song_sample_rate = load_audio(song_path)
            song_waveform = song_waveform.to(device)
            
            # Converti a stereo se necessario per l'output finale
            song_channels = song_waveform.shape[0]
            if song_channels == 1 and args.formato_output != "aac":
                song_waveform = convert_to_stereo(song_waveform)
            
            song_length = song_waveform.shape[1]
            
            for noise_path in noise_files:
                noise_basename = os.path.splitext(os.path.basename(noise_path))[0]
                # Rimuovi il suffisso "_zeromean" dal nome del rumore
                noise_clean_name = noise_basename.replace("_zeromean", "")
                
                start_time = time.time()
                print(f"\nElaborazione di '{song_clean_name}' con rumore '{noise_clean_name}'...")
                
                try:
                    # Carica il rumore usando la funzione migliorata
                    noise_waveform, noise_sample_rate = load_audio(noise_path)
                    noise_waveform = noise_waveform.to(device)
                    
                    # Normalizza il rumore
                    noise_waveform = normalize_audio(noise_waveform)
                    
                    # Applica l'aggiustamento del volume se specificato
                    if args.volume_rumore_db != 0:
                        noise_waveform = adjust_volume(noise_waveform, args.volume_rumore_db)
                        print(f"Volume rumore regolato di {args.volume_rumore_db} dB")
                    
                    # Ricampiona il rumore se necessario
                    if noise_sample_rate != song_sample_rate:
                        print(f"Ricampionamento da {noise_sample_rate}Hz a {song_sample_rate}Hz")
                        noise_waveform = resample_if_needed(noise_waveform, noise_sample_rate, song_sample_rate)
                    
                    # Adatta la durata del rumore a quella della canzone (loop o troncamento)
                    original_length = noise_waveform.shape[1]
                    noise_waveform = loop_or_truncate(noise_waveform, song_sample_rate, song_length, song_sample_rate)
                    new_length = noise_waveform.shape[1]
                    
                    if original_length < new_length:
                        print(f"Rumore esteso da {original_length} a {new_length} campioni (loop)")
                    elif original_length > new_length:
                        print(f"Rumore troncato da {original_length} a {new_length} campioni")
                    
                    # Converti a stereo se necessario
                    noise_channels = noise_waveform.shape[0]
                    if noise_channels == 1 and song_waveform.shape[0] == 2:
                        noise_waveform = convert_to_stereo(noise_waveform)
                        print("Rumore convertito da mono a stereo")
                    elif noise_channels == 2 and song_waveform.shape[0] == 1:
                        # Nel caso in cui il rumore sia stereo ma la canzone sia mono
                        noise_waveform = torch.mean(noise_waveform, dim=0, keepdim=True)
                        print("Rumore convertito da stereo a mono")
                    
                    # Assicurati che le forme siano compatibili
                    if noise_waveform.shape[0] != song_waveform.shape[0]:
                        if noise_waveform.shape[0] > song_waveform.shape[0]:
                            noise_waveform = noise_waveform[:song_waveform.shape[0]]
                        else:
                            noise_waveform = convert_to_stereo(noise_waveform)
                            song_waveform = convert_to_stereo(song_waveform)
                    
                    # Sovrapponi il rumore alla canzone
                    merged_waveform = song_waveform + noise_waveform
                    
                    # Normalizza il risultato per evitare clipping
                    merged_waveform = normalize_audio(merged_waveform)
                    
                    # Definisci il nome del file di output
                    output_filename = f"[{noise_clean_name}] {song_clean_name}.{args.formato_output if args.formato_output != 'aac' else 'm4a'}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Salva il file risultante usando la funzione migliorata
                    save_audio(
                        merged_waveform,
                        song_sample_rate,
                        output_path,
                        chosen_format
                    )
                    
                    processing_time = time.time() - start_time
                    print(f"✓ File salvato in: {output_path}")
                    print(f"  Tempo di elaborazione: {processing_time:.2f}s")
                    processed_count += 1
                    
                except Exception as e:
                    print(f"✗ Errore nell'elaborazione del rumore '{noise_path}':")
                    print(f"  {str(e)}")
                    error_count += 1
                    continue
                    
        except Exception as e:
            print(f"✗ Errore nell'elaborazione della canzone '{song_path}':")
            print(f"  {str(e)}")
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