#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Audio Processing Script for Neural Network Training Data Generation
Creates pairs of audio signals (input, target) by mixing songs with different noise samples
"""

import os
import argparse
import random
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import subprocess
import tempfile
from typing import List, Tuple, Optional

# Default global constants
DEFAULT_NOISE_DIR = "../UrbanSound8K/audio/"
DEFAULT_MUSIC_DIR = "../musdb18/train/"
DEFAULT_OUTPUT_DIR = "output_audio_generato"
ITER_SONGS = 5
ITER_NOISE = 20

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate audio pairs for neural network training')
    parser.add_argument('--music_dir', type=str, help='Path to MP4 music directory')
    parser.add_argument('--noise_dir', type=str, help='Path to noise WAV files directory')
    parser.add_argument('--output_dir', type=str, help='Path for saving generated audio files')
    
    args = parser.parse_args()
    
    # Use provided paths or defaults
    path_canzoni = args.music_dir if args.music_dir else DEFAULT_MUSIC_DIR
    path_rumori = args.noise_dir if args.noise_dir else DEFAULT_NOISE_DIR
    path_output = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    
    return path_canzoni, path_rumori, path_output

def setup_device():
    """Setup PyTorch device (GPU if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_output_directory(path_output):
    """Create output directory if it doesn't exist"""
    os.makedirs(path_output, exist_ok=True)

def load_mp4_audio(file_path, device):
    """
    Load audio from MP4 file (first track) using ffmpeg if needed
    Returns tensor and sample rate
    """
    try:
        # Try direct loading with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform.to(device), sample_rate
    except Exception:
        # If direct loading fails, use ffmpeg to extract audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            temp_path = temp_file.name
            # Extract first audio track (0) using ffmpeg
            subprocess.run([
                'ffmpeg', '-y', '-i', file_path, 
                '-map', '0:a:0', '-c:a', 'pcm_s16le', 
                temp_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Load the temporary WAV file
            waveform, sample_rate = torchaudio.load(temp_path)
            return waveform.to(device), sample_rate

def normalize_tensor(tensor):
    """Normalize tensor to range [-1.0, 1.0]"""
    if tensor.dtype.is_floating_point:
        max_abs_val = torch.max(torch.abs(tensor)).clamp(min=1e-8)
        return (tensor / max_abs_val).float()
    else:
        # If integer tensor, divide by max possible value for the dtype
        info = torch.iinfo(tensor.dtype)
        return (tensor.float() / info.max)

def adjust_channels(tensor, target_channels):
    """Ensure tensor has the same number of channels as target"""
    current_channels = tensor.shape[0]
    
    if current_channels == target_channels:
        return tensor
    elif current_channels == 1 and target_channels == 2:
        # Mono to stereo: duplicate the channel
        return tensor.repeat(2, 1)
    elif current_channels == 2 and target_channels == 1:
        # Stereo to mono: average the channels
        return torch.mean(tensor, dim=0, keepdim=True)
    else:
        # More complex cases: just use the first 'target_channels' channels
        return tensor[:target_channels]

def adjust_length(tensor, target_length):
    """Adjust tensor length to match target_length by repeating or truncating"""
    current_length = tensor.shape[1]
    
    if current_length == target_length:
        return tensor
    elif current_length > target_length:
        # Truncate
        return tensor[:, :target_length]
    else:
        # Calculate how many complete repeats we need
        repeats = target_length // current_length
        remainder = target_length % current_length
        
        # Create the repeated tensor
        repeated = torch.cat([tensor] * repeats, dim=1)
        
        # Add the remainder if needed
        if remainder > 0:
            remainder_tensor = tensor[:, :remainder]
            result = torch.cat([repeated, remainder_tensor], dim=1)
            return result
        else:
            return repeated

def prepare_noise_tensor(noise_path, sr_canzone, channels, length, device):
    """Load, normalize, resample, zero-mean, and adjust noise tensor"""
    # Load noise
    tensor_rumore, sr_rumore = torchaudio.load(noise_path)
    tensor_rumore = tensor_rumore.to(device)
    
    # Normalize
    tensor_rumore = normalize_tensor(tensor_rumore)
    
    # Resample if needed
    if sr_rumore != sr_canzone:
        resampler = T.Resample(sr_rumore, sr_canzone).to(device)
        tensor_rumore = resampler(tensor_rumore)
    
    # Zero mean
    tensor_rumore = tensor_rumore - tensor_rumore.mean(dim=1, keepdim=True)
    
    # Match channels
    tensor_rumore = adjust_channels(tensor_rumore, channels)
    
    # Match length
    tensor_rumore = adjust_length(tensor_rumore, length)
    
    return tensor_rumore

def get_random_noise_pair(path_rumori, prev_pair=None):
    """
    Select two noise files from different subdirectories, 
    ensuring they're different from the previous pair
    """
    # Get subdirectory paths
    noise_subdirs = [p for p in Path(path_rumori).iterdir() if p.is_dir()]
    if len(noise_subdirs) < 2:
        raise ValueError(f"Need at least 2 subdirectories in {path_rumori}")
    
    # Select two different subdirectories
    subdir1, subdir2 = random.sample(noise_subdirs, 2)
    
    # Get WAV files from each subdirectory
    files1 = list(subdir1.glob('*.wav'))
    files2 = list(subdir2.glob('*.wav'))
    
    if not files1 or not files2:
        raise ValueError(f"No WAV files found in one of the subdirectories")
    
    # Select random files
    file1 = random.choice(files1)
    file2 = random.choice(files2)
    
    # Ensure the pair is different from the previous one
    if prev_pair and (file1, file2) == prev_pair:
        # Try to select different files
        attempts = 0
        while (file1, file2) == prev_pair and attempts < 5:
            file1 = random.choice(files1)
            file2 = random.choice(files2)
            attempts += 1
        
        # If still the same, swap subdirectories
        if (file1, file2) == prev_pair:
            file1 = random.choice(files2)
            file2 = random.choice(files1)
    
    return file1, file2

def process_song_and_noises(song_path, path_rumori, path_output, device):
    """
    Process a song with multiple noise pairs to create input/target pairs
    Returns a list of (input, target) tensor pairs for the song
    """
    # Load and normalize song
    print(f"Loading song: {Path(song_path).name}")
    tensor_canzone_base, sr_canzone = load_mp4_audio(song_path, device)
    tensor_canzone_base = normalize_tensor(tensor_canzone_base).to(device)
    
    # Get song dimensions
    num_channels = tensor_canzone_base.shape[0]
    L_canzone = tensor_canzone_base.shape[1]
    
    # Initialize list to store tensor pairs
    lista_coppie_tensori = []
    
    # Track previous noise pair to avoid repetition
    prev_noise_pair = None
    
    # Process ITER_NOISE noise pairs
    for i in range(ITER_NOISE):
        if i % 10 == 0:  # Reduced feedback
            print(f"  Processing noise pair {i+1}/{ITER_NOISE}")
        
        # Select two different noise files
        file_rumore1, file_rumore2 = get_random_noise_pair(path_rumori, prev_noise_pair)
        prev_noise_pair = (file_rumore1, file_rumore2)
        
        # Prepare noise tensors
        tensor_rumore1_proc = prepare_noise_tensor(
            file_rumore1, sr_canzone, num_channels, L_canzone, device
        )
        
        tensor_rumore2_proc = prepare_noise_tensor(
            file_rumore2, sr_canzone, num_channels, L_canzone, device
        )
        
        # Create input tensor (song + noise1)
        mix_input_temp = tensor_canzone_base + tensor_rumore1_proc
        peak_input = torch.max(torch.abs(mix_input_temp)).clamp(min=1e-8)
        tensor_input = mix_input_temp / peak_input
        
        # Create target tensor (song + noise2)
        mix_target_temp = tensor_canzone_base + tensor_rumore2_proc
        peak_target = torch.max(torch.abs(mix_target_temp)).clamp(min=1e-8)
        tensor_target = mix_target_temp / peak_target
        
        # Save audio files
        nome_base_canzone = Path(song_path).stem
        nome_base_rumore1 = file_rumore1.stem
        nome_base_rumore2 = file_rumore2.stem
        
        nome_file_input = f"{nome_base_canzone}_{nome_base_rumore1}_INPUT.wav"
        nome_file_target = f"{nome_base_canzone}_{nome_base_rumore2}_TARGET.wav"
        
        path_file_input = Path(path_output) / nome_file_input
        path_file_target = Path(path_output) / nome_file_target
        
        # Save the audio files
        torchaudio.save(
            path_file_input, 
            tensor_input.cpu(), 
            sr_canzone, 
            encoding="PCM_S", 
            bits_per_sample=16
        )
        
        torchaudio.save(
            path_file_target, 
            tensor_target.cpu(), 
            sr_canzone, 
            encoding="PCM_S", 
            bits_per_sample=16
        )
        
        # Store tensors for later use
        lista_coppie_tensori.append((tensor_input.clone(), tensor_target.clone()))
    
    return lista_coppie_tensori

def main():
    """Main function to process songs and noises"""
    print("Avvio script...")
    
    # Parse arguments and setup
    path_canzoni, path_rumori, path_output = parse_arguments()
    device = setup_device()
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    ensure_output_directory(path_output)
    print(f"Salvataggio file in: {path_output}")
    
    # Get list of MP4 files
    song_files = list(Path(path_canzoni).glob("*.mp4"))
    if not song_files:
        raise ValueError(f"No MP4 files found in {path_canzoni}")
    
    # Track previously selected song to avoid repetition
    prev_song = None
    
    # Main results list
    all_song_results = []
    
    # Process ITER_SONGS songs
    for i in range(min(ITER_SONGS, len(song_files))):
        print(f"Elaborazione canzone {i+1}/{ITER_SONGS}")
        
        # Select random song (different from previous)
        available_songs = [s for s in song_files if s != prev_song]
        if not available_songs:
            available_songs = song_files  # Reset if all songs used
        
        song_path = random.choice(available_songs)
        prev_song = song_path
        
        # Process song with noise pairs
        song_tensor_pairs = process_song_and_noises(song_path, path_rumori, path_output, device)
        
        # Append results
        all_song_results.append(song_tensor_pairs)
    
    print("Elaborazione completata. Restituzione dati.")
    return all_song_results

if __name__ == "__main__":
    main()