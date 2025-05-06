#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Audio Processing Script for Neural Network Training Data Generation
Creates pairs of audio signals (input, target) by mixing songs with different noise samples
"""

import os
import argparse
import random
import shutil
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import subprocess
import tempfile
from typing import List, Tuple, Optional

# Default global constants
DEFAULT_NOISE_DIR = "..\\UrbanSound8K\\audio\\"
DEFAULT_MUSIC_DIR = "..\\musdb18\\train\\"
DEFAULT_OUTPUT_DIR = "tmp"
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
    noise_path = args.noise_dir if args.noise_dir else DEFAULT_NOISE_DIR
    path_output = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    
    return path_canzoni, noise_path, path_output

def setup_device():
    """Setup PyTorch device (GPU if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_output_directory(path_output):
    """Create output directory if it doesn't exist"""
    if os.path.isdir(path_output):
        shutil.rmtree(path_output)

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
        # Fallback to ffmpeg: create a temporary WAV file name
        fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)  # Close the OS-level file descriptor

        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', file_path, '-map', '0:a:0', '-c:a', 'pcm_s16le', temp_wav_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Load the temporary WAV file
            waveform, sample_rate = torchaudio.load(temp_wav_path)
            return waveform.to(device), sample_rate
        except subprocess.CalledProcessError as e_ffmpeg:
            # Re-raise if ffmpeg command itself fails
            raise RuntimeError(f"ffmpeg conversion failed for {file_path}") from e_ffmpeg
        except Exception as e_load_wav:
            # Re-raise if loading the temporary WAV fails
            raise RuntimeError(f"Failed to load temporary WAV {temp_wav_path} (from {file_path}) after ffmpeg conversion.") from e_load_wav
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass # Silently ignore errors during temporary file cleanup

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

def get_random_noise_pair(noise_path, prev_pair=None):
    """
    Select two noise files from different subdirectories, 
    ensuring they're different from the previous pair
    """
    # Get subdirectory paths
    noise_subdirs = [p for p in Path(noise_path).iterdir() if p.is_dir()]
    if len(noise_subdirs) < 2:
        raise ValueError(f"Need at least 2 subdirectories in {noise_path}")
      
    # Select two different subdirectories
    subdir1, subdir2 = random.sample(noise_subdirs, 2)

    # Get WAV files from each subdirectory
    files1 = list(subdir1.glob('*.wav'))
    files2 = list(subdir2.glob('*.wav'))

    if not files1 or not files2:
        raise ValueError(f"No WAV files found in one of the subdirectories")

    # Function to extract classID from filename
    def get_class_id(filename):
        parts = filename.stem.split('-')
        if len(parts) >= 2:
            return parts[1]
        return None
    
    # Select random files with different classIDs
    
    while 1:
        file1 = random.choice(files1)
        file2 = random.choice(files2)
        
        # Check if files have different classIDs
        class_id1 = get_class_id(file1)
        class_id2 = get_class_id(file2)
        
        # Ensure different classIDs and different from previous pair
        if class_id1 and class_id2 and class_id1 != class_id2 and (not prev_pair or (file1, file2) != prev_pair):
            return file1, file2
        

def process_song_and_noises(song_path, noise_path, path_output, device, i):
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
    tensor_tuples_list = []
    
    # Track previous noise pair to avoid repetition
    prev_noise_pair = None

    high_n = np.random.randint(ITER_NOISE)
    
    # Process ITER_NOISE noise pairs
    for j in range(1, high_n+1):
        if j % 5 == 0:  # Reduced feedback
            print(f"  Processing noise pair {j}/{high_n}")
        
        # Select two different noise files
        file_rumore1, file_rumore2 = get_random_noise_pair(noise_path, prev_noise_pair)
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
        nome_base_rumore1 = file_rumore1.stem
        nome_base_rumore2 = file_rumore2.stem
        
        nome_file_input = f"S{i}_N{j}_{nome_base_rumore1}_INPUT.wav"
        nome_file_target = f"S{i}_N{j}_{nome_base_rumore2}_TARGET.wav"
        
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
        tensor_tuples_list.append((tensor_input.clone(), tensor_target.clone()))
    
    return tensor_tuples_list

def main():
    """Main function to process songs and noises"""
    print("Avvio script...")
    
    # Parse arguments and setup
    path_canzoni, noise_path, path_output = parse_arguments()
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

    high_s = np.random.randint(ITER_SONGS)
    
    # Process ITER_SONGS songs
    for i in range(1, high_s+1):
        print(f"Elaborazione canzone {i}/{high_s}")
        
        # Select random song (different from previous)
        available_songs = [s for s in song_files if s != prev_song]
        if not available_songs:
            available_songs = song_files  # Reset if all songs used
        
        song_path = random.choice(available_songs)
        prev_song = song_path
        
        # Process song with noise pairs
        song_tensor_pairs = process_song_and_noises(song_path, noise_path, path_output, device, i)
        
        # Append results
        all_song_results.append(song_tensor_pairs)
    
    print("Elaborazione completata. Restituzione dati.")
    print(all_song_results)
    return all_song_results

if __name__ == "__main__":
    main()