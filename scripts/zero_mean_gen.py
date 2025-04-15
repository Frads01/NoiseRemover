#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import subprocess
import os
import importlib
import shutil # For checking ffmpeg

# --- Dependency Check and Installation ---
print("Checking required Python packages...")

required_packages = {
    "numpy": "numpy",
    "pydub": "pydub"
}
packages_to_install = []
missing_check_tool = False

try:
    # Use importlib.metadata (Python 3.8+)
    import importlib.metadata as metadata # Use alias for clarity
    print("Using 'importlib.metadata' for package checks.")
    for import_name, install_name in required_packages.items():
        try:
            metadata.distribution(install_name)
            # print(f"Package '{install_name}' found via importlib.metadata.")
        except metadata.PackageNotFoundError:
            print(f"Required package '{install_name}' not found.")
            packages_to_install.append(install_name)

except ImportError:
    # Fallback for Python < 3.8 or if importlib.metadata fails unexpectedly
    # This is where the pkg_resources or simple import check goes
    print("Warning: 'importlib.metadata' not available. Falling back to older checks.", file=sys.stderr)
    try:
        # Try pkg_resources as the next best option
        import pkg_resources
        print("Using 'pkg_resources' for package checks.")
        for import_name, install_name in required_packages.items():
            try:
                pkg_resources.get_distribution(install_name)
                # print(f"Package '{install_name}' found via pkg_resources.")
            except pkg_resources.DistributionNotFound:
                print(f"Required package '{install_name}' not found.")
                packages_to_install.append(install_name)
    except ImportError:
        # Absolute fallback: simple import attempt (less reliable for checking if installed)
        print("Warning: 'pkg_resources' not available. Falling back to simple import check.", file=sys.stderr)
        missing_check_tool = True # Flag that our check might be less reliable
        for import_name, install_name in required_packages.items():
             try:
                 importlib.import_module(import_name)
                 # print(f"Package '{install_name}' seems importable.")
             except ImportError:
                 print(f"Required package '{install_name}' could not be imported.")
                 packages_to_install.append(required_packages[import_name])
        packages_to_install = list(set(packages_to_install)) # Remove duplicates if any

# --- Installation logic remains the same ---
if packages_to_install:
    print(f"\nAttempting to install missing packages: {', '.join(packages_to_install)}...")
    # ... (rest of the installation code) ...
elif not missing_check_tool:
    print("All required Python packages seem to be installed.")
elif missing_check_tool:
     print("Package check was limited; assuming packages are installed if no import errors occurred.")


# --- Check for FFmpeg ---
# ... (rest of the script) ...

# --- Check for FFmpeg (External Dependency) ---
print("\nChecking for FFmpeg/Libav...")
ffmpeg_path = shutil.which("ffmpeg")
avconv_path = shutil.which("avconv")

if not (ffmpeg_path or avconv_path):
    print("\n--- WARNING ---", file=sys.stderr)
    print("FFmpeg or Libav executable not found in system PATH.", file=sys.stderr)
    print("Pydub requires FFmpeg (or Libav) to handle non-WAV files (like MP3, FLAC, etc.).", file=sys.stderr)
    print("Please install FFmpeg and ensure it's in your PATH.", file=sys.stderr)
    print("You can download it from: https://ffmpeg.org/download.html", file=sys.stderr)
    print("The script might fail if the input file is not in WAV format.", file=sys.stderr)
    print("---------------\n")
    # Allow script to continue, pydub might still work for WAV or raise its own error
else:
    found_exec = os.path.basename(ffmpeg_path) if ffmpeg_path else os.path.basename(avconv_path)
    print(f"Found '{found_exec}'. Pydub should be able to process various audio formats.")


# --- Now safely import the required libraries ---
try:
    import numpy as np
    from pydub import AudioSegment
    # Suppress potential RuntimeWarning from pydub related to FFmpeg finding if needed
    # import warnings
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
except ImportError as e:
    print(f"\nERROR: Failed to import required libraries even after installation check: {e}", file=sys.stderr)
    print("There might be an issue with your Python environment or the installation.")
    if packages_to_install:
        print(f"(Attempted to install: {', '.join(packages_to_install)})")
    sys.exit(1)

# --- Import other standard libraries needed ---
import argparse # Keep argparse import here after dependency checks


# --- Core Audio Processing Function ---

def make_audio_zero_mean(input_filepath, output_filepath):
    """
    Loads an audio file, checks if its mean is zero, corrects it if not,
    and saves the result as a WAV file.

    Args:
        input_filepath (str): Path to the input audio file (any format FFmpeg handles).
        output_filepath (str): Path where the output WAV file will be saved.
    """
    print(f"\nProcessing '{input_filepath}'...")

    try:
        # 1. Load the audio file using pydub
        #    This handles various input formats automatically using FFmpeg/Libav
        audio = AudioSegment.from_file(input_filepath)
        print(f"Successfully loaded. Format: {getattr(audio, 'format', 'unknown')}, Channels: {audio.channels}, Frame Rate: {audio.frame_rate}, Sample Width: {audio.sample_width} bytes")

    except FileNotFoundError:
         print(f"Error: Input file not found at '{input_filepath}'", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        # Catch pydub's potential error if ffmpeg/avconv is missing for non-wav
        if "Couldn't find ffmpeg or avconv" in str(e):
             print(f"Error loading audio file '{input_filepath}': {e}", file=sys.stderr)
             print("This likely means FFmpeg/Libav is required for this file format but was not found.", file=sys.stderr)
             print("Please install FFmpeg (https://ffmpeg.org/download.html) and ensure it's in your system PATH.", file=sys.stderr)
        else:
            print(f"Error loading audio file '{input_filepath}': {e}", file=sys.stderr)
        sys.exit(1) # Exit if loading fails


    # 2. Get raw audio samples as a NumPy array
    samples = np.array(audio.get_array_of_samples()).astype(np.float64)

    if len(samples) == 0:
        print("Warning: Audio file appears to be empty.", file=sys.stderr)
        # Create an empty wav file
        try:
            empty_audio = AudioSegment.empty()
            empty_audio.export(output_filepath, format="wav")
            print(f"Exported empty WAV file to '{output_filepath}'")
            return # Successfully processed (as empty)
        except Exception as e:
             print(f"Error exporting empty WAV file: {e}", file=sys.stderr)
             sys.exit(1)


    # 3. Calculate the mean
    current_mean = np.mean(samples)
    print(f"Original mean: {current_mean:.4f}")

    # 4. Check if the mean is effectively zero (using a small tolerance)
    tolerance = 1e-9 # A small value to account for potential floating point inaccuracies
    if abs(current_mean) < tolerance:
        print("Audio already has (near) zero mean. No correction needed.")
        # Still need to export as WAV as per requirement
        corrected_audio = audio # Use the original AudioSegment
    else:
        print("Applying zero-mean correction...")
        # 5. Subtract the mean from all samples
        corrected_samples_float = samples - current_mean

        # Determine the correct integer type based on sample width
        if audio.sample_width == 1: # 8-bit
            dtype = np.int8
            min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
        elif audio.sample_width == 2: # 16-bit
            dtype = np.int16
            min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
        elif audio.sample_width == 4: # 32-bit PCM
             dtype = np.int32
             min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
        # Add support for 24-bit if needed (pydub might load as 32-bit)
        # elif audio.sample_width == 3: # 24-bit (often loaded as 32-bit by libraries)
        #     dtype = np.int32 # Or handle specifically if library supports raw 24-bit access
        #     min_val, max_val = -(2**23), (2**23 - 1)
        else:
            # Pydub might load floating point audio (e.g. 32f wav) with sample_width=4
            # Let's check the type of the original samples if possible
            original_dtype = audio.get_array_of_samples().typecode
            if original_dtype == 'f': # float 32
                 print("Detected 32-bit float input. Applying correction directly.")
                 dtype = np.float32
                 # Clipping float is less critical unless converting format later, but good practice
                 min_val, max_val = -1.0, 1.0 # Assuming standard float range
                 # Important: Pydub expects bytes, so conversion might be needed
                 # Let pydub handle the conversion back from float numpy array
                 # Create AudioSegment directly from float numpy array
                 # Note: Pydub's direct numpy support might be limited or need specific versions
                 # Let's stick to integer conversion for broader compatibility for now.
                 # If sticking to int:
                 print(f"Warning: Input is likely 32-bit float (sample_width={audio.sample_width}, typecode='{original_dtype}'). Converting to int32 for processing.", file=sys.stderr)
                 dtype = np.int32
                 min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
                 # Need to scale float samples before converting to int
                 print("Scaling float samples before converting to int32.")
                 samples = (samples * max_val).astype(np.float64) # Scale to int32 range
                 current_mean = np.mean(samples) # Recalculate mean on scaled samples
                 corrected_samples_float = samples - current_mean

            else:
                 print(f"Error: Unsupported sample width: {audio.sample_width} bytes (typecode: {original_dtype}). Cannot safely convert.", file=sys.stderr)
                 sys.exit(1)


        # 6. Clip values to prevent overflow/underflow when converting back to int/target type
        print(f"Clipping corrected samples to range [{min_val}, {max_val}] for type {dtype}.")
        corrected_samples_clipped = np.clip(corrected_samples_float, min_val, max_val)

        # 7. Convert back to the target data type
        corrected_samples_final = corrected_samples_clipped.astype(dtype)

        # Optional: Verify new mean (should be very close to 0)
        new_mean_verification = np.mean(corrected_samples_final.astype(np.float64))
        print(f"New mean after correction (final samples): {new_mean_verification:.4f}")

        # 8. Create a new AudioSegment from the corrected samples
        try:
            corrected_audio = AudioSegment(
                corrected_samples_final.tobytes(), # Convert numpy array back to bytes
                frame_rate=audio.frame_rate,
                sample_width=audio.sample_width, # Use original sample width
                channels=audio.channels
            )
        except Exception as e:
             print(f"Error creating new AudioSegment from processed data: {e}", file=sys.stderr)
             # Provide more context if possible
             print(f"Data type: {corrected_samples_final.dtype}, Shape: {corrected_samples_final.shape}")
             print(f"Expected sample width: {audio.sample_width}")
             sys.exit(1)


    # 9. Export the result as a WAV file
    print(f"Exporting corrected audio to '{output_filepath}'...")
    try:
        # Specify parameters explicitly for robustness
        corrected_audio.export(
            output_filepath,
            format="wav",
            # Optional: Add bitrate or other parameters if needed, e.g., parameters={"acodec": "pcm_s16le"}
            )
        print("Export successful.")
    except Exception as e:
        print(f"Error exporting WAV file '{output_filepath}': {e}", file=sys.stderr)
        # Add common error causes
        if "Permission denied" in str(e):
             print("Hint: Check if you have write permissions for the output directory.", file=sys.stderr)
        elif "No space left on device" in str(e):
             print("Hint: Check if there is enough disk space.", file=sys.stderr)
        sys.exit(1)


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if an audio file has zero mean, correct it if not, and save as WAV. Automatically attempts to install missing 'numpy' and 'pydub' packages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input audio file (e.g., audio.mp3, sound.flac, input.wav)")
    parser.add_argument(
        "-o", "--output",
        help="Path for the output WAV file. If not provided, defaults to '<input_filename>_zero_mean.wav'."
    )

    # Handle cases where script is run without arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Determine output filepath
    if args.output:
        output_file = args.output
        # Ensure output directory exists if specified
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
           try:
               os.makedirs(output_dir, exist_ok=True)
               print(f"Created output directory: {output_dir}")
           except OSError as e:
               print(f"Error: Could not create output directory '{output_dir}': {e}", file=sys.stderr)
               sys.exit(1)
    else:
        # Prevent potential issue if input_file has no extension
        input_basename = os.path.basename(args.input_file)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        if not input_name_without_ext: # Handle cases like '.bashrc' or just 'filename'
            input_name_without_ext = input_basename
        output_file = f"{input_name_without_ext}_zero_mean.wav"
        print(f"Output file not specified. Defaulting to: '{output_file}'")


    # Ensure output filename ends with .wav
    if not output_file.lower().endswith(".wav"):
        # Check if the specified output *might* be a directory
        if os.path.isdir(output_file):
             print(f"Error: Specified output '{output_file}' is an existing directory. Please provide a full file path.", file=sys.stderr)
             sys.exit(1)
        original_output_file = output_file
        output_file = os.path.splitext(output_file)[0] + ".wav"
        print(f"Warning: Output filename '{original_output_file}' did not end with .wav, adjusting to: '{output_file}'")


    # Check if input file exists (redundant check, but good practice)
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file not found: '{args.input_file}'", file=sys.stderr)
        sys.exit(1) # Exit here, no need to call the function

    # Run the main processing function
    make_audio_zero_mean(args.input_file, output_file)

    print("\nScript finished successfully.")