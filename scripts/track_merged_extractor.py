#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import subprocess
import os
import importlib
import shutil # For checking ffmpeg

# --- Dependency Check and Installation ---
print("Checking required Python packages...")

# Add ffmpeg-python to required packages
required_packages = {
    "ffmpeg": "ffmpeg-python", # Module name is ffmpeg, install name is ffmpeg-python
    "numpy": "numpy",         # Keep for potential future use
    "pydub": "pydub"          # Keep for potential future use or format checks
}
packages_to_install = []
missing_check_tool = False

# --- [Dependency check and installation logic remains the same as before] ---
# ... (omitted for brevity, assume the previous block is here) ...
try:
    # Use importlib.metadata (Python 3.8+)
    import importlib.metadata as metadata
    print("Using 'importlib.metadata' for package checks.")
    for import_name, install_name in required_packages.items():
        try:
            metadata.distribution(install_name)
        except metadata.PackageNotFoundError:
            # Handle special case where module name differs from install name
            if install_name == "ffmpeg-python":
                 try:
                    importlib.import_module("ffmpeg") # Check if the module itself exists
                 except ImportError:
                    print(f"Required package '{install_name}' (module: {import_name}) not found.")
                    packages_to_install.append(install_name)
            else:
                print(f"Required package '{install_name}' not found.")
                packages_to_install.append(install_name)

except ImportError:
    # Fallback for Python < 3.8 or if importlib.metadata fails
    print("Warning: 'importlib.metadata' not available. Falling back to older checks.", file=sys.stderr)
    try:
        import pkg_resources # pyright: ignore[reportMissingImports]
        print("Using 'pkg_resources' for package checks.")
        for import_name, install_name in required_packages.items():
            try:
                pkg_resources.get_distribution(install_name)
            except pkg_resources.DistributionNotFound:
                print(f"Required package '{install_name}' not found.")
                packages_to_install.append(install_name)
    except ImportError:
        print("Warning: 'pkg_resources' not available. Falling back to simple import check.", file=sys.stderr)
        missing_check_tool = True
        for import_name, install_name in required_packages.items():
             try:
                 importlib.import_module(import_name)
             except ImportError:
                 print(f"Required package '{install_name}' could not be imported.")
                 packages_to_install.append(install_name) # Use install name for pip
        packages_to_install = list(set(packages_to_install))

# --- Installation logic ---
if packages_to_install:
    print(f"\nAttempting to install missing packages: {', '.join(packages_to_install)}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        install_command = [sys.executable, "-m", "pip", "install", "-q"] + packages_to_install
        print(f"Running: {' '.join(install_command)}")
        subprocess.check_call(install_command)
        print("Successfully installed missing packages.")
        # Invalidate import caches if possible (especially important for ffmpeg-python)
        importlib.invalidate_caches()
        if 'pkg_resources' in sys.modules:
             pkg_resources._initialize_master_working_set() # Refresh pkg_resources cache
        print("Attempting to re-verify imports...")
        for import_name in required_packages.keys():
             try:
                  importlib.import_module(import_name)
                  print(f"Successfully imported '{import_name}' after installation.")
             except ImportError as e:
                  print(f"Warning: Could not import '{import_name}' immediately after installation: {e}", file=sys.stderr)
                  # If it's ffmpeg, the executable might still be the issue
                  if import_name == 'ffmpeg':
                      print("Ensure the FFmpeg *executable* is installed and in your PATH.", file=sys.stderr)

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to install packages using pip: {e}", file=sys.stderr)
        print(f"Please try installing manually: pip install {' '.join(packages_to_install)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during package installation: {e}", file=sys.stderr)
        sys.exit(1)
elif not missing_check_tool:
    print("All required Python packages seem to be installed.")
elif missing_check_tool:
     print("Package check was limited; assuming packages are installed if no import errors occurred.")


# --- Check for FFmpeg Executable (CRUCIAL) ---
print("\nChecking for FFmpeg executable...")
ffmpeg_path = shutil.which("ffmpeg")

if not ffmpeg_path:
    print("\n--- FATAL ERROR ---", file=sys.stderr)
    print("FFmpeg executable not found in system PATH.", file=sys.stderr)
    print("This script relies heavily on FFmpeg to process the Stem MP4 file.", file=sys.stderr)
    print("Please install FFmpeg and ensure it's in your PATH.", file=sys.stderr)
    print("Download from: https://ffmpeg.org/download.html", file=sys.stderr)
    print("-------------------\n")
    sys.exit(1) # Exit, cannot proceed without ffmpeg executable
else:
    print(f"Found FFmpeg executable at: {ffmpeg_path}")


# --- Now safely import the required libraries ---
try:
    import ffmpeg # ffmpeg-python library
except ImportError as e:
    print(f"\nERROR: Failed to import required libraries even after installation check: {e}", file=sys.stderr)
    print("There might be an issue with your Python environment or the installation.")
    if 'ffmpeg' in str(e).lower():
         print("Ensure both 'ffmpeg-python' (pip package) and the FFmpeg executable are installed.", file=sys.stderr)
    sys.exit(1)

# --- Import other standard libraries needed ---
import argparse

# --- Core Stem Processing Function ---

def process_stem_file(input_mp4, output_wav, method='extract'):
    """
    Extracts or sums tracks from a Stem MP4 file using FFmpeg and saves as WAV.

    Args:
        input_mp4 (str): Path to the input Stem MP4 file.
        output_wav (str): Path for the output WAV file.
        method (str): 'extract' to get track 0, 'sum' to sum tracks 1-4.
    """
    print(f"\nProcessing '{input_mp4}' using method: '{method}'")
    print(f"Output will be saved to: '{output_wav}'")

    # Basic check if input exists
    if not os.path.isfile(input_mp4):
        print(f"Error: Input file not found: '{input_mp4}'", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists (this handles cases where output_wav includes a path)
    output_dir = os.path.dirname(output_wav)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}': {e}", file=sys.stderr)
            sys.exit(1)

    # Define the input stream
    try:
        in_file = ffmpeg.input(input_mp4)
    except Exception as e:
        print(f"Error defining input stream for '{input_mp4}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Define the processing based on the method ---
    if method == 'extract':
        print("Method: Extracting mixture track (stream 0)")
        processed_stream = in_file['a:0']

    elif method == 'sum':
        print("Method: Summing tracks 1, 2, 3, 4")
        try:
            s_drums = in_file['a:1']
            s_bass = in_file['a:2']
            s_accomp = in_file['a:3']
            s_vocals = in_file['a:4']
            processed_stream = ffmpeg.filter(
                [s_drums, s_bass, s_accomp, s_vocals],
                'amix',
                inputs=4,
                duration='longest',
                dropout_transition=0,
                normalize=False
            )
        except Exception as e:
            print(f"Error setting up stream mapping or amix filter: {e}", file=sys.stderr)
            print("This might indicate the input file doesn't have the expected 5 audio streams.", file=sys.stderr)
            sys.exit(1)
    else:
        # This case should not be reachable due to argparse choices, but good practice
        print(f"Error: Unknown method '{method}'. Use 'extract' or 'sum'.", file=sys.stderr)
        sys.exit(1)

    # --- Define the output ---
    output_process = ffmpeg.output(processed_stream, output_wav, acodec='pcm_s16le')

    # --- Run FFmpeg ---
    print("Running FFmpeg command...")
    try:
        stdout, stderr = output_process.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        # print("FFmpeg stderr:\n", stderr.decode('utf-8')) # Uncomment for debugging ffmpeg issues
        print(f"Successfully created '{output_wav}'")

    except ffmpeg.Error as e:
        print("\n--- FFmpeg Error ---", file=sys.stderr)
        print("FFmpeg command failed.", file=sys.stderr)
        print("FFmpeg stderr output:", file=sys.stderr)
        print(e.stderr.decode('utf-8'), file=sys.stderr)
        print("--------------------\n", file=sys.stderr)
        if os.path.exists(output_wav):
             try:
                 os.remove(output_wav)
                 print(f"Removed potentially incomplete output file: '{output_wav}'")
             except OSError:
                 print(f"Warning: Could not remove incomplete output file '{output_wav}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during FFmpeg execution: {e}", file=sys.stderr)
        sys.exit(1)


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts or sums tracks from a Stem MP4 file (like MUSDB18) into a WAV file using FFmpeg. Output defaults to '<input_filename>.wav'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input Stem MP4 file.")
    parser.add_argument(
        "-o", "--output",
        # No longer required, default is None
        help="Path for the output WAV file. If not provided, defaults to '<input_filename>.wav' in the same directory as the input."
    )
    parser.add_argument(
        "-m", "--method",
        choices=['extract', 'sum'],
        default='extract',
        help="Method to use: 'extract' the first track (mixture), or 'sum' tracks 1-4."
    )

    # Handle cases where script is run without input file argument
    if len(sys.argv) == 1 or sys.argv[1].startswith('-'): # Basic check if first arg is not the input file
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # --- Determine output filename ---
    if args.output:
        # User specified an output path
        output_file = args.output
        # Ensure output filename ends with .wav (and handle if it's a directory)
        if not output_file.lower().endswith(".wav"):
            if os.path.isdir(output_file):
                 print(f"Error: Specified output '{output_file}' is an existing directory. Please provide a full file path ending in .wav or omit -o.", file=sys.stderr)
                 sys.exit(1)
            original_output_file = output_file
            output_file = os.path.splitext(output_file)[0] + ".wav"
            print(f"Warning: Output filename '{original_output_file}' did not end with .wav, adjusting to: '{output_file}'")
    else:
        # Default: Use input filename with .wav extension in the same directory
        input_path = args.input_file
        input_dir = os.path.dirname(input_path)
        input_basename = os.path.basename(input_path)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        output_file = os.path.join(input_dir, f"{input_name_without_ext}.wav")
        print(f"Output file not specified. Defaulting to: '{output_file}'")


    # Final check: Ensure input file exists before calling the function
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file not found: '{args.input_file}'", file=sys.stderr)
        sys.exit(1)

    # Run the main processing function
    process_stem_file(args.input_file, output_file, args.method)

    print("\nScript finished successfully.")