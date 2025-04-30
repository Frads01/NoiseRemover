#Aggiunge rumore bianco con ampiezza 0.01 e 0.008 ad un dataset di canzoni (tracca 1 dell'.mp4)
# e salva rispettivamente il risultato in due cartelle separate
import os
import sys
import subprocess
import importlib
import shutil

# === DEPENDENCY CHECK ===
print("Controllo pacchetti Python richiesti...")

required_packages = {
    "numpy": "numpy",
    "moviepy": "moviepy",
    "pydub": "pydub"
}
packages_to_install = []

try:
    import importlib.metadata as metadata
    for module, pkg in required_packages.items():
        try:
            metadata.distribution(pkg)
        except metadata.PackageNotFoundError:
            try:
                importlib.import_module(module)
            except ImportError:
                packages_to_install.append(pkg)
except ImportError:
    import pkg_resources
    for module, pkg in required_packages.items():
        try:
            pkg_resources.get_distribution(pkg)
        except pkg_resources.DistributionNotFound:
            packages_to_install.append(pkg)

if packages_to_install:
    print(f"Installazione pacchetti mancanti: {', '.join(packages_to_install)}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)

# === FFMPEG CHECK ===
print("Controllo eseguibile FFmpeg...")
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    print("⚠️ FFmpeg non trovato nel PATH di sistema.")
    print("Scaricalo da: https://ffmpeg.org/download.html")
    sys.exit(1)
else:
    print(f"✔ FFmpeg trovato: {ffmpeg_path}")

# === SAFE IMPORTS ===
import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pydub import AudioSegment

# === FUNZIONE PRINCIPALE ===
def aggiungi_rumore_bianco_audio(cartella):
    for file in os.listdir(cartella):
        if file.lower().endswith(".mp4") and "_rumore_" not in file:
            try:
                video_path = os.path.join(cartella, file)

                # Carica audio
                audio_clip = AudioFileClip(video_path)
                audio_data = audio_clip.to_soundarray(fps=audio_clip.fps)
                mono_audio = audio_data[:, 0]  # Converti in mono

                for ampiezza in [0.01, 0.008]:
                    sottocartella = os.path.join(cartella, f"rumore_{ampiezza}")
                    os.makedirs(sottocartella, exist_ok=True)

                    rumore = np.random.normal(0, ampiezza, len(mono_audio))
                    audio_con_rumore = mono_audio + rumore
                    audio_con_rumore = np.clip(audio_con_rumore, -1.0, 1.0)
                    audio_int16 = (audio_con_rumore * np.iinfo(np.int16).max).astype(np.int16)

                    audio_segment = AudioSegment(
                        audio_int16.tobytes(),
                        frame_rate=audio_clip.fps,
                        sample_width=2,
                        channels=1
                    )

                    nome_file_output = f"{os.path.splitext(file)[0]}_rumore_{ampiezza}.wav"
                    output_path = os.path.join(sottocartella, nome_file_output)

                    audio_segment.export(output_path, format="wav")
                    print(f"🎧 Salvato: {output_path}")

                audio_clip.close()

            except Exception as e:
                print(f"❌ Errore durante l'elaborazione di {file}: {e}")

# === AVVIO SCRIPT ===
if __name__ == "__main__":
    cartella = 'aggiungi il path della cartella che contiene le canzoni in cui aggiungere rumore'
    if not os.path.isdir(cartella):
        print(f"Cartella non trovata: {cartella}")
        sys.exit(1)
    aggiungi_rumore_bianco_audio(cartella)
    print("\n✅ Script completato con successo.")
