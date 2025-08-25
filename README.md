
# Riduzione del rumore delle canzoni senza dati di addestramento puliti

## Descrizione

Questo repository implementa un sistema di **denoising audio musicale** tramite **Deep Learning**, seguendo la metodologia **Noise2Noise**: l’addestramento avviene esclusivamente con segnali rumorosi, **senza la necessità di dati puliti come target**.  
L’obiettivo è dimostrare che è possibile addestrare una rete neurale profonda (Deep Complex U-Net a 20 livelli) direttamente su dati musicali rumorosi per la riduzione del rumore, superando i limiti dei metodi basati su speech o dati puliti, e di validarne le prestazioni e la pipeline di ricerca.

---

## Requisiti in Python

- Creare un ambiente Python ≥3.8 (consigliato uso di venv/conda).  
- Installare i pacchetti richiesti:

```
pip install -r requirements.txt
```

---

## Generazione dei dataset

* **Dataset utilizzati**: [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html), [musdb18](https://sigsep.github.io/datasets/musdb.html)

* **[Dataset generati](LinkDatasetKaggle.md)**

---

- Per generare un dataset con soli rumori reali:

    ```bash
    python merger.py --path-songs <dir_canzoni> --path-noises <dir_rumori> --iter-songs <num_canzoni> --iter-noise <num_rumori_per_canzone> --use-cuda --input-dir <dir_output_input> --target-dir <dir_output_target>
    ```

    - `--path-songs`: Directory delle canzoni in formato MP4 per l'elaborazione.
    - `--path-noises`: Directory contenente file di rumore WAV, suddivisi in sottocartelle fold1,...,fold10.
    - `--iter-songs`: Numero massimo di canzoni da processare.
    - `--iter-noise`: Numero massimo di coppie di rumori da sovrapporre per ogni canzone.
    - `--use-cuda`: Flag opzionale per usare la GPU CUDA, se disponibile.
    - `--input-dir`: Directory dove salvare i file audio di input (canzone + rumore 1).
    - `--target-dir`: Directory dove salvare i file audio target (canzone + rumore 2 o canzone pulita).

- Per generare un dataset per una specifica classe N:

    ```bash
    python merger_class.py --class-num <numero_categoria> --path-canzoni <dir_canzoni> --path-rumori <dir_rumori> --iter-songs <num_canzoni> --iter-noise <num_rumori_per_canzone> --use-cuda --input-dir <dir_output_input> --target-dir <dir_output_target>
    ```
    - `--class-num`: Numero intero della categoria rumore (0-9) da includere nel dataset.
    - `--path-songs`: Directory delle canzoni musdb18.
    - `--path-noises`: Directory dei rumori UrbanSound8K.
    - `--iter-songs`: Numero massimo di canzoni da usare.
    - `--iter-noise`: Numero massimo di rumori da sovrapporre per canzone.
    - `--use-cuda`: Usa GPU CUDA se disponibile.
    - `--input-dir`: Directory output per i file INPUT.
    - `--target-dir`: Directory output per i file TARGET.

- Per generare un dataset con rumori reali e rumore bianco:

    ```bash
    python merger_mixed_white.py --path-songs <dir_canzoni> --path-noises <dir_rumori> --iter-songs <num_canzoni> --iter-noise <num_rumori_per_canzone> --use-cuda --input-dir <dir_output_input> --target-dir <dir_output_target>
    ```

    - `--path-songs`: Directory delle canzoni.
    - `--path-noises`: Directory contenente i file di rumore RAW (UVSound8K).
    - `--iter-songs`: Numero di canzoni per l'elaborazione.
    - `--iter-noise`: Numero di coppie di rumori per canzone.
    - `--use-cuda`: Usa GPU CUDA se disponibile.
    - `--input-dir`: Directory output per file INPUT.
    - `--target-dir`: Directory output per file TARGET.

- Per generare un dataset con solo rumore bianco:

    ```bash
    python merger_white.py --path-songs <dir_canzoni> --iter-songs <num_canzoni> --iter-white-noise <num_rumori_bianchi_per_canzone> --use-cuda --input-dir <dir_output_input> --target-dir <dir_output_target>
    ```

    - `--path-songs`: Directory delle canzoni (MP4).
    - `--iter-songs`: Numero di canzoni da elaborare.
    - `--iter-white-noise`: Numero di coppie di rumore bianco generate per canzone.
    - `--use-cuda`: Usa la GPU CUDA se disponibile.
    - `--input-dir`: Directory per salvare i file INPUT.
    - `--target-dir`: Directory per salvare i file TARGET.

- Per l'analisi di correlazione tra coppie di file audio WAV:

    ```bash
    python correlation.py <path_audio1> <path_audio2> <output_file>
    ```

    - `<path_audio1>`: Percorso della prima directory contenente file audio WAV.
    - `<path_audio2>`: Percorso della seconda directory contenente file audio WAV.
    - `<output_file>`: Percorso del file di output in cui salvare i risultati dell'analisi.

---

## Allenare un nuovo modello

Per allenare un nuovo modello, è disponibile il notebook [`NoiseRemover.ipynb`](NoiseRemover.ipynb). E' necessario dapprima generare un nuovo dataset (singola classe, rumore bianco o mixed). Se utilizzate Windows, impostare `soundfile` come backend torchaudio. Se utilizzate Linux, impostare `sox` come backend torchaudio. Il file .pth dei pesi viene salvato per ogni epoca di addestramento nella directory `Weights`.

---

## Verifica dell'inferenza del modello su pesi preaddestrati

Abbiamo addestrato il nostro modello per tutte le 10 classi di UrbanSound8k (da 0 a 9) e il rumore bianco gaussiano. Tutti i pesi del nostro modello sono salvati nella directory [`pretrained weights`](pretrained_weights).
Nel file [`NoiseRemover.ipynb`](NoiseRemover.ipynb), selezionare il file `.pth` dei pesi per il modello da utilizzare. Indicare le cartelle di test contenenti l'audio che si desidera pulire. Verranno calcolate anche le metriche di qualità audio. I file wav risultanti verranno salvati nella directory `Samples`.

---

## Ridurre il rumore di un intero brano

Per ridurre il rumore di un intero brano, è possibile utilizzare il notebook [`denoising_entire_song.ipynb`](scripts\denoising_entire_song.ipynb) modificando le costanti PATH all'inizio, sostituendo `MODEL_PATH`, `INPUT_AUDIO_PATH` ed eventualmente `OUTPUT_AUDIO_PATH` con il path dei pesi del rumore che si vuol eliminare, del file di input e del file di output.

---

## Usare un brano come dataset di partenza

Il notebook [`generate_best_segments.ipynb`](scripts\generate_best_segments.ipynb) permette di salvare in una cartella i segmenti che combinano in maniera ottimale correlazione e diversità spettrale, ed eventualmente utilizzarlo come nuovo dataset di partenza.

---

## Esempi e risultati

@TODO

---

## Riferimenti

- https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data
- https://arxiv.org/abs/1803.04189 – Noise2Noise, Lehtinen et al. 2018
- https://www.isca-speech.org/archive/interspeech_2021/kashyap21_interspeech.html – Kashyap et al., Interspeech 2021
- https://arxiv.org/abs/1903.03107 – Deep Complex U-Net, Choi et al. 2019
- https://arxiv.org/abs/1705.09792 – Deep Complex Networks, Trabelsi et al. 2018
- https://arxiv.org/abs/1505.04597 – U-Net, Ronneberger et al. 2015
- https://sigsep.github.io/datasets/musdb.html
- https://urbansounddataset.weebly.com/

---

