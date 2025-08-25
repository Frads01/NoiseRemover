import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt
import librosa  # Necessario per il ricampionamento
import os
import glob
import argparse
import datetime


# --- CLASSE PER GESTIRE SEGNALI AUDIO ---
class AudioSignal:
    def __init__(self, data, rate, path="N/A"):
        self.data = np.asarray(data, dtype=np.float32)
        self.rate = int(rate)
        self.path = path

    def __len__(self):
        return len(self.data)

    @property
    def duration(self):
        return len(self.data) / self.rate if self.rate > 0 else 0

    def resample(self, target_rate):
        if self.rate == target_rate:
            return self
        if self.rate <= 0:
            print(
                f"Errore: Sample rate originale non valido ({self.rate} Hz) per il segnale da '{self.path}'. Impossibile ricampionare.")
            return AudioSignal(self.data.copy(), self.rate,
                               self.path + " (ricampionamento fallito - rate originale non valido)")

        print(f"Ricampionamento da {self.rate} Hz a {target_rate} Hz per il segnale da '{self.path}'")
        try:
            resampled_data = librosa.resample(self.data.astype(np.float32), orig_sr=self.rate, target_sr=target_rate)
            return AudioSignal(resampled_data, target_rate, self.path + " (ricampionato)")
        except Exception as e:
            print(f"Errore durante il ricampionamento di '{self.path}' da {self.rate} a {target_rate}: {e}")
            return AudioSignal(self.data.copy(), self.rate, self.path + " (ricampionamento fallito)")


# --- FUNZIONE PER CARICARE FILE WAV ---
def load_wav_to_signal(path):
    try:
        rate, data = wav.read(path)
        # Converti in mono se stereo
        if data.ndim > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        # Normalizza a float32 [-1, 1] se è un tipo intero
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
            max_abs_val = np.max(np.abs(data))
            if max_abs_val > 1.0 and max_abs_val < 2.0 ** 15:
                data /= max_abs_val
        return AudioSignal(data, rate, path)
    except FileNotFoundError:
        print(f"Errore: File '{path}' non trovato.")
        return None
    except Exception as e:
        print(f"Errore nel caricamento di '{path}': {e}")
        return None


# --- FUNZIONE PER CALCOLARE CORRELAZIONE E AUTOCORRELAZIONE ---
def calculate_correlation_analysis(audio_signal1, audio_signal2):
    """
    Calcola correlazione incrociata, autocorrelazione e covarianza tra due segnali audio.
    Implementa i concetti teorici di correlazione, covarianza e autocorrelazione.

    Returns:
        dict: Dizionario contenente i risultati dell'analisi
    """
    if not isinstance(audio_signal1, AudioSignal) or not isinstance(audio_signal2, AudioSignal):
        print("Errore: gli input devono essere oggetti AudioSignal.")
        return None

    s1 = audio_signal1
    s2 = audio_signal2

    # Gestione Sample Rate Diversi
    if s1.rate <= 0 or s2.rate <= 0:
        print(f"Errore: Sample rate non validi. S1: {s1.rate} Hz, S2: {s2.rate} Hz")
        return None

    if s1.rate != s2.rate:
        s2 = s2.resample(s1.rate)
        if s2.rate != s1.rate:
            print(f"Fallimento nel ricampionamento. Analisi interrotta.")
            return None

    data1 = s1.data
    data2 = s2.data

    if len(data1) == 0 or len(data2) == 0:
        print("Errore: uno o entrambi i segnali audio sono vuoti.")
        return None

    # Troncamento alla lunghezza minima
    min_len = min(len(data1), len(data2))
    if min_len < 2:
        print("Errore: lunghezza minima dei segnali insufficiente per calcolare la correlazione.")
        return None

    x1 = data1[:min_len]
    x2 = data2[:min_len]

    try:
        # 1. CORRELAZIONE (Momento misto di ordine (1,1))
        # m(1,1)_XX = E{x1 * x2} = correlazione tra i due segnali
        correlation_coefficient = np.corrcoef(x1, x2)[0, 1]

        # 2. MEDIE dei segnali
        mean_x1 = np.mean(x1)
        mean_x2 = np.mean(x2)

        # 3. COVARIANZA (Momento misto centrato)
        # σ(x1,x2) = E{(x1-mx1)(x2-mx2)} = E{x1*x2} - mx1*mx2
        covariance = np.mean((x1 - mean_x1) * (x2 - mean_x2))

        # Verifica teorica: covarianza = correlazione - prodotto delle medie
        theoretical_covariance = np.mean(x1 * x2) - mean_x1 * mean_x2

        # 4. POTENZA dei segnali (E{x^2})
        power_x1 = np.mean(x1 ** 2)
        power_x2 = np.mean(x2 ** 2)

        # 5. AUTOCORRELAZIONE per τ=0 (massimo dell'autocorrelazione)
        # Rx(0) = E{x^2} = potenza del segnale
        autocorr_x1_zero = power_x1
        autocorr_x2_zero = power_x2

        # 6. INTERCORRELAZIONE per τ=0
        # Rxy(0) = E{x1 * x2}
        intercorrelation_zero = np.mean(x1 * x2)

        # 7. COEFFICIENTE DI CORRELAZIONE NORMALIZZATO
        # ρ = σ(x1,x2) / (σx1 * σx2)
        std_x1 = np.std(x1, ddof=1) if len(x1) > 1 else 0
        std_x2 = np.std(x2, ddof=1) if len(x2) > 1 else 0

        if std_x1 > 0 and std_x2 > 0:
            normalized_correlation = covariance / (std_x1 * std_x2)
        else:
            normalized_correlation = 0.0

        # 8. TEST DI INCORRELAZIONE
        # Se covarianza ≈ 0, i segnali sono incorrelati
        tolerance = 1e-6
        are_uncorrelated = abs(covariance) < tolerance

        results = {
            'correlation_coefficient': correlation_coefficient,
            'covariance': covariance,
            'theoretical_covariance': theoretical_covariance,
            'mean_x1': mean_x1,
            'mean_x2': mean_x2,
            'power_x1': power_x1,
            'power_x2': power_x2,
            'autocorr_x1_zero': autocorr_x1_zero,
            'autocorr_x2_zero': autocorr_x2_zero,
            'intercorrelation_zero': intercorrelation_zero,
            'normalized_correlation': normalized_correlation,
            'std_x1': std_x1,
            'std_x2': std_x2,
            'are_uncorrelated': are_uncorrelated,
            'sample_rate': s1.rate,
            'signal_length': min_len
        }

        return results

    except Exception as e:
        print(f"Errore nel calcolo dell'analisi di correlazione: {e}")
        return None


# --- FUNZIONE PER OTTENERE LISTA ORDINATA DI FILE WAV ---
def get_sorted_audio_files(directory):
    if not os.path.exists(directory):
        print(f"Errore: Directory '{directory}' non trovata.")
        return []
    pattern = os.path.join(directory, "*.wav")
    files = glob.glob(pattern)
    files.sort()
    return files


# --- FUNZIONE PER PROCESSARE TUTTE LE COPPIE ---
def process_all_pairs(path_audio1, path_audio2, output_file):
    print(f"Ricerca file in: {path_audio1}")
    files1 = get_sorted_audio_files(path_audio1)

    print(f"Ricerca file in: {path_audio2}")
    files2 = get_sorted_audio_files(path_audio2)

    if not files1:
        print(f"Nessun file .wav trovato in {path_audio1}")
        return
    if not files2:
        print(f"Nessun file .wav trovato in {path_audio2}")
        return

    print(f"Trovati {len(files1)} file in {path_audio1}")
    print(f"Trovati {len(files2)} file in {path_audio2}")

    num_pairs = min(len(files1), len(files2))
    print(f"Processando {num_pairs} coppie di file...")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    detailed_results = []

    for i in range(num_pairs):
        file1 = files1[i]
        file2 = files2[i]

        filename1 = os.path.splitext(os.path.basename(file1))[0]
        filename2 = os.path.splitext(os.path.basename(file2))[0]

        print(f"\nProcessando coppia {i + 1}/{num_pairs}:")
        print(f"  File 1: {filename1}")
        print(f"  File 2: {filename2}")

        audio_signal_1 = load_wav_to_signal(file1)
        audio_signal_2 = load_wav_to_signal(file2)

        if audio_signal_1 is not None and audio_signal_2 is not None:
            analysis_results = calculate_correlation_analysis(audio_signal_1, audio_signal_2)

            if analysis_results is not None:
                try:
                    pair_id = filename1.split("-")[1]
                except IndexError:
                    pair_id = filename1

                correlation = analysis_results['correlation_coefficient']
                covariance = analysis_results['covariance']
                power_x1 = analysis_results['power_x1']
                power_x2 = analysis_results['power_x2']
                are_uncorrelated = analysis_results['are_uncorrelated']

                result_line = (f"{pair_id} - Correlazione: {correlation:.6f}, "
                               f"Covarianza: {covariance:.6f}, "
                               f"Incorrelati: {'Sì' if are_uncorrelated else 'No'}, "
                               f"Potenza_X1: {power_x1:.6f}, "
                               f"Potenza_X2: {power_x2:.6f}")

                results.append(result_line)

                detailed_result = {
                    'pair_id': pair_id,
                    'filename1': filename1,
                    'filename2': filename2,
                    **analysis_results
                }
                detailed_results.append(detailed_result)

                print(f"  Correlazione: {correlation:.6f}")
                print(f"  Covarianza: {covariance:.6f}")
                print(f"  Incorrelati: {'Sì' if are_uncorrelated else 'No'}")
                print(f"  Potenza X1: {power_x1:.6f}")
                print(f"  Potenza X2: {power_x2:.6f}")

            else:
                try:
                    pair_id = filename1.split("-")[1]
                except IndexError:
                    pair_id = filename1
                result_line = f"{pair_id} - Correlazione: ERRORE, Covarianza: ERRORE, Incorrelati: ERRORE, Potenza_X1: ERRORE, Potenza_X2: ERRORE"
                results.append(result_line)
                print(f"  Errore nel calcolo dell'analisi di correlazione")
        else:
            try:
                pair_id = filename1.split("-")[1] if audio_signal_1 is not None else f"ERROR_{i + 1}"
            except IndexError:
                pair_id = filename1 if audio_signal_1 is not None else f"ERROR_{i + 1}"
            result_line = f"{pair_id} - Correlazione: ERRORE_CARICAMENTO, Covarianza: ERRORE_CARICAMENTO, Incorrelati: ERRORE_CARICAMENTO, Potenza_X1: ERRORE_CARICAMENTO, Potenza_X2: ERRORE_CARICAMENTO"
            results.append(result_line)
            print(f"  Errore nel caricamento dei file")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Risultati Analisi di Correlazione Audio\n")
            f.write("# Basato su teoria della correlazione per processi stocastici\n")
            f.write("#\n")
            f.write("# PARAMETRI CALCOLATI:\n")
            f.write("# - Correlazione: Coefficiente di correlazione di Pearson (momento misto di ordine 1,1)\n")
            f.write("# - Covarianza: Momento misto centrato σ(x1,x2) = E{(x1-mx1)(x2-mx2)}\n")
            f.write("# - Incorrelati: Test se |covarianza| < 1e-6 (Sì/No)\n")
            f.write("# - Potenza_X1: E{x1²} = Rx1(0) (autocorrelazione a τ=0)\n")
            f.write("# - Potenza_X2: E{x2²} = Rx2(0) (autocorrelazione a τ=0)\n")
            f.write("#\n")
            f.write("# FORMATO: ID_COPPIA - Correlazione: VALORE, Covarianza: VALORE, Incorrelati: SÌ/NO, Potenza_X1: VALORE, Potenza_X2: VALORE\n")
            f.write("#\n")
            f.write(f"# Generato il: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"# Directory input: {path_audio1}\n")
            f.write(f"# Directory target: {path_audio2}\n")
            f.write("#\n")

            for result in results:
                f.write(result + '\n')

            f.write("\n# === STATISTICHE FINALI ===\n")

            successful_correlations = [r for r in results if not ('ERRORE' in r)]
            f.write(f"# Totale coppie processate: {len(results)}\n")
            f.write(f"# Correlazioni calcolate con successo: {len(successful_correlations)}\n")

            if successful_correlations:
                correlations_values = []
                covariances_values = []
                powers_x1_values = []
                powers_x2_values = []
                uncorrelated_count = 0

                for result in successful_correlations:
                    try:
                        parts = result.split(', ')
                        corr_value = float(parts[0].split('Correlazione: ')[1])
                        cov_value = float(parts[1].split('Covarianza: ')[1])
                        uncorr_status = parts[2].split('Incorrelati: ')[1]
                        power_x1_value = float(parts[3].split('Potenza_X1: ')[1])
                        power_x2_value = float(parts[4].split('Potenza_X2: ')[1])

                        correlations_values.append(corr_value)
                        covariances_values.append(cov_value)
                        powers_x1_values.append(power_x1_value)
                        powers_x2_values.append(power_x2_value)

                        if uncorr_status == 'Sì':
                            uncorrelated_count += 1
                    except:
                        pass

                if correlations_values:
                    correlations_array = np.array(correlations_values)
                    covariances_array = np.array(covariances_values)
                    powers_x1_array = np.array(powers_x1_values)
                    powers_x2_array = np.array(powers_x2_values)

                    f.write("#\n")
                    f.write("# STATISTICHE CORRELAZIONI:\n")
                    f.write(f"# Media: {np.mean(correlations_array):.6f}\n")
                    f.write(f"# Mediana: {np.median(correlations_array):.6f}\n")
                    f.write(f"# Deviazione standard: {np.std(correlations_array):.6f}\n")
                    f.write(f"# Massimo: {np.max(correlations_array):.6f}\n")
                    f.write(f"# Minimo: {np.min(correlations_array):.6f}\n")

                    f.write("#\n")
                    f.write("# STATISTICHE COVARIANZE:\n")
                    f.write(f"# Media: {np.mean(covariances_array):.6f}\n")
                    f.write(f"# Deviazione standard: {np.std(covariances_array):.6f}\n")
                    f.write(f"# Massimo: {np.max(covariances_array):.6f}\n")
                    f.write(f"# Minimo: {np.min(covariances_array):.6f}\n")

                    f.write("#\n")
                    f.write("# STATISTICHE POTENZE:\n")
                    f.write(f"# Potenza X1 - Media: {np.mean(powers_x1_array):.6f}, Dev.Std: {np.std(powers_x1_array):.6f}\n")
                    f.write(f"# Potenza X2 - Media: {np.mean(powers_x2_array):.6f}, Dev.Std: {np.std(powers_x2_array):.6f}\n")

                    high_corr = np.sum(correlations_array > 0.8)
                    medium_corr = np.sum((correlations_array > 0.5) & (correlations_array <= 0.8))
                    low_corr = np.sum(correlations_array <= 0.5)
                    near_zero = np.sum(np.abs(correlations_array) < 0.1)

                    f.write("#\n")
                    f.write("# INTERPRETAZIONE TEORICA:\n")
                    f.write(f"# Coppie altamente correlate (>0.8): {high_corr}\n")
                    f.write(f"# Coppie moderatamente correlate (0.5-0.8): {medium_corr}\n")
                    f.write(f"# Coppie debolmente correlate (≤0.5): {low_corr}\n")
                    f.write(f"# Coppie praticamente incorrelate (|r|<0.1): {near_zero}\n")
                    f.write(f"# Coppie statisticamente incorrelate (test covarianza): {uncorrelated_count}\n")

        print(f"\nRisultati salvati in: {output_file}")
        print(f"Totale coppie processate: {len(results)}")

        successful_correlations = [r for r in results if not ('ERRORE' in r)]
        print(f"Correlazioni calcolate con successo: {len(successful_correlations)}")

        if successful_correlations:
            correlations_values = []
            for result in successful_correlations:
                try:
                    corr_value = float(result.split('Correlazione: ')[1].split(',')[0])
                    correlations_values.append(corr_value)
                except:
                    pass

            if correlations_values:
                correlations_array = np.array(correlations_values)
                print(f"\n=== STATISTICHE CORRELAZIONI ===")
                print(f"Correlazione media: {np.mean(correlations_array):.6f}")
                print(f"Correlazione mediana: {np.median(correlations_array):.6f}")
                print(f"Deviazione standard: {np.std(correlations_array):.6f}")
                print(f"Correlazione massima: {np.max(correlations_array):.6f}")
                print(f"Correlazione minima: {np.min(correlations_array):.6f}")

                high_corr = np.sum(correlations_array > 0.8)
                medium_corr = np.sum((correlations_array > 0.5) & (correlations_array <= 0.8))
                low_corr = np.sum(correlations_array <= 0.5)

                print(f"\n=== INTERPRETAZIONE TEORICA ===")
                print(f"Coppie altamente correlate (>0.8): {high_corr}")
                print(f"Coppie moderatamente correlate (0.5-0.8): {medium_corr}")
                print(f"Coppie debolmente correlate (≤0.5): {low_corr}")

                near_zero = np.sum(np.abs(correlations_array) < 0.1)
                print(f"Coppie praticamente incorrelate (|r|<0.1): {near_zero}")

    except Exception as e:
        print(f"Errore nel salvare il file dei risultati: {e}")


# --- BLOCCO MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisi di correlazione audio tra due directory di file WAV"
    )
    parser.add_argument("path_audio1", help="Percorso della prima directory di file audio (.wav)")
    parser.add_argument("path_audio2", help="Percorso della seconda directory di file audio (.wav)")
    parser.add_argument("output_file", help="Percorso del file di output dei risultati")

    args = parser.parse_args()

    print("=== Analisi di Correlazione Audio - Implementazione Teorica ===")
    print(" Directory input 1:", args.path_audio1)
    print(" Directory input 2:", args.path_audio2)
    print(" File output:", args.output_file)
    print("=" * 70)

    if not os.path.exists(args.path_audio1):
        print(f"Errore: Directory '{args.path_audio1}' non trovata.")
        exit(1)
    if not os.path.exists(args.path_audio2):
        print(f"Errore: Directory '{args.path_audio2}' non trovata.")
        exit(1)

    process_all_pairs(args.path_audio1, args.path_audio2, args.output_file)

    print("\n=== Esecuzione completata ===")
    print("I risultati includono:")
    print("- Coefficiente di correlazione di Pearson")
    print("- Analisi di covarianza (momento misto centrato)")
    print("- Test di incorrelazione statistica")
    print("- Calcolo delle potenze dei segnali")
    print("- Statistiche descrittive complete")
    print("- Tutte le informazioni sono salvate nel file di output")
