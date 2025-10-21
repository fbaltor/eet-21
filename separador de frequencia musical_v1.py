import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter, firwin
import os

# -------- utilidades DSP --------


def butter_filter(data, cutoff, fs, btype='low', order=6):
    nyq = 0.5 * fs
    wn = np.array(cutoff)/nyq if isinstance(cutoff,
                                            (list, tuple, np.ndarray)) else cutoff/nyq
    b, a = butter(order, wn, btype=btype, analog=False)
    return lfilter(b, a, data)


def fir_filter(data, cutoff, fs, btype='low', numtaps=1025):
    nyq = 0.5 * fs
    if btype == 'low':
        taps = firwin(numtaps, cutoff/nyq, pass_zero=True, window="hamming")
    elif btype == 'high':
        taps = firwin(numtaps, cutoff/nyq, pass_zero=False, window="hamming")
    elif btype == 'band':
        taps = firwin(numtaps, [c/nyq for c in cutoff],
                      pass_zero=False, window="hamming")
    return lfilter(taps, 1.0, data)


def to_float32(x):
    x = x.astype(np.float32)
    m = np.max(np.abs(x))
    return x if m < 1e-8 else (x / m)


def save16(path, y, sr):
    y = to_float32(y)
    sf.write(path, y, sr, subtype="PCM_16")


# -------- entrada --------
infile = "Michael Jackson - Billie Jean.wav"
if not os.path.exists(infile):
    sr = 44100
    t = np.linspace(0, 3, 3*sr, endpoint=False)
    y = (0.6*np.sin(2*np.pi*110*t) + 0.4 *
         np.sin(2*np.pi*440*t) + 0.2*np.sin(2*np.pi*8000*t))
    save16("Michael Jackson - Billie Jean.wav", y, sr)
    infile = "Michael Jackson - Billie Jean.wav"

# Carrega mono
y, sr = librosa.load(infile, sr=None, mono=True)

# -------- HPSS melhorado --------
y_harm, y_perc = librosa.effects.hpss(y, margin=(5, 1), power=2.0)

# -------- separação de “stems” --------
use_fir = True
filt = fir_filter if use_fir else butter_filter

# Baixo: ≤120 Hz
bass = filt(y_harm, 120, sr, btype='low')

# Voz: 300–3400 Hz
voice = filt(y_harm, [300, 3400], sr, btype='band')

# Bateria bruta: percussivo
drums = y_perc

# Bateria limpa: percussivo menos voz
drums_clean = drums - voice
drums_clean = to_float32(drums_clean)

# Pratos: ≥7000 Hz
cymbals = filt(y_perc, 7000, sr, btype='high')

# Reconstrução
recon = bass + voice + drums_clean

# -------- salva arquivos --------
save16("stem_bass.wav", bass, sr)
save16("stem_voice.wav", voice, sr)
save16("stem_drums.wav", drums, sr)
save16("stem_drums_clean.wav", drums_clean, sr)
save16("stem_cymbals.wav", cymbals, sr)
save16("stem_recon.wav", recon, sr)

print("Gerado: stem_bass.wav, stem_voice.wav, stem_drums.wav, stem_drums_clean.wav, stem_cymbals.wav, stem_recon.wav")
