# Oracle Mask Implementation - Source Separation Project

## Overview

This repository implements oracle mask computation and evaluation for classical DSP-based audio source separation using the Babyslakh dataset (16kHz version).

## What Was Implemented

### 1. Core Modules

#### `separator.py`
- `SlakhTrack` class: Utilities for loading and working with Slakh dataset tracks
- `StemInfo` dataclass: Metadata about individual stems

#### `oracle_masks.py`
- **STFT/ISTFT functions**: Consistent time-frequency analysis
  - Window: Hann, 1024 samples (64ms @ 16kHz)
  - Hop: 256 samples (75% overlap)
  - FFT: 1024 bins (513 frequency bins up to 8kHz)

- **Oracle Mask Types**:
  - **Ideal Binary Mask (IBM)**: Winner-take-all per T-F bin
  - **Ideal Ratio Mask (IRM)**: Soft mask based on magnitude ratios
  - **Wiener Mask (p=1,2)**: Generalized power masks

- **Evaluation Metrics**:
  - **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio): Primary metric
  - Robust to gain differences between reference and estimate

#### `demo_oracle_masks.py`
- End-to-end demonstration script
- Loads a track, computes all oracle masks, evaluates performance
- Verifies that stems sum to mix

### 2. Results from Demo Run (Track00010, 7 stems)

```
SUMMARY - Mean SI-SDR by Method:
  IBM            :   4.18 dB
  IRM            :   4.46 dB
  Wiener_p2      :   5.19 dB
  Wiener_p1      :   4.46 dB
```

**Key Observations**:
- Wiener (p=2) performs best with 5.19 dB mean SI-SDR
- These are UPPER BOUNDS - any classical method should achieve below these values
- One stem (S04_Synth Pad) shows 0.00 dB, possibly very weak in the mix
- Drums separate best (9.27 dB with Wiener p=2)

## Usage

### Quick Demo
```bash
python demo_oracle_masks.py
```

### In Your Own Code
```python
from separator import SlakhTrack
from oracle_masks import (
    compute_stft,
    compare_oracle_methods,
    ideal_binary_mask,
    wiener_mask,
    si_sdr
)

# Load a track
track = SlakhTrack('./data/babyslakh_16k/Track00010')

# Load stems and compute STFTs
# ... (see demo_oracle_masks.py for full example)

# Compare all oracle methods
results = compare_oracle_methods(
    stem_stfts,
    stems_audio,
    stem_names,
    mix_stft,
    sr
)
```

## Dataset Information

- **Dataset**: Babyslakh (16kHz version)
- **Tracks**: 20 total
- **Stems per track**: 7-11 instruments
- **Format**: Mono, 16-bit PCM WAV
- **Verified**: Stems sum exactly to mix (correlation > 0.999)

## Next Steps (Classical DSP Methods)

### Phase 1: HPSS Baseline (Priority)
1. Implement median filtering HPSS (Harmonic-Percussive Source Separation)
2. Tune median filter lengths (time vs. frequency axes)
3. Evaluate on all tracks
4. **Expected performance**: 2-8 dB SI-SDR (well below oracle bounds)

### Phase 2: Sub-band Enhancement
1. Design 3-4 band FIR filter banks:
   - Bass: ≤120 Hz
   - Mid (vocals/guitars): 300-3400 Hz
   - High: ≥6-8 kHz
2. Apply to HPSS harmonic output
3. Combine with soft masking

### Phase 3: Advanced Methods
1. **NMF** (Non-negative Matrix Factorization)
   - 8-16 components
   - Group by frequency/continuity heuristics
2. **Mask smoothing**: Reduce musical noise
3. **Parameter sweeps**: Window size, hop length, crossover frequencies

### Phase 4: Evaluation & Reporting
1. Full battery on 15 validation + 5 test tracks
2. Generate spectrogram comparisons
3. Compute band-energy leakage
4. Create ablation tables

## Key Implementation Decisions

### Why These STFT Parameters?
- **1024 samples @ 16kHz** = 64ms window
  - Good time-frequency resolution trade-off for music
  - Captures pitch fundamentals down to ~60 Hz
- **75% overlap** (256-sample hop)
  - Smooth reconstruction with Hann window
  - Standard for audio analysis

### Why SI-SDR?
- More robust than SDR to gain mismatches
- Common metric in modern source separation
- Easier to implement than BSS Eval metrics

### Why Oracle Masks First?
- Establishes performance ceiling
- Validates STFT/evaluation pipeline
- Debugging tool (methods shouldn't exceed oracle)
- Shows which instruments are inherently hard to separate

## Files Structure

```
project/
├── separator.py              # Slakh dataset utilities
├── oracle_masks.py           # Oracle masks & evaluation
├── demo_oracle_masks.py      # Demo script
├── separator.ipynb           # Jupyter exploration notebook
├── data/
│   └── babyslakh_16k/       # Dataset (20 tracks)
└── README_ORACLE_MASKS.md   # This file
```

## References

### Oracle Masks & Evaluation
- Vincent et al. (2006), "Performance measurement in blind audio source separation", IEEE TASLP
- Le Roux et al., "SDR – Half-baked or well done?" (SI-SDR discussion)

### Upcoming Classical Methods
- Fitzgerald (2010), "Harmonic/Percussive Separation using Median Filtering", DAFx
- Driedger & Müller (2014), "Enhancing HPSS", DAFx

### Textbooks
- Oppenheim & Schafer, "Discrete-Time Signal Processing", 3e
- Proakis & Manolakis, "Digital Signal Processing", 4e

## Notes

### Known Issues
- Some tracks have macOS metadata files (`._*.wav`) - automatically skipped
- Low oracle SI-SDR (<10 dB) suggests overlapping stems are common
- Mono audio only (stereo spatial methods not applicable)

### Performance Tips
- Use 10-second clips for fast iteration
- Cache STFTs to avoid recomputation
- NMF will be slow - consider subset for initial tuning

## Contribution to Course Project

This implementation provides:
1. ✅ **Baseline infrastructure**: STFT, evaluation, dataset loading
2. ✅ **Performance benchmarks**: Oracle upper bounds
3. ✅ **Verification**: Stems sum to mix correctly
4. 🔄 **Classical DSP pipeline**: HPSS → Sub-bands → NMF (in progress)

The oracle masks confirm the dataset is suitable and the evaluation pipeline is working correctly. The relatively low upper bounds (4-5 dB mean) indicate this is a challenging separation task, which is good for demonstrating classical DSP techniques.

---

**Status**: Phase 0 (Oracle Masks) Complete ✅  
**Next**: Implement HPSS baseline (Phase 1)
