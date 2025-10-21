"""
Oracle Mask Computation and Evaluation for Source Separation

This module implements oracle masks (IBM, IRM, Wiener) and evaluation metrics
for establishing performance upper bounds in source separation tasks.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Optional


# STFT Configuration (optimized for 16kHz audio)
WINDOW_SIZE = 1024  # 64ms @ 16kHz
HOP_LENGTH = 256    # 75% overlap
FFT_SIZE = 1024     # 513 frequency bins up to 8kHz
WINDOW = 'hann'     # Hann window for smooth reconstruction


def compute_stft(audio: np.ndarray, sr: int, 
                 window_size: int = WINDOW_SIZE,
                 hop_length: int = HOP_LENGTH,
                 fft_size: int = FFT_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) of audio signal.
    
    Args:
        audio: Audio signal (1D array)
        sr: Sample rate
        window_size: Window size in samples
        hop_length: Hop length in samples
        fft_size: FFT size
        
    Returns:
        f: Frequency array
        t: Time array
        Zxx: Complex STFT matrix (freq x time)
    """
    f, t, Zxx = signal.stft(
        audio,
        fs=sr,
        window=WINDOW,
        nperseg=window_size,
        noverlap=window_size - hop_length,
        nfft=fft_size
    )
    return f, t, Zxx


def compute_istft(Zxx: np.ndarray, sr: int,
                  window_size: int = WINDOW_SIZE,
                  hop_length: int = HOP_LENGTH,
                  fft_size: int = FFT_SIZE) -> np.ndarray:
    """
    Compute inverse STFT to reconstruct audio signal.
    
    Args:
        Zxx: Complex STFT matrix
        sr: Sample rate
        window_size: Window size in samples
        hop_length: Hop length in samples
        fft_size: FFT size
        
    Returns:
        audio: Reconstructed audio signal
    """
    _, audio = signal.istft(
        Zxx,
        fs=sr,
        window=WINDOW,
        nperseg=window_size,
        noverlap=window_size - hop_length,
        nfft=fft_size
    )
    return audio


def ideal_binary_mask(stem_stfts: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute Ideal Binary Mask (IBM).
    
    For each time-frequency bin, assigns 1 to the stem with largest magnitude,
    0 to all others. This represents the most aggressive separation strategy.
    
    Args:
        stem_stfts: List of complex STFT arrays, one per stem
        
    Returns:
        List of binary masks (0 or 1) for each stem
    """
    magnitudes = np.array([np.abs(stft) for stft in stem_stfts])
    max_indices = np.argmax(magnitudes, axis=0)
    
    masks = []
    for i in range(len(stem_stfts)):
        mask = (max_indices == i).astype(float)
        masks.append(mask)
    
    return masks


def ideal_ratio_mask(stem_stfts: List[np.ndarray], 
                     epsilon: float = 1e-10) -> List[np.ndarray]:
    """
    Compute Ideal Ratio Mask (IRM).
    
    IRM_i = |S_i| / sum_j(|S_j|)
    
    Soft mask based on magnitude ratios. Sums to 1 across stems at each T-F bin.
    
    Args:
        stem_stfts: List of complex STFT arrays
        epsilon: Small constant to avoid division by zero
        
    Returns:
        List of soft masks (0 to 1) for each stem
    """
    magnitudes = np.array([np.abs(stft) for stft in stem_stfts])
    total_magnitude = np.sum(magnitudes, axis=0) + epsilon
    
    masks = []
    for mag in magnitudes:
        mask = mag / total_magnitude
        masks.append(mask)
    
    return masks


def wiener_mask(stem_stfts: List[np.ndarray],
                power: float = 2,
                epsilon: float = 1e-10) -> List[np.ndarray]:
    """
    Compute Wiener-style mask (generalized power mask).
    
    Mask_i = |S_i|^p / sum_j(|S_j|^p)
    
    Power determines the mask's behavior:
    - p=1: Same as IRM (magnitude-based)
    - p=2: Traditional Wiener filter (power-based)
    - Higher p: More aggressive separation
    
    Args:
        stem_stfts: List of complex STFT arrays
        power: Power exponent (typically 1 or 2)
        epsilon: Small constant to avoid division by zero
        
    Returns:
        List of soft masks for each stem
    """
    magnitudes = np.array([np.abs(stft) for stft in stem_stfts])
    powers = magnitudes ** power
    total_power = np.sum(powers, axis=0) + epsilon
    
    masks = []
    for mag_power in powers:
        mask = mag_power / total_power
        masks.append(mask)
    
    return masks


def apply_masks_and_reconstruct(mix_stft: np.ndarray,
                                masks: List[np.ndarray],
                                sr: int) -> List[np.ndarray]:
    """
    Apply masks to mixture STFT and reconstruct audio signals.
    
    Args:
        mix_stft: Complex STFT of mixture
        masks: List of masks to apply
        sr: Sample rate
        
    Returns:
        List of reconstructed audio signals
    """
    reconstructed = []
    for mask in masks:
        masked_stft = mix_stft * mask
        audio = compute_istft(masked_stft, sr)
        reconstructed.append(audio)
    
    return reconstructed


def si_sdr(reference: np.ndarray, 
           estimate: np.ndarray,
           epsilon: float = 1e-10) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    SI-SDR is more robust than SDR to gain differences between reference
    and estimate. Higher values indicate better separation quality.
    
    Args:
        reference: Ground truth signal
        estimate: Estimated signal
        epsilon: Small constant for numerical stability
        
    Returns:
        SI-SDR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    
    # Scale-invariant projection
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + epsilon)
    target = alpha * reference
    
    # Distortion
    residual = estimate - target
    
    # SI-SDR in dB
    si_sdr_value = 10 * np.log10(
        (np.sum(target**2) + epsilon) / (np.sum(residual**2) + epsilon)
    )
    
    return si_sdr_value


def evaluate_separation(reference_stems: List[np.ndarray],
                       estimated_stems: List[np.ndarray],
                       stem_names: List[str]) -> Dict[str, float]:
    """
    Evaluate separation quality using SI-SDR.
    
    Args:
        reference_stems: List of ground truth stem signals
        estimated_stems: List of estimated stem signals
        stem_names: List of stem names for reporting
        
    Returns:
        Dictionary mapping stem names to SI-SDR values, plus 'mean'
    """
    results = {}
    
    for i, name in enumerate(stem_names):
        sdr = si_sdr(reference_stems[i], estimated_stems[i])
        results[name] = sdr
    
    # Compute mean SI-SDR
    results['mean'] = np.mean(list(results.values()))
    
    return results


def print_results(results: Dict[str, float], method_name: str, width: int = 40):
    """
    Pretty-print evaluation results.
    
    Args:
        results: Dictionary from evaluate_separation()
        method_name: Name of the method being evaluated
        width: Column width for formatting
    """
    print(f"\n{method_name}:")
    print("-" * width)
    
    # Print individual stem results
    for key, value in results.items():
        if key != 'mean':
            print(f"  {key:30s}: {value:6.2f} dB")
    
    # Print mean
    print(f"  {'Mean SI-SDR':30s}: {results['mean']:6.2f} dB")


def compare_oracle_methods(stem_stfts: List[np.ndarray],
                          reference_stems: List[np.ndarray],
                          stem_names: List[str],
                          mix_stft: np.ndarray,
                          sr: int) -> Dict[str, Dict[str, float]]:
    """
    Compute and compare all oracle mask methods.
    
    Args:
        stem_stfts: List of ground-truth stem STFTs
        reference_stems: List of ground-truth stem audio signals
        stem_names: Names of stems
        mix_stft: Mixture STFT
        sr: Sample rate
        
    Returns:
        Dictionary mapping method names to their evaluation results
    """
    print("=" * 70)
    print("ORACLE MASK PERFORMANCE (Upper Bounds)")
    print("=" * 70)
    
    all_results = {}
    
    # IBM
    print("\nComputing Ideal Binary Masks...")
    ibm_masks = ideal_binary_mask(stem_stfts)
    ibm_reconstructed = apply_masks_and_reconstruct(mix_stft, ibm_masks, sr)
    ibm_results = evaluate_separation(reference_stems, ibm_reconstructed, stem_names)
    print_results(ibm_results, "Ideal Binary Mask (IBM)")
    all_results['IBM'] = ibm_results
    
    # IRM
    print("\nComputing Ideal Ratio Masks...")
    irm_masks = ideal_ratio_mask(stem_stfts)
    irm_reconstructed = apply_masks_and_reconstruct(mix_stft, irm_masks, sr)
    irm_results = evaluate_separation(reference_stems, irm_reconstructed, stem_names)
    print_results(irm_results, "Ideal Ratio Mask (IRM)")
    all_results['IRM'] = irm_results
    
    # Wiener p=2
    print("\nComputing Wiener Masks (p=2)...")
    wiener2_masks = wiener_mask(stem_stfts, power=2)
    wiener2_reconstructed = apply_masks_and_reconstruct(mix_stft, wiener2_masks, sr)
    wiener2_results = evaluate_separation(reference_stems, wiener2_reconstructed, stem_names)
    print_results(wiener2_results, "Wiener Mask (p=2)")
    all_results['Wiener_p2'] = wiener2_results
    
    # Wiener p=1
    print("\nComputing Wiener Masks (p=1)...")
    wiener1_masks = wiener_mask(stem_stfts, power=1)
    wiener1_reconstructed = apply_masks_and_reconstruct(mix_stft, wiener1_masks, sr)
    wiener1_results = evaluate_separation(reference_stems, wiener1_reconstructed, stem_names)
    print_results(wiener1_results, "Wiener Mask (p=1)")
    all_results['Wiener_p1'] = wiener1_results
    
    print("\n" + "=" * 70)
    
    return all_results


def get_stft_params():
    """Return the STFT parameters as a dictionary."""
    return {
        'window_size': WINDOW_SIZE,
        'hop_length': HOP_LENGTH,
        'fft_size': FFT_SIZE,
        'window': WINDOW
    }


def print_stft_config(sr: int):
    """Print STFT configuration details."""
    print("STFT Configuration:")
    print(f"  Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE/sr*1000:.1f} ms)")
    print(f"  Hop length: {HOP_LENGTH} samples ({HOP_LENGTH/sr*1000:.1f} ms)")
    print(f"  FFT size: {FFT_SIZE}")
    print(f"  Frequency bins: {FFT_SIZE//2 + 1} (0 to {sr/2:.0f} Hz)")
    print(f"  Overlap: {(1 - HOP_LENGTH/WINDOW_SIZE)*100:.0f}%")
