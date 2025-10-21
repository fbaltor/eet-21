"""
Demo: Oracle Mask Performance on Babyslakh Dataset

This script demonstrates how to compute oracle mask upper bounds
for source separation on the Babyslakh dataset.
"""

import soundfile as sf
import numpy as np
from pathlib import Path
import sys

# Import our modules
from separator import SlakhTrack
from oracle_masks import (
    compute_stft, 
    compare_oracle_methods,
    print_stft_config
)


def load_track_stems(track: SlakhTrack, trim_samples: int = None):
    """
    Load all stems from a track and compute their STFTs.
    
    Args:
        track: SlakhTrack object
        trim_samples: Optional number of samples to trim to
        
    Returns:
        stem_names: List of stem names
        stems_audio: List of audio arrays
        stems_stft: List of STFT arrays
        sr: Sample rate
    """
    stem_names = []
    stems_audio = []
    stems_stft = []
    sr = None
    
    for stem_info in track.get_all_stems():
        stem_path = track.get_stem_paths(stem_info.stem_id)['audio']
        
        # Skip macOS metadata files
        if stem_path.name.startswith('._'):
            continue
            
        try:
            audio, sr = sf.read(stem_path)
            
            # Trim if requested
            if trim_samples is not None:
                audio = audio[:trim_samples]
            
            # Compute STFT
            f, t, Zxx = compute_stft(audio, sr)
            
            # Create descriptive name
            stem_name = f"{stem_info.stem_id}_{stem_info.inst_class}"
            
            stem_names.append(stem_name)
            stems_audio.append(audio)
            stems_stft.append(Zxx)
            
            print(f"  Loaded {stem_name}")
            
        except Exception as e:
            print(f"  Warning: Could not load {stem_path.name}: {e}")
            continue
    
    return stem_names, stems_audio, stems_stft, sr


def main():
    # Configuration
    TRIM_DURATION = 10  # seconds
    DATASET_ROOT = Path("./data/babyslakh_16k")
    
    print("=" * 70)
    print("ORACLE MASK DEMO - Babyslakh Source Separation")
    print("=" * 70)
    
    # Find all tracks
    all_tracks = sorted([d for d in DATASET_ROOT.iterdir() 
                        if d.is_dir() and d.name.startswith('Track')])
    
    print(f"\nFound {len(all_tracks)} tracks in dataset")
    
    # Sort by stem count and pick one with few stems for demo
    track_stem_counts = []
    for track_dir in all_tracks:
        try:
            t = SlakhTrack(track_dir)
            num_stems = t.get_track_info()['num_stems']
            track_stem_counts.append((track_dir, num_stems))
        except:
            continue
    
    track_stem_counts.sort(key=lambda x: x[1])
    
    # Use track with ~7 stems (good for demo)
    demo_track_dir = track_stem_counts[1][0] if len(track_stem_counts) > 1 else track_stem_counts[0][0]
    demo_track = SlakhTrack(demo_track_dir)
    
    print(f"\nSelected track: {demo_track_dir.name}")
    print(f"Number of stems: {demo_track.get_track_info()['num_stems']}")
    print(f"\nInstruments:")
    for stem in demo_track.get_all_stems():
        print(f"  - {stem.stem_id}: {stem.inst_class} ({stem.midi_program_name})")
    
    # Load mix
    print(f"\nLoading mix...")
    mix_audio, sr = sf.read(demo_track.get_mix_path())
    trim_samples = int(sr * TRIM_DURATION)
    mix_audio = mix_audio[:trim_samples]
    
    print(f"Mix duration: {len(mix_audio) / sr:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    
    # Print STFT configuration
    print()
    print_stft_config(sr)
    
    # Compute mix STFT
    print(f"\nComputing mix STFT...")
    f, t, mix_stft = compute_stft(mix_audio, sr)
    print(f"STFT shape: {mix_stft.shape} (freq x time)")
    
    # Load all stems
    print(f"\nLoading stems...")
    stem_names, stems_audio, stems_stft, _ = load_track_stems(demo_track, trim_samples)
    
    print(f"\nLoaded {len(stem_names)} stems successfully")
    
    # Verify stems sum to mix
    reconstructed_mix = np.sum(stems_audio, axis=0)
    correlation = np.corrcoef(mix_audio.flatten(), reconstructed_mix.flatten())[0, 1]
    print(f"\nVerification: Correlation between mix and sum of stems = {correlation:.6f}")
    
    if correlation < 0.99:
        print("WARNING: Low correlation detected! Stems may not sum to mix correctly.")
    
    # Compute oracle masks and evaluate
    print()
    results = compare_oracle_methods(
        stems_stft,
        stems_audio,
        stem_names,
        mix_stft,
        sr
    )
    
    # Summary comparison
    print("\nSUMMARY - Mean SI-SDR by Method:")
    print("-" * 40)
    for method, res in results.items():
        print(f"  {method:15s}: {res['mean']:6.2f} dB")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("\nInterpretation:")
    print("- These are UPPER BOUNDS (oracle masks use ground truth)")
    print("- Any classical method should achieve SI-SDR below these values")
    print("- Higher SI-SDR = better separation quality")
    print("- Typical range: 5-30 dB for oracle masks")
    print("=" * 70)


if __name__ == "__main__":
    main()
