"""
Slakh Dataset Utilities

Utility classes for working with the Slakh audio source separation dataset.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class StemInfo:
    """Information about a single instrument stem."""
    stem_id: str
    inst_class: str
    midi_program_name: str
    program_num: int
    is_drum: bool
    integrated_loudness: Optional[float]
    plugin_name: str
    audio_rendered: bool
    midi_saved: bool


class SlakhTrack:
    """
    A simple tool to work with Slakh dataset tracks.
    
    Usage:
        track = SlakhTrack('/path/to/track/directory')
        info = track.get_track_info()
        drums = track.get_stems_by_class('Drums')
        stem = track.get_stem('S00')
    """
    
    def __init__(self, track_dir: Union[str, Path]):
        """
        Initialize with the path to a track directory.
        
        Args:
            track_dir: Path to the track directory containing metadata.yaml
        """
        self.track_dir = Path(track_dir)
        self.metadata_path = self.track_dir / 'metadata.yaml'
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.yaml not found in {track_dir}")
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = yaml.safe_load(f)
    
    def get_track_info(self) -> Dict:
        """Get general track information."""
        return {
            'uuid': self.metadata.get('UUID'),
            'normalized': self.metadata.get('normalized'),
            'overall_gain': self.metadata.get('overall_gain'),
            'normalization_factor': self.metadata.get('normalization_factor'),
            'target_peak': self.metadata.get('target_peak'),
            'num_stems': len(self.metadata.get('stems', {}))
        }
    
    def get_stem_info(self, stem_id: str) -> Optional[StemInfo]:
        """
        Get information about a specific stem.
        
        Args:
            stem_id: Stem identifier (e.g., 'S00', 'S01')
            
        Returns:
            StemInfo object or None if stem doesn't exist
        """
        stems = self.metadata.get('stems', {})
        if stem_id not in stems:
            return None
        
        stem_data = stems[stem_id]
        return StemInfo(
            stem_id=stem_id,
            inst_class=stem_data.get('inst_class'),
            midi_program_name=stem_data.get('midi_program_name'),
            program_num=stem_data.get('program_num'),
            is_drum=stem_data.get('is_drum', False),
            integrated_loudness=stem_data.get('integrated_loudness'),
            plugin_name=stem_data.get('plugin_name'),
            audio_rendered=stem_data.get('audio_rendered', False),
            midi_saved=stem_data.get('midi_saved', False)
        )
    
    def get_all_stems(self) -> List[StemInfo]:
        """Get information about all stems in the track."""
        stems = self.metadata.get('stems', {})
        result = []
        for stem_id in stems.keys():
            stem_info = self.get_stem_info(stem_id)
            if stem_info is not None:
                result.append(stem_info)
        return result
    
    def get_stems_by_class(self, inst_class: str) -> List[StemInfo]:
        """
        Get all stems of a specific instrument class.
        
        Args:
            inst_class: Instrument class (e.g., 'Guitar', 'Drums', 'Piano')
            
        Returns:
            List of StemInfo objects
        """
        return [
            stem for stem in self.get_all_stems()
            if stem.inst_class == inst_class
        ]
    
    def get_drum_stems(self) -> List[StemInfo]:
        """Get all drum stems."""
        return [stem for stem in self.get_all_stems() if stem.is_drum]
    
    def get_stem_paths(self, stem_id: str) -> Dict[str, Path]:
        """
        Get file paths for a specific stem.
        
        Args:
            stem_id: Stem identifier (e.g., 'S00')
            
        Returns:
            Dictionary with 'audio' and 'midi' paths
        """
        return {
            'audio': self.track_dir / 'stems' / f'{stem_id}.wav',
            'midi': self.track_dir / 'MIDI' / f'{stem_id}.mid'
        }
    
    def get_mix_path(self) -> Path:
        """Get path to the mix.wav file."""
        return self.track_dir / 'mix.wav'
    
    def list_available_instruments(self) -> List[str]:
        """Get a list of all unique instrument classes in the track."""
        return list(set(stem.inst_class for stem in self.get_all_stems()))
    
    def __repr__(self) -> str:
        info = self.get_track_info()
        return f"SlakhTrack(uuid='{info['uuid']}', stems={info['num_stems']})"
