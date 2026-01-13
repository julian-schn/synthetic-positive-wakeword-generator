"""Manifest generation for provenance tracking (FR-5)."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ManifestEntry:
    """A single manifest entry for a generated sample."""
    
    filename: str
    seed: int
    wakeword: str
    voice_config: dict[str, Any]
    speed: float
    augmentations: dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ManifestWriter:
    """Writes manifest entries to JSONL file."""
    
    def __init__(self, manifest_path: Path):
        """Initialize manifest writer.
        
        Args:
            manifest_path: Path to manifest.jsonl file
        """
        self.manifest_path = manifest_path
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def write_entry(self, entry: ManifestEntry) -> None:
        """Append a single entry to the manifest."""
        with open(self.manifest_path, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
    
    def write_entries(self, entries: list[ManifestEntry]) -> None:
        """Write multiple entries to the manifest."""
        with open(self.manifest_path, 'a') as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict()) + '\n')
