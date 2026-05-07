"""
Tacit Video Annotator — hierarchical physical-state labeling for laboratory procedure videos.

Public API:
    extract.extract_dense_frames(video, out_dir, interval=30)
    extract.extract_triage_frames(video, out_dir, n=10)
    label.label_frame(image_path, branch, client) -> Annotation
    label.label_video(frames_dir, branch, client) -> list[Annotation]
    bundle.build_bundle(annotations_path, output_root) -> Path
    append.append_to_master(annotations_path, master_xlsx) -> AppendStats
    triage.triage_video(video, branch, client) -> TriageDecision
"""

from .__version__ import __version__
from .schema import Annotation, AuditEntry, TriageDecision

__all__ = ["__version__", "Annotation", "AuditEntry", "TriageDecision"]
