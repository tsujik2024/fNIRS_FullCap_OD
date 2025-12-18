"""
fNIRS_FullCap_2025 - A complete processing pipeline for full-head fNIRS data
"""

from .processing.pipeline_manager import PipelineManager

__version__ = "1.0.0"
__all__ = ['PipelineManager']

# Package-level imports that should be available when users do:
# `import fnirs_FullCap_2025`
from .read.loaders import read_txt_file
from .viz.visualizer import FNIRSVisualizer

# Configure package-level logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())