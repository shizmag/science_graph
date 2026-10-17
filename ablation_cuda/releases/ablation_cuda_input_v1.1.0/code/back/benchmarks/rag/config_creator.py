#!/usr/bin/env python3
"""
Science Graph — Configuration Creator and Preset Manager.
Provides shared utilities, presets, and monkey-patching logic for custom runs.
Facade module that delegates to core.config.
"""

import sys
from pathlib import Path

# Set up python path to resolve src and core imports correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import (
    DEFAULT_COMPONENTS,
    DEFAULT_HYPERPARAMS,
    CUSTOM_PRESET_COMPONENTS,
    CUSTOM_PRESET_HYPERPARAMS_NT,
    CUSTOM_PRESET_HYPERPARAMS,
    RAGPreset,
    GraphPreset,
    BM25Preset,
    CustomPresetHyperparams,
    get_custom_preset_weights,
    add_custom_config_arguments,
    build_custom_config,
    patch_config_for_custom,
    restore_baseline_config_patch,
)

__all__ = [
    "DEFAULT_COMPONENTS",
    "DEFAULT_HYPERPARAMS",
    "CUSTOM_PRESET_COMPONENTS",
    "CUSTOM_PRESET_HYPERPARAMS_NT",
    "CUSTOM_PRESET_HYPERPARAMS",
    "RAGPreset",
    "GraphPreset",
    "BM25Preset",
    "CustomPresetHyperparams",
    "get_custom_preset_weights",
    "add_custom_config_arguments",
    "build_custom_config",
    "patch_config_for_custom",
    "restore_baseline_config_patch",
]
