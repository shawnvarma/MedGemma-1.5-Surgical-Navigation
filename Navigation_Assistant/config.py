"""Configuration constants and utility functions.

All paths are configurable via environment variables so the same codebase
runs locally (with data in data/) and on HuggingFace Spaces.
"""

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

IS_SPACES = os.environ.get("SPACE_ID") is not None
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

if IS_SPACES:
    SCAN_PATH = BASE_DIR / "BraTS20_Training_001_flair.nii"
    SEG_PATH = BASE_DIR / "BraTS20_Training_001_seg.nii"
else:
    DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
    SCAN_PATH = Path(os.environ.get(
        "SCAN_PATH", str(DATA_DIR / "BraTS20_Training_001_flair.nii")))
    SEG_PATH = Path(os.environ.get(
        "SEG_PATH", str(DATA_DIR / "BraTS20_Training_001_seg.nii")))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-1.5-4b-it")

_DEFAULT_LORA = BASE_DIR / "checkpoints" / "tissue_lora"
_DEFAULT_REASONING_LORA = BASE_DIR / "checkpoints" / "reasoning_lora"

LORA_PATH = Path(os.environ["LORA_PATH"]) if os.environ.get("LORA_PATH") \
    else (_DEFAULT_LORA if _DEFAULT_LORA.exists() else None)
REASONING_LORA_PATH = Path(os.environ["REASONING_LORA_PATH"]) \
    if os.environ.get("REASONING_LORA_PATH") \
    else (_DEFAULT_REASONING_LORA if _DEFAULT_REASONING_LORA.exists() else None)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

MAX_TRAJECTORY_POINTS = 5
DISPLAY_SIZE = 560  # ~2.3x native (240 -> 560)
SCALE = DISPLAY_SIZE / 240

# ---------------------------------------------------------------------------
# BraTS segmentation labels (0=Background, 1=NCR/NET, 2=Edema, 4=Enhancing)
# ---------------------------------------------------------------------------

TISSUE_LABELS = {
    0: {"name": "BACKGROUND", "color": (128, 128, 128)},
    1: {"name": "NCR_NET",    "color": (255, 0, 0)},
    2: {"name": "EDEMA",      "color": (0, 255, 0)},
    4: {"name": "ENHANCING",  "color": (255, 255, 0)},
}

TISSUE_NAME_MAP = {0: "BACKGROUND", 1: "NCR_NET", 2: "EDEMA", 4: "ENHANCING"}

TISSUE_DISPLAY_NAMES = {
    "BACKGROUND": "Healthy Tissue",
    "NCR_NET": "Necrotic Tissue",
    "EDEMA": "Edema",
    "ENHANCING": "Enhancing Tumor",
}


def get_tissue_display_name(tissue: str) -> str:
    """Get user-friendly tissue name for display."""
    return TISSUE_DISPLAY_NAMES.get(tissue, tissue)


# ---------------------------------------------------------------------------
# Resection guidance (scaffold-enforced, not model-generated)
# ---------------------------------------------------------------------------

RESECTION_GUIDANCE = {
    "BACKGROUND": {
        "class": "PRESERVE",
        "guidance": "Do not resect - functional brain tissue",
        "color": "red",
    },
    "NCR_NET": {
        "class": "RESECT",
        "guidance": "Safe to resect - necrotic/non-enhancing core",
        "color": "green",
    },
    "EDEMA": {
        "class": "CAUTION",
        "guidance": "Caution - functional tissue infiltrated by swelling",
        "color": "red",
    },
    "ENHANCING": {
        "class": "TARGET",
        "guidance": "Primary target - active enhancing tumor",
        "color": "green",
    },
}

# ---------------------------------------------------------------------------
# Eloquent cortex — damage produces specific, predictable deficits
# ---------------------------------------------------------------------------

ELOQUENT_REGIONS = {
    # Cortical - motor / sensory
    "Precentral Gyrus": "primary motor cortex - movement control",
    "Postcentral Gyrus": "primary somatosensory cortex",
    # Cortical - language (typically left hemisphere)
    "Inferior Frontal Gyrus, pars opercularis":
        "Broca's area - speech production",
    "Inferior Frontal Gyrus, pars triangularis":
        "Broca's area - speech production",
    "Superior Temporal Gyrus, posterior division":
        "Wernicke's area - speech comprehension",
    # Cortical - vision
    "Occipital Pole": "primary visual cortex",
    "Intracalcarine Cortex": "primary visual cortex",
    # Subcortical
    "Hippocampus": "memory formation",
    "Thalamus": "sensory relay - critical structure",
    "Brain-Stem": "vital functions - DO NOT RESECT",
}

# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------


def clean_model_response(text: str) -> str:
    """Remove internal thinking tokens and reasoning preamble.

    MedGemma sometimes emits ``thought`` tokens and reasoning steps
    that should not be shown to users.
    """
    if not text:
        return text

    text = re.sub(r'^thought\s*\n*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Identify the core[^:]*:\s*', '', text,
                  flags=re.IGNORECASE)
    text = re.sub(r'^Recall[^:]*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Answer with exactly one word[^.]*\.?\s*', '', text,
                  flags=re.IGNORECASE)
    text = re.sub(r'Answer with one word[^.]*\.?\s*', '', text,
                  flags=re.IGNORECASE)
    text = re.sub(r'\s*TUMOR or EDEMA\.?\s*$', '', text,
                  flags=re.IGNORECASE)

    return text.strip()
