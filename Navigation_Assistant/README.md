# MedGemma 1.5 Surgical Navigation Assistant

Tool-augmented VLM for brain tumor resection guidance.  MedGemma 1.5 4B
is scaffolded with ground-truth segmentation masks, a Harvard-Oxford brain
atlas, and deterministic geometry tools so that every measurement it cites
is verifiably correct.

## Architecture

```
MRI Input (FLAIR)
    |
    v
[GT Segmentation Mask] --> 100% accurate tissue lookup
    |
[Harvard-Oxford Atlas]  --> Brain region + eloquent cortex warnings
    |
[Geometry Tools]        --> Distances, volumes, margins
    |
    v
[MedGemma 1.5 4B] <-- Image + verified measurements injected
    |
    v
Grounded clinical guidance
```

**Core insight:** The scaffold computes verified facts; MedGemma reasons
about them.

## Module structure

| File           | Purpose                                          |
|----------------|--------------------------------------------------|
| `config.py`    | Paths, constants, tissue labels                  |
| `state.py`     | Global application state singleton               |
| `tools.py`     | Scaffold tools (mask, atlas, distances, volumes)  |
| `inference.py` | Model loading, `run_inference()`, trajectory analysis |
| `rendering.py` | Image rendering, overlays, 3-D Plotly view       |
| `handlers.py`  | Gradio event handlers                            |
| `app.py`       | UI layout and entry point                        |

## Setup

```bash
# Create environment
conda create -n medgemma python=3.11 -y
conda activate medgemma

# Install dependencies
pip install -r requirements.txt
```

## Data

A sample BraTS 2021 scan and segmentation mask are bundled in `data/`.
To use a different patient, set environment variables:

```bash
export SCAN_PATH=/path/to/flair.nii
export SEG_PATH=/path/to/seg.nii
```

## Running

```bash
python app.py
```

This loads MedGemma 1.5 4B in 4-bit quantization (~3 GB VRAM), the
Harvard-Oxford brain atlas, both LoRA adapters, and launches a Gradio
interface with a shareable URL.

### LoRA adapters

Both fine-tuned adapters are bundled in `checkpoints/` and loaded
automatically.  To override with different checkpoints:

```bash
export LORA_PATH=/path/to/tissue_lora_checkpoint
export REASONING_LORA_PATH=/path/to/reasoning_lora_checkpoint
```

## Hardware requirements

- GPU: 8+ GB VRAM (tested on RTX 4080 16 GB)
- RAM: 16+ GB
- Disk: ~5 GB (model weights downloaded on first run)

## Live demo

https://huggingface.co/spaces/Summicron50mm/medgemma-surgical-nav

## Results

### Tissue Classification (single-slice, crosshair)

| Approach             | Accuracy |
|----------------------|----------|
| Base MedGemma        | 50%      |
| + LoRA fine-tuning   | 78%      |
| + GT Scaffold        | 100%     |

### Reasoning Distillation (multi-slice trajectory, n=20)

| Metric             | Base | Distilled | Teacher (Gemini 3 Flash) |
|--------------------|:----:|:---------:|:------------------------:|
| Grounding          | 90%  | 95%       | 95%                      |
| Quality (0-4)      | 2.3  | 3.2       | 3.6                      |
| Per-slice accuracy | 47%  | 79%       | 81%                      |

## Limitations

- **FLAIR only.** FLAIR cannot distinguish enhancing tumor from edema
  (both hyperintense).  T1-contrast would be needed.
- **Requires ground-truth segmentation.** This is a reasoning and
  guidance tool, not a segmentation pipeline.
- **Empirical atlas alignment.** The BraTS-to-MNI coordinate mapping
  uses an affine approximation, not proper nonlinear registration.
- **Single demo case.** One BraTS 2021 patient is included as a sample.
