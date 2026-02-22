# Evaluation: Reasoning Distillation

Scripts for the reasoning distillation experiment described in the main
README.  A Gemini 3 Flash teacher generates grounded surgical reasoning
traces; a LoRA adapter fine-tunes MedGemma 1.5 4B to reproduce them.

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_reasoning_traces.py` | Send FLAIR slices + GT measurements to Gemini, validate grounding, save traces |
| `train_reasoning_lora.py` | LoRA fine-tune MedGemma on the grounded traces |
| `eval_reasoning.py` | Compare Base vs Distilled vs Teacher on held-out patients |

## Pipeline

```
BraTS2021 patients
        |
        v
generate_reasoning_traces.py  -->  reasoning_traces/train.json
        |
        v
train_reasoning_lora.py       -->  experiments/<run>/final_checkpoint
        |
        v
eval_reasoning.py             -->  experiments/<run>/eval_results/
```

## Results

Pre-computed results from our final training run are in `results/`:

- `eval_reasoning.json` -- full eval output (20 held-out patients)
- `split.json` -- train / val / test patient split for reproducibility

### Summary (n=20)

| Metric             | Base MedGemma | Distilled Fine-Tune | Teacher (Gemini 3 Flash) |
|--------------------|:---:|:---:|:---:|
| Grounding          | 90% | 95% | 95% |
| Quality Score (0-4)| 2.3 | 3.2 | 3.6 |
| Per-slice accuracy | 47% | 79% | 81% |

Per-slice accuracy measures hallucination control (no false tissue
claims on healthy slices), not classification.

## Dependencies

Same as the main project, plus:

```
trl        # SFT trainer (train only)
google-genai  # Gemini API (trace generation only)
tqdm
```

## Data

These scripts require the BraTS 2021 dataset (not included).  Set the
`BRATS_DIR` environment variable or pass `--brats_dir` to point to your
local copy.
