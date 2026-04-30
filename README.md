# LLM Verdict Classification with Conformal Prediction

A framework for evaluating LLM-based fact-checking on the PolitiFact dataset using conformal prediction to produce prediction sets with formal coverage guarantees.

## Overview

This project classifies political statements into one of six PolitiFact verdict categories using large language models (LLMs), then applies conformal prediction to quantify uncertainty. Instead of a single point prediction, the model outputs a **prediction set** — a subset of labels guaranteed to contain the true label at least `1 - α` of the time (e.g., 90% coverage with α = 0.1).

Three conformal prediction algorithms are implemented:

| Algorithm | Nonconformity Score | Prediction Set Behaviour |
|---|---|---|
| **LAC** (Least Ambiguous Classifier) | `1 - P(true label)` | Smallest sets; may be non-contiguous |
| **APS** (Adaptive Prediction Sets) | Randomized cumulative mass | Exact marginal coverage |
| **Ordinal-APS** | APS restricted to contiguous intervals | Sets respect the ordinal truth scale |

## Verdict Categories

Statements are classified into six labels, mapped to single tokens to avoid tokenizer ambiguity:

| Token | Label | Meaning |
|---|---|---|
| A | `true` | Accurate |
| B | `mostly-true` | Mostly accurate with minor issues |
| C | `half-true` | Partially accurate |
| D | `mostly-false` | Mostly inaccurate |
| E | `false` | Completely inaccurate |
| F | `pants-fire` | Ridiculously false |

## Project Structure

```
src/
├── APS/                          # Adaptive Prediction Sets
│   ├── zero_shot_conformal_verdict_classifications_aps.py
│   └── zero_shot_conformal_verdict_classifications_aps_with_evidence.py
├── LAC/                          # Least Ambiguous Classifier
│   ├── zero_shot_conformal_verdict_classifications_lac.py
│   └── zero_shot_conformal_verdict_classifications_lac_with_evidence.py
├── ORDINAL-APS/                  # Ordinal variant (contiguous prediction sets)
│   ├── zero_shot_conformal_verdict_classifications_ordinal_aps.py
│   └── zero_shot_conformal_verdict_classifications_ordinal_aps_with_evidence.py
├── experimental/                 # Development and exploratory scripts
│   ├── test_llm_verdict_classification.py       # Basic zero-shot classification
│   ├── test_llm_consistency.py                  # Measures output entropy across runs
│   ├── evaluate_results.py                      # Evaluates a results CSV
│   ├── aggregate_trial_results.py               # Aggregates multi-trial CSV outputs
│   ├── check-ordinality.py                      # Validates prediction set contiguity
│   ├── check_explanations_llms.py               # Generates verdicts + explanations
│   ├── zero_shot_conformal_verdict_classifications_bert.py  # DeBERTa/BART baseline
│   └── ...
└── analysis/                     # Jupyter notebooks and visualisation outputs
    ├── analysis.ipynb
    ├── analyse_results_lac_llama.ipynb
    └── *.png

datasets/
├── politifact-english-no-media.json             # Full dataset (no media statements)
├── politifact-education.json                    # Topic-specific subsets
├── politifact-elections.json
├── politifact-health-care.json
├── politifact-crime.json
├── politifact-economy.json
├── politifact-immigration.json
├── politifact-jobs.json
├── politifact-taxes.json
├── politifact_evidence-*.json                   # Variants with evidence field
└── ...
```

## Prerequisites

- **Python 3.8 or newer** — Python 3.11 recommended
- **pip** — install dependencies with `pip install -r requirements.txt`
- **HuggingFace account and token** — required for gated models (Llama). Create a token at https://huggingface.co/settings/tokens
- **GPU (recommended)** — CUDA-capable GPU significantly speeds inference. 16 GB+ VRAM recommended for Llama-3.1-8B. CPU is supported but slow.
- **Disk space** — 10+ GB for model weights and caches

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. HuggingFace Authentication

Accept the Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, then authenticate:

```bash
huggingface-cli login
```

Or set the environment variable:

```bash
export HUGGINGFACE_TOKEN="your_token_here"          # bash/zsh
$env:HUGGINGFACE_TOKEN="your_token_here"            # PowerShell
```

## Running the Scripts

All main scripts accept CLI arguments. Run with `--help` to see all options.

### LAC (Least Ambiguous Classifier)

```bash
python src/LAC/zero_shot_conformal_verdict_classifications_lac.py \
  --data-path datasets/politifact-english-no-media.json \
  --max-samples 500 \
  --nums-trials 5
```

With evidence field:

```bash
python src/LAC/zero_shot_conformal_verdict_classifications_lac_with_evidence.py \
  --data-path datasets/politifact_evidence-english-no-media.json
```

### APS (Adaptive Prediction Sets)

```bash
python src/APS/zero_shot_conformal_verdict_classifications_aps.py \
  --data-path datasets/politifact-english-no-media.json \
  --max-samples 500 \
  --nums-trials 5
```

### Ordinal-APS

```bash
python src/ORDINAL-APS/zero_shot_conformal_verdict_classifications_ordinal_aps.py \
  --data-path datasets/politifact-english-no-media.json \
  --max-samples 500 \
  --nums-trials 5
```


## Configuration

Key parameters shared across all main scripts:

| Parameter | Default | Description |
|---|---|---|
| `--model-name` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--data-path` | *(required)* | Path to the dataset JSON file |
| `--max-samples` | `None` (all) | Number of statements to evaluate |
| `--nums-trials` | `2` | Trials to average for stability |
| `--output-folder` | `results/` | Directory for CSV output |
| `CALIBRATION_SPLIT` | `0.5` | Fraction of data used for calibration |
| `ALPHA` | `0.1` | Target miscoverage rate (90% coverage) |

Alternative models:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Llama-3.3-70B-Instruct` (via LlamaStack API)

## Output

Each run produces a CSV file with the following columns:

| Column | Description |
|---|---|
| `statement` | The political claim |
| `true_verdict` | Ground-truth PolitiFact label |
| `predicted_verdict` | Model's point prediction |
| `prediction_set` | Conformal prediction set (list of labels) |
| `set_size` | Number of labels in the prediction set |
| `is_covered` | Whether the true label is in the prediction set |

Reported metrics per trial:

- **Marginal Coverage** — fraction of test samples where true label is in prediction set (target: ≥ 1 - α)
- **Average Set Size** — mean number of labels per prediction set
- **Singleton Rate** — fraction of sets with exactly one label
- **Accuracy** — standard point-prediction accuracy

## Expected Runtime

| Setup | Samples | Time (approx.) |
|---|---|---|
| GPU (RTX 3090/4090) | 100 | ~1–2 min |
| GPU (RTX 3090/4090) | 1000 | ~10–20 min |
| CPU | 100 | ~30–60 min |

## Troubleshooting

**Out of Memory** — reduce `--max-samples`, use a smaller model, or switch to CPU (`DEVICE = "cpu"`).

**Model Download Issues** — verify HuggingFace authentication and internet connection; try `huggingface-cli download`.

**Slow Inference** — confirm CUDA is available (`torch.cuda.is_available()`); update CUDA drivers.
