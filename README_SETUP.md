# LLM Verdict Classification Setup

## Prerequisites

1. **Python 3.8+** installed
2. **CUDA-capable GPU** (16GB+ VRAM recommended) or CPU (slower)
3. **HuggingFace account** with Llama access

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. HuggingFace Authentication

You need to accept the Llama license and authenticate:

1. Go to: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Generate an access token: https://huggingface.co/settings/tokens
4. Login via CLI:

```bash
huggingface-cli login
```

Or set environment variable:
```bash
$env:HUGGINGFACE_TOKEN="your_token_here"  # PowerShell
```

### 3. Run the Classification Test

```bash
python test_llm_verdict_classification.py
```

## Configuration Options

Edit `test_llm_verdict_classification.py` to adjust:

- `MAX_SAMPLES = 100` - Number of samples to test (None for all ~21k)
- `DEVICE = "cuda"` - Use "cpu" if no GPU available
- `MODEL_NAME` - Try other models like:
  - `"mistralai/Mistral-7B-Instruct-v0.3"`
  - `"microsoft/Phi-3.5-mini-instruct"`
  - `"google/gemma-2-9b-it"`

## Expected Runtime

- **GPU (RTX 3090/4090)**: ~30-60 seconds for 100 samples
- **CPU**: ~10-30 minutes for 100 samples
- **Full dataset (21k)**: Scale accordingly

## Output

The script will generate:
- Console output with classification metrics
- CSV file: `verdict_classification_results_TIMESTAMP.csv`
- Confusion matrix and classification report

## Troubleshooting

### Out of Memory Error
- Reduce `MAX_SAMPLES`
- Add `device_map="auto"` to pipeline
- Use smaller model or CPU

### Model Download Issues
- Check internet connection
- Verify HuggingFace authentication
- Try manual download via `huggingface-cli`

### Slow Performance
- Ensure GPU is being used (check console output)
- Update CUDA drivers
- Use smaller batch size
