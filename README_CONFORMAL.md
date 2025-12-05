# Conformal Prediction for LLM Verdict Classification

This implementation adds **conformal prediction** to provide uncertainty quantification with coverage guarantees.

## What is Conformal Prediction?

Conformal prediction creates **prediction sets** instead of single predictions, with **theoretical coverage guarantees**. For example, with α=0.1, the true label will be in the prediction set at least 90% of the time.

## Key Features

### 1. **Token-Level Probabilities**
- Extracts probability distribution over verdict tokens from LLM
- Uses softmax over next-token logits
- More reliable than parsing text confidence

### 2. **Calibration**
- Uses 30% of data to compute nonconformity scores
- Nonconformity score: `1 - P(true_label)`
- Computes conformal threshold from calibration scores

### 3. **Prediction Sets**
- Includes all labels where `1 - P(label) ≤ threshold`
- Provides coverage guarantee: true label ∈ prediction set with probability ≥ 1-α
- Smaller sets = more confident predictions

## Usage

```bash
python test_llm_verdict_classification_conformal.py
```

## Configuration

Edit these parameters in the script:

```python
MAX_SAMPLES = 100          # Total samples to use
CALIBRATION_SPLIT = 0.3    # 30% for calibration
ALPHA = 0.1                # Target miscoverage rate (10%)
```

## Output Metrics

1. **Coverage**: % of times true label is in prediction set (should be ≥ 90% for α=0.1)
2. **Average set size**: Smaller is better (more confident)
3. **Singleton rate**: % of predictions with only one label (maximum confidence)
4. **Point prediction accuracy**: Traditional accuracy metric

## Results CSV

Saved as `conformal_results_TIMESTAMP.csv` with:
- Point prediction
- Prediction set (list of possible verdicts)
- Set size
- Coverage (whether true label was included)
- Maximum probability
- Correctness

## Example Output

```
Coverage (target: 90.0%): 0.9143 (91.4%)
Average prediction set size: 2.31
Singleton rate (confident predictions): 45.7%

Sample Prediction:
Statement: "John McCain opposed bankruptcy protections..."
True verdict: true
Point prediction: mostly-true
Prediction set: ['true', 'mostly-true']
Covered: ✓
Correct: ✗
```

## Interpretation

- **Singleton set** (size=1): High confidence - model is sure
- **Large set** (size=4+): Low confidence - model is uncertain
- **Coverage**: If > 90% (for α=0.1), conformal guarantee is maintained
- **Trade-off**: Lower α → higher coverage but larger sets

## Theory

Conformal prediction guarantees:
```
P(Y ∈ C(X)) ≥ 1 - α
```

Where:
- `Y` = true label
- `C(X)` = prediction set for input X
- `α` = miscoverage rate

This holds under exchangeability assumption (data is i.i.d. or similarly distributed).

## References

- Angelopoulos & Bates (2021): "A Gentle Introduction to Conformal Prediction"
- Vovk et al. (2005): "Algorithmic Learning in a Random World"
