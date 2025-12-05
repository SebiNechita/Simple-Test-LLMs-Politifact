"""
LLM-based Verdict Classification with Conformal Prediction
Uses token probabilities to create prediction sets with coverage guarantees
"""

import json
import torch
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from datetime import datetime
import numpy as np
import os

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 1000  # Use more samples for better calibration
CALIBRATION_SPLIT = 0.3  # 30% for calibration, 70% for test
ALPHA = 0.1  # Miscoverage rate (1-alpha = 90% coverage)

# Verdict categories
VERDICTS = ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"]

def load_data(file_path, max_samples=None):
    """Load JSONL data and prepare for classification"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line.strip())
            data.append(item)
    return data

def split_calibration_test(data, calib_ratio=0.3):
    """Split data into calibration and test sets"""
    n = len(data)
    indices = np.random.permutation(n)
    calib_size = int(n * calib_ratio)
    
    calib_data = [data[i] for i in indices[:calib_size]]
    test_data = [data[i] for i in indices[calib_size:]]
    
    return calib_data, test_data

def create_prompt(statement, originator, date, source):
    """Create a prompt for the LLM to classify the statement"""
    prompt = f"""You are a fact-checking expert. Your task is to predict the truthfulness verdict of a political statement based on the information provided.

Statement: "{statement}"
Speaker: {originator}
Date: {date}
Source: {source}

Based on this information, classify the statement into one of these categories:
- true: The statement is accurate
- mostly-true: The statement is mostly accurate with minor issues
- half-true: The statement is partially accurate
- mostly-false: The statement is mostly inaccurate
- false: The statement is inaccurate
- pants-fire: The statement is ridiculously false

Respond with ONLY the verdict category (one word from the list above), nothing else."""
    
    return prompt

def get_token_probabilities(generator, messages):
    """
    Get probability distribution over verdict tokens
    Returns: dict mapping verdict -> probability
    """
    # Get model and tokenizer
    model = generator.model
    tokenizer = generator.tokenizer
    
    # Tokenize verdict options
    verdict_token_ids = {}
    for verdict in VERDICTS:
        tokens = tokenizer.encode(verdict, add_special_tokens=False)
        verdict_token_ids[verdict] = tokens[0]  # First token of each verdict
    
    # Prepare input
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]  # Logits for next token
    
    # Get probabilities for verdict tokens
    probs = torch.softmax(next_token_logits, dim=0)
    
    verdict_probs = {}
    for verdict, token_id in verdict_token_ids.items():
        verdict_probs[verdict] = probs[token_id].item()
    
    # Normalize to sum to 1
    total = sum(verdict_probs.values())
    verdict_probs = {k: v/total for k, v in verdict_probs.items()}
    
    return verdict_probs

def compute_nonconformity_scores(data, generator):
    """
    Compute nonconformity scores for calibration set
    Score = 1 - P(true_label)
    """
    scores = []
    
    print("\nComputing nonconformity scores on calibration set...")
    for item in tqdm(data, desc="Calibration"):
        prompt = create_prompt(
            item['statement'],
            item['statement_originator'],
            item['statement_date'],
            item['statement_source']
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            probs = get_token_probabilities(generator, messages)
            
            true_label = item['verdict']
            # Nonconformity score: 1 - probability of true label
            score = 1.0 - probs.get(true_label, 0.0)
            scores.append(score)
            
        except Exception as e:
            print(f"\nError in calibration: {e}")
            scores.append(1.0)  # Worst case score
    
    return np.array(scores)

def get_conformal_threshold(scores, alpha):
    """
    Compute conformal prediction threshold
    """
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(scores, q_level)
    return threshold

def get_prediction_set(probs, threshold):
    """
    Create prediction set based on conformal threshold
    Include all labels where 1 - P(label) <= threshold
    """
    prediction_set = []
    for verdict, prob in probs.items():
        if 1.0 - prob <= threshold:
            prediction_set.append(verdict)
    return prediction_set

def classify_with_conformal(data, generator, threshold):
    """
    Classify statements with conformal prediction sets
    """
    predictions = []
    prediction_sets = []
    true_labels = []
    all_probs = []
    
    print(f"\nClassifying {len(data)} statements with conformal prediction...")
    print(f"Using device: {DEVICE}")
    print(f"Conformal threshold: {threshold:.4f}")
    
    for item in tqdm(data, desc="Testing"):
        prompt = create_prompt(
            item['statement'],
            item['statement_originator'],
            item['statement_date'],
            item['statement_source']
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            probs = get_token_probabilities(generator, messages)
            
            # Get prediction set
            pred_set = get_prediction_set(probs, threshold)
            
            # Point prediction: highest probability
            predicted_verdict = max(probs.items(), key=lambda x: x[1])[0]
            
            predictions.append(predicted_verdict)
            prediction_sets.append(pred_set)
            true_labels.append(item['verdict'])
            all_probs.append(probs)
            
        except Exception as e:
            print(f"\nError processing statement: {e}")
            predictions.append("unknown")
            prediction_sets.append(["unknown"])
            true_labels.append(item['verdict'])
            all_probs.append({})
    
    return predictions, prediction_sets, true_labels, all_probs

def evaluate_conformal_results(predictions, prediction_sets, true_labels, all_probs, data):
    """Evaluate conformal prediction results"""
    print("\n" + "="*80)
    print("CONFORMAL PREDICTION RESULTS")
    print("="*80)
    
    # Coverage: percentage of times true label is in prediction set
    coverage = np.mean([true_labels[i] in prediction_sets[i] 
                       for i in range(len(true_labels))])
    
    # Average prediction set size
    avg_set_size = np.mean([len(s) for s in prediction_sets])
    
    # Singleton predictions (set size = 1)
    singleton_rate = np.mean([len(s) == 1 for s in prediction_sets])
    
    print(f"\nCoverage (target: {1-ALPHA:.1%}): {coverage:.4f} ({coverage:.1%})")
    print(f"Average prediction set size: {avg_set_size:.2f}")
    print(f"Singleton rate (confident predictions): {singleton_rate:.1%}")
    
    # Point prediction accuracy
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "unknown"]
    filtered_predictions = [predictions[i] for i in valid_indices]
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    
    if len(filtered_predictions) < len(predictions):
        print(f"\nWarning: {len(predictions) - len(filtered_predictions)} predictions were 'unknown'")
    
    # Classification report
    print("\n" + "="*80)
    print("Point Prediction Performance:")
    print("="*80)
    print(classification_report(filtered_true_labels, filtered_predictions, 
                                labels=VERDICTS, zero_division=0))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(filtered_true_labels, filtered_predictions, labels=VERDICTS)
    cm_df = pd.DataFrame(cm, index=VERDICTS, columns=VERDICTS)
    print(cm_df)
    
    # Accuracy
    correct = sum(1 for p, t in zip(filtered_predictions, filtered_true_labels) if p == t)
    accuracy = correct / len(filtered_predictions) if filtered_predictions else 0
    print(f"\nPoint Prediction Accuracy: {accuracy:.4f} ({correct}/{len(filtered_predictions)})")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'statement': [data[i]['statement'] for i in valid_indices],
        'originator': [data[i]['statement_originator'] for i in valid_indices],
        'true_verdict': filtered_true_labels,
        'predicted_verdict': filtered_predictions,
        'prediction_set': [str(prediction_sets[i]) for i in valid_indices],
        'set_size': [len(prediction_sets[i]) for i in valid_indices],
        'covered': [filtered_true_labels[j] in prediction_sets[valid_indices[j]] 
                   for j in range(len(valid_indices))],
        'max_prob': [max(all_probs[i].values()) if all_probs[i] else 0 
                    for i in valid_indices],
        'correct': [p == t for p, t in zip(filtered_predictions, filtered_true_labels)]
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"conformal_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"\nDetailed results saved to: {results_file}")
    
    # Show examples
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 5)")
    print("="*80)
    for i in range(min(5, len(valid_indices))):
        idx = valid_indices[i]
        print(f"\nStatement: {data[idx]['statement'][:100]}...")
        print(f"True verdict: {filtered_true_labels[i]}")
        print(f"Point prediction: {filtered_predictions[i]}")
        print(f"Prediction set: {prediction_sets[idx]}")
        print(f"Covered: {'✓' if filtered_true_labels[i] in prediction_sets[idx] else '✗'}")
        print(f"Correct: {'✓' if filtered_predictions[i] == filtered_true_labels[i] else '✗'}")

def main():
    print("="*80)
    print("LLM Verdict Classification with Conformal Prediction")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    print(f"Target coverage: {1-ALPHA:.1%} (α={ALPHA})")
    
    # Check for GPU
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Warning: Running on CPU - this will be slow!")
    
    # Load data
    print("\nLoading data...")
    data = load_data('politifact_factcheck_data.json', max_samples=MAX_SAMPLES)
    print(f"Loaded {len(data)} statements")
    
    # Split into calibration and test
    calib_data, test_data = split_calibration_test(data, CALIBRATION_SPLIT)
    print(f"Calibration set: {len(calib_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Initialize model
    print(f"\nLoading model: {MODEL_NAME}")
    print("This may take a few minutes...")
    
    try:
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=DEVICE,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nNote: You may need to:")
        print("1. Accept the Llama license on HuggingFace")
        print("2. Login with: huggingface-cli login")
        print("3. Have sufficient GPU memory (16GB+ recommended)")
        return
    
    # Calibration: compute nonconformity scores
    calib_scores = compute_nonconformity_scores(calib_data, generator)
    threshold = get_conformal_threshold(calib_scores, ALPHA)
    
    print(f"\nCalibration complete!")
    print(f"Conformal threshold (q̂): {threshold:.4f}")
    
    # Test with conformal prediction
    predictions, prediction_sets, true_labels, all_probs = classify_with_conformal(
        test_data, generator, threshold
    )
    
    # Evaluate results
    evaluate_conformal_results(predictions, prediction_sets, true_labels, all_probs, test_data)
    
    print("\n" + "="*80)
    print("Classification complete!")
    print("="*80)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    main()
