"""
LLM-based Verdict Classification with Conformal Prediction 
Uses token probabilities to create prediction sets with finite-sample coverage guarantees.
Corrected to use single-token labels to avoid tokenizer collisions.
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
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = None  # Set to None to use all data
CALIBRATION_SPLIT = 0.5  # 50% for calibration, 50% for test
ALPHA = 0.1  # Miscoverage rate (1-alpha = 90% coverage)
TRIALS = 2 # Number of trials to average results over (for stability)

# FIX: Map verdicts to single-token letters to prevent tokenizer collisions
# (e.g., preventing "mostly-true" and "mostly-false" from sharing the "mostly" token)
VERDICT_MAP = {
    "A": "true",
    "B": "mostly-true",
    "C": "half-true",
    "D": "mostly-false",
    "E": "false",
    "F": "pants-fire"
}

VERDICTS = list(VERDICT_MAP.values())

def load_data(file_path, max_samples=None):
    """Load JSON array data and prepare for classification"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data

def split_calibration_test(data, calib_ratio=0.3):
    """Split data into calibration and test sets (Exchangeability)"""
    n = len(data)
    indices = np.random.permutation(n)
    calib_size = int(n * calib_ratio)
    
    calib_data = [data[i] for i in indices[:calib_size]]
    test_data = [data[i] for i in indices[calib_size:]]
    
    return calib_data, test_data

def create_prompt(statement, originator, date, source):
    """
    Create a prompt forcing single-token outputs (A-F).
    """
    prompt = f"""You are a fact-checking expert. Your task is to predict the truthfulness verdict of a political statement.

Statement: "{statement}"
Speaker: {originator}
Date: {date}
Source: {source}

Classify the statement into one of these categories:
A) true: The statement is accurate.
B) mostly-true: The statement is mostly accurate with minor issues.
C) half-true: The statement is partially accurate.
D) mostly-false: The statement is mostly inaccurate.
E) false: The statement is inaccurate.
F) pants-fire: The statement is ridiculously false.

Respond with ONLY the letter (A, B, C, D, E, or F) corresponding to your verdict."""
    
    return prompt

def get_token_probabilities(generator, messages):
    """
    Get probability distribution over verdict tokens.
    Maps A-F logits back to verdict strings.
    """
    model = generator.model
    tokenizer = generator.tokenizer
    
    # Pre-encode the target letters (A, B, C, D, E, F)
    # We strip special tokens to get the raw ID for the letter
    token_map = {}
    for letter in VERDICT_MAP.keys():
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        token_map[letter] = tokens[0] # The ID for 'A', 'B', etc.

    # Prepare input
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
    
    # Calculate Softmax over ALL tokens first
    probs = torch.softmax(next_token_logits, dim=0)
    
    # Extract only the probabilities for our target letters
    verdict_probs = {}
    for letter, verdict_string in VERDICT_MAP.items():
        token_id = token_map[letter]
        verdict_probs[verdict_string] = probs[token_id].item()
    
    # Renormalize so our 6 options sum to 1.0
    total = sum(verdict_probs.values())
    if total > 0:
        verdict_probs = {k: v/total for k, v in verdict_probs.items()}
    else:
        # Fallback if model predicts something wildly different (unlikely)
        verdict_probs = {k: 1.0/len(VERDICTS) for k in VERDICTS}
    
    return verdict_probs

def compute_nonconformity_scores(data, generator):
    """
    Compute nonconformity scores for calibration set.
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
            
            # If the dataset has a label not in our map (unlikely), handle gracefully
            if true_label not in probs:
                print(f"Warning: Label '{true_label}' not in target set.")
                scores.append(1.0)
                continue

            # Nonconformity score: 1 - probability of true label
            score = 1.0 - probs.get(true_label, 0.0)
            scores.append(score)
            
        except Exception as e:
            print(f"\nError in calibration: {e}")
            scores.append(1.0) # Conservative max score
    
    return np.array(scores)

def get_conformal_threshold(scores, alpha):
    """
    Compute conformal prediction threshold using finite-sample correction.
    """
    n = len(scores)
    # Rigorous definition of quantile for conformal prediction
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(1, k), n) # Clamp to [1, n]
    
    # Sort scores
    sorted_scores = np.sort(scores)
    
    # The k-th smallest value (index k-1)
    threshold = sorted_scores[k - 1]
    
    return threshold

def get_prediction_set(probs, threshold):
    """
    Create prediction set based on conformal threshold.
    Include all labels where 1 - P(label) <= threshold
    (Equivalently: P(label) >= 1 - threshold)
    """
    prediction_set = []
    for verdict, prob in probs.items():
        # Score for this class would be 1 - prob
        # Include if score <= threshold
        if 1.0 - prob <= threshold:
            prediction_set.append(verdict)
    
    # Fallback: if set is empty (rare), include the argmax class
    if not prediction_set:
        prediction_set.append(max(probs.items(), key=lambda x: x[1])[0])
        
    return prediction_set

def classify_with_conformal(data, generator, threshold):
    """
    Classify statements with conformal prediction sets.
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

def evaluate_conformal_results(predictions, prediction_sets, true_labels, all_probs, data, trial_num=None):
    """Evaluate and print detailed metrics. Returns metrics dict."""
    trial_str = f" (Trial {trial_num})" if trial_num is not None else ""
    print("\n" + "="*80)
    print(f"CONFORMAL PREDICTION RESULTS{trial_str}")
    print("="*80)
    
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "unknown"]
    
    if not valid_indices:
        print("No valid predictions made.")
        return None

    # Filter out failed calls
    f_preds = [predictions[i] for i in valid_indices]
    f_sets = [prediction_sets[i] for i in valid_indices]
    f_true = [true_labels[i] for i in valid_indices]
    
    # 1. Marginal Coverage
    covered = [t in s for t, s in zip(f_true, f_sets)]
    coverage = np.mean(covered)
    
    # 2. Set Size
    set_sizes = [len(s) for s in f_sets]
    avg_set_size = np.mean(set_sizes)
    
    # 3. Singleton Rate (Efficiency)
    singleton_rate = np.mean([s == 1 for s in set_sizes])
    
    print(f"\nMarginal Coverage (Target: {1-ALPHA:.1%}): {coverage:.4f} ({coverage:.1%})")
    print(f"Average Set Size: {avg_set_size:.2f}")
    print(f"Singleton Rate: {singleton_rate:.1%}")
    
    print("\n" + "="*80)
    print("Point Prediction Performance (Top-1 Accuracy):")
    print("="*80)
    
    # Filter only known labels for sklearn report
    known_labels = [l for l in f_true if l in VERDICTS]
    known_preds = [p for i, p in enumerate(f_preds) if f_true[i] in VERDICTS]
    
    accuracy = None
    if known_labels:
        print(classification_report(known_labels, known_preds, labels=VERDICTS, zero_division=0))
        accuracy = np.mean([p == t for p, t in zip(known_preds, known_labels)])
        
        cm = confusion_matrix(known_labels, known_preds, labels=VERDICTS)
        cm_df = pd.DataFrame(cm, index=VERDICTS, columns=VERDICTS)
        print("\nConfusion Matrix:")
        print(cm_df)
    
    # Save Results
    results_df = pd.DataFrame({
        'statement': [data[i]['statement'] for i in valid_indices],
        'true_verdict': f_true,
        'predicted_verdict': f_preds,
        'prediction_set': [str(s) for s in f_sets],
        'set_size': set_sizes,
        'is_covered': covered
    })
    
    if trial_num is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results/zero-shot", exist_ok=True)
        results_file = f"results/zero-shot/conformal_results_trial{trial_num}_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
    
    # Return metrics and per-statement results for aggregation
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'singleton_rate': singleton_rate,
        'accuracy': accuracy,
        'results_df': results_df
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM verdict classification with conformal prediction."
    )
    parser.add_argument(
        "--data-path",
        default="datasets/politifact-english-no-media.json",
        help="Path to the JSON dataset file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SAMPLES,
        help="Maximum number of samples to load (omit or set to 0 for all).",
    )
    parser.add_argument(
        "--nums-trials",
        type=int,
        default=TRIALS,
        help="Number of trials to average results over (for stability).",
    )
    return parser.parse_args()


def main():
    print("="*80)
    print("LLM Verdict Classification with Conformal Prediction")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Calibration Split: {CALIBRATION_SPLIT:.2f}")
    print(f"Miscoverage Rate (alpha): {ALPHA:.2f}")

    args = parse_args()
    max_samples = None if args.max_samples in (None, 0) else args.max_samples

    print(f"Data path: {args.data_path}")
    print(f"Number of trials: {args.nums_trials}")
    
    # Load data once
    try:
        data = load_data(args.data_path, max_samples=max_samples)
        print(f"Loaded {len(data)} statements")
    except FileNotFoundError:
        print(f"Error: '{args.data_path}' not found.")
        print("Please ensure the data file path is correct.")
        return

    if len(data) < 10:
        print("Error: Need at least 10 samples to run.")
        return
    
    # Store results from all trials
    all_trial_metrics = []
    all_trial_results = []  # Store per-statement results
    
    # Load model once (outside trials loop for efficiency)
    print(f"\nLoading model: {MODEL_NAME}...")
    try:
        generator = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=DEVICE,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    for i in range(args.nums_trials):
        np.random.seed(42 + i)  # Different seed per trial
        torch.manual_seed(42 + i)
        print(f"\n{'='*80}")
        print(f"TRIAL {i+1}/{args.nums_trials}")
        print(f"{'='*80}")

        # 1. Split Data (different split per trial)
        calib_data, test_data = split_calibration_test(data, CALIBRATION_SPLIT)
        print(f"Calibration set: {len(calib_data)}")
        print(f"Test set: {len(test_data)}")
        
        # 2. Calibration
        calib_scores = compute_nonconformity_scores(calib_data, generator)
        threshold = get_conformal_threshold(calib_scores, ALPHA)
        print(f"\nCalibration complete. Threshold (q_hat): {threshold:.4f}")
        
        # 3. Testing
        predictions, prediction_sets, true_labels, all_probs = classify_with_conformal(
            test_data, generator, threshold
        )
        
        # 4. Evaluation
        metrics = evaluate_conformal_results(predictions, prediction_sets, true_labels, all_probs, test_data, trial_num=i+1)
        
        if metrics:
            all_trial_metrics.append(metrics)
            # Store per-statement results with trial number
            trial_df = metrics['results_df'].copy()
            trial_df['trial'] = i + 1
            all_trial_results.append(trial_df)
    
    # Aggregate results across trials
    if all_trial_metrics:
        print("\n" + "="*80)
        print("AGGREGATED RESULTS ACROSS ALL TRIALS")
        print("="*80)
        
        # Aggregate trial-level metrics
        avg_coverage = np.mean([m['coverage'] for m in all_trial_metrics])
        std_coverage = np.std([m['coverage'] for m in all_trial_metrics])
        avg_set_size = np.mean([m['avg_set_size'] for m in all_trial_metrics])
        std_set_size = np.std([m['avg_set_size'] for m in all_trial_metrics])
        avg_singleton = np.mean([m['singleton_rate'] for m in all_trial_metrics])
        std_singleton = np.std([m['singleton_rate'] for m in all_trial_metrics])
        
        accuracies = [m['accuracy'] for m in all_trial_metrics if m['accuracy'] is not None]
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            print(f"\nPoint Prediction Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        print(f"\nMarginal Coverage (Target: {1-ALPHA:.1%}): {avg_coverage:.4f} ± {std_coverage:.4f}")
        print(f"Average Set Size: {avg_set_size:.4f} ± {std_set_size:.4f}")
        print(f"Singleton Rate: {avg_singleton:.4f} ± {std_singleton:.4f}")
        
        # Aggregate per-statement results
        combined_df = pd.concat(all_trial_results, ignore_index=True)
        
        # Calculate average set size per statement
        statement_aggregated = combined_df.groupby('statement').agg({
            'set_size': ['mean', 'std', 'count'],
            'is_covered': 'mean',
            'true_verdict': 'first'
        }).reset_index()
        
        statement_aggregated.columns = ['statement', 'avg_set_size', 'std_set_size', 'num_trials', 'coverage_rate', 'true_verdict']
        
        # Save aggregated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results/zero-shot", exist_ok=True)
        
        # Save per-statement aggregated results
        agg_file = f"results/zero-shot/education/conformal_results_aggregated_{args.nums_trials}trials_{timestamp}.csv"
        statement_aggregated.to_csv(agg_file, index=False)
        print(f"\nPer-statement aggregated results saved to: {agg_file}")
        
        # Save all individual trial results combined
        all_trials_file = f"results/zero-shot/education/conformal_results_all_trials_{args.nums_trials}trials_{timestamp}.csv"
        combined_df.to_csv(all_trials_file, index=False)
        print(f"All trial results saved to: {all_trials_file}")
        
        # Print summary statistics
        print(f"\n" + "="*80)
        print("PER-STATEMENT STATISTICS")
        print("="*80)
        print(f"Total unique statements evaluated: {len(statement_aggregated)}")
        print(f"Average prediction set size across all statements: {statement_aggregated['avg_set_size'].mean():.4f}")
        print(f"Average coverage rate across all statements: {statement_aggregated['coverage_rate'].mean():.4f}")

if __name__ == "__main__":
    # Reproducibility
    # np.random.seed(42)
    # torch.manual_seed(42)
    main()