"""
LLM Consistency Test - Check if LLM provides consistent answers
Runs the same statement through the model multiple times to measure variability
"""

import json
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from collections import Counter
import numpy as np

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RUNS = 15  # Number of times to run each statement
NUM_STATEMENTS = 1500 # Number of statements to test
TEMPERATURE = 0.1  # Temperature for generation (0.1 = more deterministic, 1.0 = more random)

# Verdict categories
VERDICTS = ["mostly-true", "half-true", "true", "mostly-false", "false", "pants-fire"]

def load_data(file_path, num_samples=None):
    """Load JSONL data"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            item = json.loads(line.strip())
            data.append(item)
    return data

def create_prompt(statement, originator, date, source):
    """Create a prompt for the LLM to classify the statement"""
    prompt = f"""You are a fact-checking expert. Your task is to predict the truthfulness verdict of a political statement based on the information provided.

Statement: "{statement}"
Speaker: {originator}
Date: {date}
Source: {source}

Based on this information, classify the statement into one of these categories:
- true: The statement is accurate and there is nothing significant missing.
- mostly-true: The statement is accurate but needs clarification or additional information.
- half-true: The statement is partially accurate but leaves out important details or takes things out of context.
- mostly-false: The statement contains an element of truth but ignores critical facts that would give a different impression.
- false: The statement is not accurate.
- pants-fire: The statement is not accurate and makes a ridiculous claim.

Respond with ONLY the verdict category (one word from the list above), nothing else."""
    
    return prompt

def extract_verdict(response_text):
    """Extract verdict from LLM response"""
    response = response_text.strip().lower()
    
    # Look for exact matches
    for verdict in VERDICTS:
        if response == verdict:
            return verdict
    
    # If no match found, return the most likely based on keywords
    if "true" in response and "mostly" not in response and "half" not in response:
        return "true"
    elif "false" in response and "mostly" not in response and "pants" not in response:
        return "false"
    
    return "unknown"

def test_consistency(item, generator, num_runs):
    """Run the same statement multiple times and collect predictions"""
    prompt = create_prompt(
        item['statement'],
        item['statement_originator'],
        item['statement_date'],
        item['statement_source']
    )
    
    predictions = []
    raw_responses = []
    
    for _ in range(num_runs):
        try:
            messages = [{"role": "user", "content": prompt}]
            
            outputs = generator(
                messages,
                max_new_tokens=50,
                temperature=TEMPERATURE,
                do_sample=True if TEMPERATURE > 0 else False,
                top_p=0.9,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            # Extract the response
            response = outputs[0]['generated_text'][-1]['content']
            raw_responses.append(response)
            predicted_verdict = extract_verdict(response)
            predictions.append(predicted_verdict)
            
        except Exception as e:
            print(f"\nError processing statement: {e}")
            predictions.append("unknown")
            raw_responses.append("ERROR")
    
    return predictions, raw_responses

def analyze_consistency(predictions, true_verdict):
    """Analyze the consistency of predictions"""
    # Count occurrences
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0] if counter else ("unknown", 0)
    
    # Calculate metrics
    total_predictions = len(predictions)
    most_common_verdict = most_common[0]
    most_common_count = most_common[1]
    consistency_rate = most_common_count / total_predictions if total_predictions > 0 else 0
    
    # Check if any prediction matches the true verdict
    correct_predictions = sum(1 for p in predictions if p == true_verdict)
    accuracy_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate entropy (measure of uncertainty)
    entropy = 0
    for verdict, count in counter.items():
        if count > 0:
            prob = count / total_predictions
            entropy -= prob * np.log2(prob)
    
    return {
        'predictions': predictions,
        'counter': dict(counter),
        'most_common_verdict': most_common_verdict,
        'most_common_count': most_common_count,
        'consistency_rate': consistency_rate,
        'correct_predictions': correct_predictions,
        'accuracy_rate': accuracy_rate,
        'entropy': entropy,
        'unique_verdicts': len(counter)
    }

def main():
    print("="*80)
    print("LLM Consistency Test")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Runs per statement: {NUM_RUNS}")
    print(f"Number of statements to test: {NUM_STATEMENTS}")
    
    # Check for GPU
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    print("\nLoading data...")
    data = load_data('politifact_factcheck_data.json', num_samples=NUM_STATEMENTS)
    print(f"Loaded {len(data)} statements for testing")
    
    # Initialize the model
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
        return
    
    # Test consistency for each statement
    results = []
    
    print("\n" + "="*80)
    print("TESTING CONSISTENCY")
    print("="*80)
    
    for idx, item in enumerate(tqdm(data, desc="Testing statements")):
        print(f"\n{'='*80}")
        print(f"Statement {idx + 1}/{len(data)}")
        print(f"{'='*80}")
        print(f"Statement: {item['statement'][:100]}...")
        print(f"Speaker: {item['statement_originator']}")
        print(f"True verdict: {item['verdict']}")
        print(f"\nRunning {NUM_RUNS} times...")
        
        # Run multiple times
        predictions, raw_responses = test_consistency(item, generator, NUM_RUNS)
        
        # Analyze consistency
        analysis = analyze_consistency(predictions, item['verdict'])
        
        # Display results
        print(f"\nPredictions: {predictions}")
        print(f"Distribution: {analysis['counter']}")
        print(f"Most common: {analysis['most_common_verdict']} ({analysis['most_common_count']}/{NUM_RUNS})")
        print(f"Consistency rate: {analysis['consistency_rate']:.2%}")
        print(f"Accuracy rate: {analysis['accuracy_rate']:.2%} (matches true verdict)")
        print(f"Entropy: {analysis['entropy']:.3f} (0 = perfectly consistent, higher = more variable)")
        print(f"Unique verdicts: {analysis['unique_verdicts']}")
        
        # Store results
        results.append({
            'statement_id': idx,
            'statement': item['statement'],
            'originator': item['statement_originator'],
            'true_verdict': item['verdict'],
            'predictions': predictions,
            'raw_responses': raw_responses,
            'distribution': analysis['counter'],
            'most_common_verdict': analysis['most_common_verdict'],
            'most_common_count': analysis['most_common_count'],
            'consistency_rate': analysis['consistency_rate'],
            'correct_predictions': analysis['correct_predictions'],
            'accuracy_rate': analysis['accuracy_rate'],
            'entropy': analysis['entropy'],
            'unique_verdicts': analysis['unique_verdicts']
        })
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    avg_consistency = np.mean([r['consistency_rate'] for r in results])
    avg_accuracy = np.mean([r['accuracy_rate'] for r in results])
    avg_entropy = np.mean([r['entropy'] for r in results])
    avg_unique_verdicts = np.mean([r['unique_verdicts'] for r in results])
    
    print(f"Average consistency rate: {avg_consistency:.2%}")
    print(f"Average accuracy rate: {avg_accuracy:.2%}")
    print(f"Average entropy: {avg_entropy:.3f}")
    print(f"Average unique verdicts per statement: {avg_unique_verdicts:.2f}")
    
    # Find most and least consistent statements
    most_consistent = max(results, key=lambda x: x['consistency_rate'])
    least_consistent = min(results, key=lambda x: x['consistency_rate'])
    
    print(f"\nMost consistent statement:")
    print(f"  Statement: {most_consistent['statement'][:80]}...")
    print(f"  Consistency: {most_consistent['consistency_rate']:.2%}")
    print(f"  Distribution: {most_consistent['distribution']}")
    
    print(f"\nLeast consistent statement:")
    print(f"  Statement: {least_consistent['statement'][:80]}...")
    print(f"  Consistency: {least_consistent['consistency_rate']:.2%}")
    print(f"  Distribution: {least_consistent['distribution']}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary CSV
    summary_df = pd.DataFrame([{
        'statement_id': r['statement_id'],
        'statement': r['statement'][:100] + '...',
        'true_verdict': r['true_verdict'],
        'most_common_verdict': r['most_common_verdict'],
        'consistency_rate': r['consistency_rate'],
        'accuracy_rate': r['accuracy_rate'],
        'entropy': r['entropy'],
        'unique_verdicts': r['unique_verdicts'],
        'distribution': str(r['distribution'])
    } for r in results])
    
    summary_file = f"consistency_summary_{MODEL_NAME.replace('/', '_')}_temp{TEMPERATURE}_runs{NUM_RUNS}_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"\nSummary saved to: {summary_file}")
    
    # Save detailed results (including all predictions)
    detailed_rows = []
    for r in results:
        for run_idx, (pred, raw) in enumerate(zip(r['predictions'], r['raw_responses'])):
            detailed_rows.append({
                'statement_id': r['statement_id'],
                'run': run_idx + 1,
                'statement': r['statement'],
                'true_verdict': r['true_verdict'],
                'prediction': pred,
                'raw_response': raw,
                'correct': pred == r['true_verdict']
            })
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_file = f"consistency_detailed_{MODEL_NAME.replace('/', '_')}_temp{TEMPERATURE}_runs{NUM_RUNS}_{timestamp}.csv"
    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8')
    print(f"Detailed results saved to: {detailed_file}")
    
    print("\n" + "="*80)
    print("Consistency test complete!")
    print("="*80)
    print(f"\nInterpretation:")
    print(f"- Consistency rate: % of predictions that match the most common prediction")
    print(f"- Accuracy rate: % of predictions that match the true verdict")
    print(f"- Entropy: 0 = perfectly consistent, ~{np.log2(len(VERDICTS)):.2f} = uniformly random")
    print(f"- Temperature {TEMPERATURE}: Lower = more deterministic, higher = more random")

if __name__ == "__main__":
    main()
