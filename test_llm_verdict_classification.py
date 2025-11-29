"""
LLM-based Verdict Classification for Politifact Dataset
Uses Llama-3.1-8B-Instruct to classify statements into 6 verdict categories
"""

import json
import torch
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from datetime import datetime
import os

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 100  # Start with 1000 samples for testing, set to None for full dataset
BATCH_SIZE = 1  # Process one at a time for instruction-following models

# Verdict categories
VERDICTS = ["mostly-true", "half-true", "true", "mostly-false", "false", "pants-fire"]

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
    # Clean the response
    response = response_text.strip().lower()
    # print(f"LLM Response: {response}")
    # Look for exact matches
    for verdict in VERDICTS:
        # print(f"Checking verdict: {verdict}")
        if response == verdict:
            # print(f"Matched verdict: {verdict}")
            return verdict
    
    # If no match found, return the most likely based on keywords
    if "true" in response and "mostly" not in response and "half" not in response:
        return "true"
    elif "false" in response and "mostly" not in response and "pants" not in response:
        return "false"
    
    return "unknown"

def classify_statements(data, generator):
    """Classify all statements using the LLM"""
    predictions = []
    true_labels = []
    
    print(f"\nClassifying {len(data)} statements...")
    print(f"Using device: {DEVICE}")
    
    for item in tqdm(data, desc="Processing statements"):
        # Create prompt
        prompt = create_prompt(
            item['statement'],
            item['statement_originator'],
            item['statement_date'],
            item['statement_source']
        )
        
        # Generate prediction
        try:
            messages = [{"role": "user", "content": prompt}]
            
            outputs = generator(
                messages,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=False,
                top_p=0.9,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            # Extract the response
            response = outputs[0]['generated_text'][-1]['content']
            predicted_verdict = extract_verdict(response)
            
        except Exception as e:
            print(f"\nError processing statement: {e}")
            predicted_verdict = "unknown"
        
        predictions.append(predicted_verdict)
        true_labels.append(item['verdict'])
    
    return predictions, true_labels

def evaluate_results(predictions, true_labels, data):
    """Evaluate and display classification results"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Filter out unknown predictions
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "unknown"]
    filtered_predictions = [predictions[i] for i in valid_indices]
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    
    if len(filtered_predictions) < len(predictions):
        print(f"\nWarning: {len(predictions) - len(filtered_predictions)} predictions were 'unknown' and excluded")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(filtered_true_labels, filtered_predictions, 
                                labels=VERDICTS, zero_division=0))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(filtered_true_labels, filtered_predictions, labels=VERDICTS)
    cm_df = pd.DataFrame(cm, index=VERDICTS, columns=VERDICTS)
    print(cm_df)
    
    # Calculate overall accuracy
    correct = sum(1 for p, t in zip(filtered_predictions, filtered_true_labels) if p == t)
    accuracy = correct / len(filtered_predictions) if filtered_predictions else 0
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{len(filtered_predictions)})")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'statement': [data[i]['statement'] for i in valid_indices],
        'originator': [data[i]['statement_originator'] for i in valid_indices],
        'true_verdict': filtered_true_labels,
        'predicted_verdict': filtered_predictions,
        'correct': [p == t for p, t in zip(filtered_predictions, filtered_true_labels)]
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"verdict_classification_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"\nDetailed results saved to: {results_file}")
    
    # Show some examples
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 5)")
    print("="*80)
    for i in range(min(5, len(valid_indices))):
        idx = valid_indices[i]
        print(f"\nStatement: {data[idx]['statement'][:100]}...")
        print(f"True verdict: {filtered_true_labels[i]}")
        print(f"Predicted: {filtered_predictions[i]}")
        print(f"Correct: {'✓' if filtered_predictions[i] == filtered_true_labels[i] else '✗'}")

def main():
    print("="*80)
    print("LLM Verdict Classification Test")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'All'}")
    
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
    
    # Display verdict distribution
    verdict_counts = {}
    for item in data:
        verdict = item['verdict']
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    
    print("\nVerdict distribution in sample:")
    for verdict in VERDICTS:
        count = verdict_counts.get(verdict, 0)
        print(f"  {verdict:15s}: {count:4d} ({count/len(data)*100:.1f}%)")
    
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
        print("\nNote: You may need to:")
        print("1. Accept the Llama license on HuggingFace")
        print("2. Login with: huggingface-cli login")
        print("3. Have sufficient GPU memory (16GB+ recommended)")
        return
    
    # Classify statements
    predictions, true_labels = classify_statements(data, generator)
    
    # Evaluate results
    evaluate_results(predictions, true_labels, data)
    
    print("\n" + "="*80)
    print("Classification complete!")
    print("="*80)

if __name__ == "__main__":
    main()
