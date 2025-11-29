"""
Evaluate LLM Verdict Classification Results from CSV
Analyzes accuracy, generates classification report, and confusion matrix
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import glob
import os

# Verdict categories
VERDICTS = ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"]

def load_results(csv_file):
    """Load results from CSV file"""
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def evaluate_predictions(df):
    """Evaluate predictions and generate metrics"""
    
    # Extract predictions and true labels
    true_labels = df['true_verdict'].tolist()
    predicted_labels = df['predicted_verdict'].tolist()
    
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nTotal predictions: {len(predicted_labels)}")
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    correct = sum(df['correct'])
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(predicted_labels)})")
    
    # Classification report
    print("\n" + "="*80)
    print("Classification Report:")
    print("="*80)
    print(classification_report(true_labels, predicted_labels, 
                                labels=VERDICTS, zero_division=0))
    
    # Confusion matrix
    print("\n" + "="*80)
    print("Confusion Matrix:")
    print("="*80)
    cm = confusion_matrix(true_labels, predicted_labels, labels=VERDICTS)
    cm_df = pd.DataFrame(cm, index=VERDICTS, columns=VERDICTS)
    print(cm_df)
    print("\nRows = True Labels, Columns = Predicted Labels")
    
    # Per-class accuracy
    print("\n" + "="*80)
    print("Per-Class Accuracy:")
    print("="*80)
    for i, verdict in enumerate(VERDICTS):
        total = sum(cm[i])
        if total > 0:
            class_accuracy = cm[i][i] / total
            print(f"{verdict:15s}: {class_accuracy:.4f} ({cm[i][i]}/{total})")
        else:
            print(f"{verdict:15s}: N/A (no samples)")
    
    # Most common errors
    print("\n" + "="*80)
    print("Most Common Misclassifications (Top 10):")
    print("="*80)
    
    error_df = df[df['correct'] == False].copy()
    if len(error_df) > 0:
        error_counts = error_df.groupby(['true_verdict', 'predicted_verdict']).size()
        error_counts = error_counts.sort_values(ascending=False)
        
        for i, ((true_v, pred_v), count) in enumerate(error_counts.head(10).items()):
            print(f"{i+1}. {true_v:15s} → {pred_v:15s}: {count:3d} times")
    else:
        print("No errors found!")
    
    # Sample incorrect predictions
    print("\n" + "="*80)
    print("Sample Incorrect Predictions (First 5):")
    print("="*80)
    
    incorrect = df[df['correct'] == False].head(5)
    for idx, row in incorrect.iterrows():
        print(f"\nStatement: {row['statement'][:100]}...")
        print(f"Originator: {row['originator']}")
        print(f"True verdict: {row['true_verdict']}")
        print(f"Predicted: {row['predicted_verdict']}")
    
    # Sample correct predictions
    print("\n" + "="*80)
    print("Sample Correct Predictions (First 5):")
    print("="*80)
    
    correct_df = df[df['correct'] == True].head(5)
    for idx, row in correct_df.iterrows():
        print(f"\nStatement: {row['statement'][:100]}...")
        print(f"Originator: {row['originator']}")
        print(f"Verdict: {row['true_verdict']}")

def main():
    print("="*80)
    print("LLM Verdict Classification Results Evaluator")
    print("="*80)
    
    # Check for command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Find most recent results file
        results_files = glob.glob('verdict_classification_results_*.csv')
        if not results_files:
            print("\nNo results files found!")
            print("Usage: python evaluate_results.py [csv_file]")
            print("   or: Place verdict_classification_results_*.csv in current directory")
            sys.exit(1)
        
        # Sort by modification time, get most recent
        results_files.sort(key=os.path.getmtime, reverse=True)
        csv_file = results_files[0]
        print(f"\nUsing most recent file: {csv_file}")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"\nError: File '{csv_file}' not found!")
        sys.exit(1)
    
    print(f"Loading results from: {csv_file}\n")
    
    # Load and evaluate
    df = load_results(csv_file)
    evaluate_predictions(df)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
