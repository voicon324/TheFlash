#!/usr/bin/env python3
"""
VNPT AI - The Builder Track 2
Entry-point script for Docker submission

This script runs inference like: python main.py infer --dataset test
But reads from /code/private_test.json and outputs submission.csv, submission_time.csv
"""

import json
import time
import sys
import os
from pathlib import Path

print("Starting predict.py...", flush=True)

# Set working directory to /code (for Docker environment)
try:
    os.chdir("/code")
    print(f"Working directory: {os.getcwd()}", flush=True)
except Exception as e:
    print(f"Warning: Could not change to /code: {e}", flush=True)

# Add pipeline directory to path
sys.path.insert(0, "/code/pipeline")
print("Added pipeline to path", flush=True)

try:
    from data_loader import Question
    print("Imported data_loader", flush=True)
    from inference import LLMInference
    print("Imported inference", flush=True)
    from config import api_config, pipeline_config
    print("Imported config", flush=True)
except Exception as e:
    print(f"Import error: {e}", flush=True)
    sys.exit(1)

# Constants
INPUT_FILE = "/code/private_test.json"
OUTPUT_DIR = Path("/code/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create output directory
OUTPUT_CSV = OUTPUT_DIR / "submission.csv"
OUTPUT_TIME_CSV = OUTPUT_DIR / "submission_time.csv"
RESULTS_FILE = OUTPUT_DIR / "results_small_submission_docker.json"


def load_test_questions(file_path: str) -> list:
    """Load questions from JSON file (same as data_loader.load_questions)."""
    import re
    
    print(f"Loading test data from {file_path}...", flush=True)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        qid = item.get('qid', '')
        question_text = item.get('question', '')
        choices = item.get('choices', [])
        answer = item.get('answer', None)
        
        # Extract context if present (same logic as data_loader)
        context = None
        raw_question = question_text
        
        if any(question_text.startswith(indicator) for indicator in ["Đoạn thông tin", "[1]", "-- Đoạn văn", "-- Document", "Title:"]):
            matches = list(re.finditer(r"Câu hỏi:\s*", question_text))
            if matches:
                last_match = matches[-1]
                raw_question = question_text[last_match.end():].strip()
                context = question_text[:last_match.start()].strip()
        
        q = Question(
            qid=qid,
            question=question_text,
            choices=choices,
            answer=answer,
            context=context,
            raw_question=raw_question
        )
        questions.append(q)
    
    print(f"Loaded {len(questions)} questions", flush=True)
    return questions


def run_inference(questions: list, use_large: bool = False) -> list:
    """
    Run inference like main.py infer --dataset test
    With checkpointing and time measurement per sample.
    """
    print(f"\n{'='*60}", flush=True)
    print("Step 3: Running Inference", flush=True)
    print(f"{'='*60}", flush=True)
    
    model_name = "large" if use_large else "small"
    print(f"Using model: {model_name}", flush=True)
    
    # Initialize LLM (same as main.py)
    llm = LLMInference(use_large=use_large, use_oss=False, use_react=False, use_cot=False)
    
    # Checkpointing: Load existing results
    existing_results = []
    processed_qids = set()
    
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            processed_qids = {r['qid'] for r in existing_results}
            print(f"Loaded {len(existing_results)} existing results from checkpoint.", flush=True)
        except Exception as e:
            print(f"Error loading checkpoint: {e}", flush=True)
            existing_results = []
    
    results = existing_results
    
    # Filter questions to process
    questions_to_process = [q for q in questions if q.qid not in processed_qids]
    
    if not questions_to_process:
        print("All questions already processed!", flush=True)
        return results
    
    print(f"Processing {len(questions_to_process)} questions (skipping {len(processed_qids)} already done)...", flush=True)
    
    correct = sum(1 for r in existing_results if r.get('correct'))
    total_with_answer = sum(1 for r in existing_results if r.get('ground_truth'))
    
    try:
        for i, q in enumerate(questions_to_process):
            current_idx = len(existing_results) + i + 1
            print(f"\n[{current_idx}/{len(questions)}] {q.qid}", flush=True)
            
            # Measure inference time for this single sample
            start_time = time.time()
            
            # Get prediction (same as main.py)
            predicted = llm.answer_question(q)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            result = {
                "qid": q.qid,
                "predicted": predicted,
                "ground_truth": q.answer,
                "time": round(inference_time, 4)
            }
            
            # Check correctness
            if q.answer:
                total_with_answer += 1
                is_correct = predicted == q.answer
                if is_correct:
                    correct += 1
                result["correct"] = is_correct
                print(f"  Predicted: {predicted}, Ground truth: {q.answer}, Correct: {is_correct}", flush=True)
            else:
                print(f"  Predicted: {predicted}, Time: {result['time']:.4f}s", flush=True)
            
            results.append(result)
            
            # Save checkpoint immediately
            with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Progress update
            if current_idx % 10 == 0 and total_with_answer > 0:
                acc = correct / total_with_answer * 100
                print(f"\n  --- Progress: {correct}/{total_with_answer} = {acc:.1f}% ---\n", flush=True)
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Results saved up to this point.", flush=True)
        return results
    
    # Final accuracy
    if total_with_answer > 0:
        final_acc = correct / total_with_answer * 100
        print(f"\n{'='*60}", flush=True)
        print(f"Final Accuracy: {correct}/{total_with_answer} = {final_acc:.2f}%", flush=True)
        print(f"{'='*60}", flush=True)
    
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)
    
    return results


def save_submissions(results: list):
    """Save results to submission.csv and submission_time.csv."""
    import pandas as pd
    
    # Sort by qid
    results_sorted = sorted(results, key=lambda x: x['qid'])
    
    # submission.csv - only qid and answer
    df_submission = pd.DataFrame([
        {"qid": r["qid"], "answer": r["predicted"]}
        for r in results_sorted
    ])
    df_submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved submission.csv to {OUTPUT_CSV}", flush=True)
    print(f"Total rows: {len(df_submission)}", flush=True)
    
    # submission_time.csv - qid, answer, and time
    df_time = pd.DataFrame([
        {"qid": r["qid"], "answer": r["predicted"], "time": r.get("time", 0)}
        for r in results_sorted
    ])
    df_time.to_csv(OUTPUT_TIME_CSV, index=False)
    print(f"Saved submission_time.csv to {OUTPUT_TIME_CSV}", flush=True)


def main():
    """Main entry point - runs like: python main.py infer --dataset test"""
    print("="*60, flush=True)
    print("VNPT AI - The Builder Track 2", flush=True)
    print("Docker Submission Pipeline", flush=True)
    print("="*60, flush=True)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}", flush=True)
        print("Please ensure the test data is mounted at /code/private_test.json", flush=True)
        sys.exit(1)
    
    # Load test questions
    questions = load_test_questions(INPUT_FILE)
    
    if not questions:
        print("ERROR: No questions loaded", flush=True)
        sys.exit(1)
    
    # Run inference (using small model, same as main.py infer --dataset test)
    results = run_inference(questions, use_large=False)
    
    # Save submissions
    save_submissions(results)
    
    print("\n" + "="*60, flush=True)
    print("Pipeline completed successfully!", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
