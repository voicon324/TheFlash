#!/usr/bin/env python3
"""
VNPT AI Validation Pipeline - Main Orchestrator

Usage:
    python main.py preprocess   - Load and preprocess data
    python main.py embed        - Create embeddings (requires API)
    python main.py infer        - Run inference (requires API)
    python main.py run          - Run full pipeline
    python main.py eval         - Evaluate results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from config import api_config, pipeline_config
from data_loader import (
    Question, 
    load_val_data, 
    load_test_data, 
    save_results
)
from embedding import embedding_manager, embed_text
from search import VectorSearcher
from inference import llm_small, llm_large, llm_oss


def preprocess_data(dataset: str = "val"):
    """
    Step 1: Load and preprocess data.
    
    Args:
        dataset: 'val' or 'test'
    """
    print(f"=" * 60)
    print(f"Step 1: Preprocessing {dataset} dataset")
    print(f"=" * 60)
    
    # Load data
    if dataset == "val":
        questions = load_val_data()
    else:
        questions = load_test_data()
    
    print(f"\nLoaded {len(questions)} questions")
    
    # Statistics
    with_context = sum(1 for q in questions if q.has_context())
    without_context = len(questions) - with_context
    
    print(f"\nStatistics:")
    print(f"  - Questions with embedded context: {with_context}")
    print(f"  - Questions without context: {without_context}")
    
    # Show sample questions
    print(f"\n--- Sample Questions ---")
    for i, q in enumerate(questions[:3]):
        print(f"\n[{i+1}] QID: {q.qid}")
        print(f"    Has context: {q.has_context()}")
        print(f"    Choices: {len(q.choices)}")
        if q.answer:
            print(f"    Answer: {q.answer}")
    
    # Save preprocessed info
    output_dir = Path(pipeline_config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    info = {
        "dataset": dataset,
        "total_questions": len(questions),
        "with_context": with_context,
        "without_context": without_context,
        "questions": [
            {
                "qid": q.qid,
                "has_context": q.has_context(),
                "num_choices": len(q.choices),
                "answer": q.answer
            }
            for q in questions
        ]
    }
    
    info_file = output_dir / f"{dataset}_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved info to {info_file}")
    
    return questions


def create_embeddings(questions: List[Question]):
    """
    Step 2: Create embeddings for questions.
    
    Args:
        questions: List of Question objects
    """
    print(f"\n" + "=" * 60)
    print(f"Step 2: Creating Embeddings")
    print(f"=" * 60)
    
    # Try to load existing cache
    if embedding_manager.load_cache():
        print("Loaded existing embeddings cache")
    
    # Prepare texts to embed (questions without context need embeddings for retrieval)
    texts_to_embed = []
    cache_keys = []
    
    for q in questions:
        # Embed the question text for potential retrieval
        text = q.raw_question if q.raw_question else q.question
        texts_to_embed.append(text)
        cache_keys.append(q.qid)
    
    print(f"\nEmbedding {len(texts_to_embed)} questions...")
    
    # Embed in batches
    embeddings = embedding_manager.get_embeddings_batch(
        texts_to_embed, 
        cache_keys=cache_keys,
        show_progress=True
    )
    
    # Count successful embeddings
    successful = sum(1 for e in embeddings if e is not None)
    print(f"\nSuccessfully embedded: {successful}/{len(texts_to_embed)}")
    
    # Save cache
    embedding_manager.save_cache()
    
    return embeddings


def run_inference(
    questions: List[Question], 
    use_large: bool = False,
    use_oss: bool = False,
    limit: Optional[int] = None,
    name_option: Optional[str] = "",
    use_react: bool = False,

    use_cot: bool = False,
    use_rag: bool = False
):
    """
    Step 3: Run inference using LLM.
    
    Args:
        questions: List of Question objects
        use_large: Whether to use large model
        use_oss: Whether to use OSS 20B model
        limit: Limit number of questions (for testing)
        name_option: Name option for results file
        use_react: Whether to use ReAct agent

        use_cot: Whether to use Chain of Thought prompting
        use_rag: Whether to use RAG
    """
    # Set RAG config based on argument
    pipeline_config.ENABLE_RAG = use_rag
    print(f"\n" + "=" * 60)
    print(f"Step 3: Running Inference")
    print(f"=" * 60)
    
    # Create LLM instance with appropriate options
    from inference import LLMInference
    
    if use_oss:
        model_name = "oss20b"
    elif use_large:
        model_name = "large"
    else:
        model_name = "small"
    
    llm = LLMInference(use_large=use_large, use_oss=use_oss, use_react=use_react, use_cot=use_cot)
    
    print(f"Using model: {model_name}")
    
    # Checkpointing: Load existing results
    output_dir = Path(pipeline_config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"results_{model_name}_{name_option}.json"
    
    existing_results = []
    processed_qids = set()
    
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            processed_qids = {r['qid'] for r in existing_results}
            print(f"Loaded {len(existing_results)} existing results from checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            existing_results = []
    
    results = existing_results
    
    # Filter questions to process
    if limit:
        questions = questions[:limit]
        print(f"Limited to first {limit} questions")
        
    questions_to_process = [q for q in questions if q.qid not in processed_qids]
    
    if not questions_to_process:
        print("All questions already processed!")
        return results

    print(f"Processing {len(questions_to_process)} questions (skipping {len(processed_qids)} already done)...")
    
    correct = sum(1 for r in existing_results if r.get('correct'))
    total_with_answer = sum(1 for r in existing_results if r.get('ground_truth'))
    
    try:
        for i, q in enumerate(questions_to_process):
            # i starts from 0 for this batch
            current_idx = len(existing_results) + i + 1
            print(f"\n[{current_idx}/{len(questions)}] {q.qid}")
            
            # Get prediction
            predicted = llm.answer_question(q)
            
            result = {
                "qid": q.qid,
                "predicted": predicted,
                "ground_truth": q.answer,
            }
            
            # Check correctness
            if q.answer:
                total_with_answer += 1
                is_correct = predicted == q.answer
                if is_correct:
                    correct += 1
                result["correct"] = is_correct
                print(f"  Predicted: {predicted}, Ground truth: {q.answer}, Correct: {is_correct}")
            else:
                print(f"  Predicted: {predicted}")
            
            results.append(result)
            
            # Save checkpoint immediately
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Progress update
            if current_idx % 10 == 0 and total_with_answer > 0:
                acc = correct / total_with_answer * 100
                print(f"\n  --- Progress: {correct}/{total_with_answer} = {acc:.1f}% ---\n")
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Results saved up to this point.")
        return results
    
    # Final accuracy
    if total_with_answer > 0:
        final_acc = correct / total_with_answer * 100
        print(f"\n" + "=" * 60)
        print(f"Final Accuracy: {correct}/{total_with_answer} = {final_acc:.2f}%")
        print(f"=" * 60)
    
    print(f"\nResults saved to {results_file}")
    
    return results


def evaluate_results(results_file: str = None):
    """
    Evaluate prediction results.
    
    Args:
        results_file: Path to results JSON file
    """
    print(f"\n" + "=" * 60)
    print(f"Evaluation")
    print(f"=" * 60)
    
    if results_file is None:
        output_dir = Path(pipeline_config.OUTPUT_DIR)
        results_file = output_dir / "results_small.json"
    else:
        results_file = Path(results_file)
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    correct = 0
    total = 0
    
    for r in results:
        if r.get("ground_truth"):
            total += 1
            if r.get("predicted") == r.get("ground_truth"):
                correct += 1
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    else:
        print("No ground truth available for evaluation")
    
    # Error analysis
    errors = [r for r in results if r.get("correct") == False]
    print(f"\nTotal errors: {len(errors)}")
    
    if errors:
        print("\nSample errors:")
        for e in errors[:5]:
            print(f"  {e['qid']}: Predicted {e['predicted']}, Ground truth {e['ground_truth']}")


def run_full_pipeline(
    dataset: str = "val",
    use_large: bool = False,
    use_oss: bool = False,
    limit: Optional[int] = None,
    name_option: Optional[str] = "",
    use_react: bool = False,

    use_cot: bool = False,
    use_rag: bool = False
):
    """
    Run the full pipeline.
    
    Args:
        dataset: 'val' or 'test'
        use_large: Whether to use large model
        use_oss: Whether to use OSS 20B model
        limit: Limit number of questions
        use_react: Whether to use ReAct agent

        use_cot: Whether to use Chain of Thought prompting
        use_rag: Whether to use RAG
    """
    print("=" * 60)
    print("VNPT AI Validation Pipeline")
    print("=" * 60)
    
    # Step 1: Preprocess
    questions = preprocess_data(dataset)
    
    # Step 2: Embeddings (optional, for future retrieval)
    # create_embeddings(questions)
    
    # Step 3: Inference
    # Step 3: Inference
    results = run_inference(
        questions, 
        use_large=use_large, 
        use_oss=use_oss, 
        limit=limit, 
        name_option=name_option, 
        use_react=use_react, 
        use_cot=use_cot,
        use_rag=use_rag
    )
    
    # Step 4: Evaluate
    if use_oss:
        model_name = "oss20b"
    elif use_large:
        model_name = "large"
    else:
        model_name = "small"
    results_file = Path(pipeline_config.OUTPUT_DIR) / f"results_{model_name}_{name_option}.json"
    evaluate_results(results_file)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VNPT AI Validation Pipeline")
    parser.add_argument(
        "command",
        choices=["preprocess", "embed", "infer", "run", "eval"],
        help="Command to execute"
    )
    parser.add_argument(
        "--dataset",
        choices=["val", "test"],
        default="val",
        help="Dataset to use (default: val)"
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use large model instead of small"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for testing)"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Results file for evaluation"
    )
    parser.add_argument(
        "--name-option",
        type=str,
        default="",
        help="Name option for results file"
    )
    parser.add_argument(
        "--oss",
        action="store_true",
        help="Use OSS 20B model (via ngrok server)"
    )
    parser.add_argument(
        "--react",
        action="store_true",
        help="Use ReAct agent for step-by-step reasoning"
    )
    parser.add_argument(
        "--cot",
        action="store_true",

        help="Use Chain of Thought prompting"
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Use Retrieval Augmented Generation (RAG)"
    )
    
    args = parser.parse_args()
    
    # Check API credentials
    if args.command in ["embed", "infer", "run"]:
        if not (api_config.SMALL_AUTH or api_config.DEFAULT_AUTH):
            print("Warning: API credentials (SMALL_AUTH/DEFAULT_AUTH) not found. Check api-keys.json or .env")
        if not (api_config.SMALL_TOKEN_ID or api_config.DEFAULT_TOKEN_ID):
            print("Warning: Token ID not found")
        if not (api_config.SMALL_TOKEN_KEY or api_config.DEFAULT_TOKEN_KEY):
            print("Warning: Token Key not found")
    
    # Execute command
    if args.command == "preprocess":
        preprocess_data(args.dataset)
    
    elif args.command == "embed":
        questions = load_val_data() if args.dataset == "val" else load_test_data()
        create_embeddings(questions)
    
    elif args.command == "infer":
        questions = load_val_data() if args.dataset == "val" else load_test_data()
        run_inference(
            questions, 
            use_large=args.large, 
            use_oss=args.oss, 
            limit=args.limit, 
            name_option=args.name_option, 
            use_react=args.react, 
            use_cot=args.cot,
            use_rag=args.rag
        )
    
    elif args.command == "run":
        run_full_pipeline(
            dataset=args.dataset,
            use_large=args.large,
            use_oss=args.oss,
            limit=args.limit,
            name_option=args.name_option,
            use_react=args.react,
            use_cot=args.cot,
            use_rag=args.rag
        )
    
    elif args.command == "eval":
        evaluate_results(args.results_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
