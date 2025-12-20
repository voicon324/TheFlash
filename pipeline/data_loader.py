# Data Loader Module
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

from config import pipeline_config


@dataclass
class Question:
    """Represents a single question from the dataset"""
    qid: str
    question: str
    choices: List[str]
    answer: Optional[str] = None  # Ground truth answer (A, B, C, D, ...)
    context: Optional[str] = None  # Extracted context if present
    raw_question: Optional[str] = None  # Question without context
    
    def get_choice_text(self, choice_letter: str) -> str:
        """Get the text of a choice by its letter (A, B, C, ...)"""
        idx = ord(choice_letter.upper()) - ord('A')
        if 0 <= idx < len(self.choices):
            return self.choices[idx]
        return ""
    
    def format_choices(self) -> str:
        """Format choices as A. choice1, B. choice2, etc."""
        import string
        letters = string.ascii_uppercase
        return '\n'.join(
            f"{letters[i]}. {choice}" 
            for i, choice in enumerate(self.choices)
            if i < len(letters)
        )
    
    def has_context(self) -> bool:
        """Check if question has embedded context"""
        return self.context is not None and len(self.context) > 0


def extract_context_and_question(text: str) -> tuple:
    """
    Extract context and question from text.
    Context is usually prefixed with "Đoạn thông tin:" or similar.
    
    Returns:
        (context, question) tuple
    """
    # Patterns for context extraction
    context_patterns = [
        r"Đoạn thông tin[:\s]*\n?(.*?)(?=Câu hỏi[:\s]*)",
        r"\[1\] Tiêu đề:.*?(?=Câu hỏi[:\s]*)",
        r"-- Đoạn văn \d+ --.*?(?=Câu hỏi[:\s]*)",
        r"-- Document \d+ --.*?(?=Câu hỏi[:\s]*)",
        r"Title:.*?(?=Câu hỏi[:\s]*)",
    ]
    
    # Try to extract context
    context = None
    question = text
    
    # Check if text starts with context indicator
    if any(text.startswith(indicator) for indicator in ["Đoạn thông tin", "[1]", "-- Đoạn văn", "-- Document", "Title:"]):
        # Find "Câu hỏi:" to split
        # We look for the LAST occurrence to avoid issues where "Câu hỏi" appears in the context
        # We also enforce a colon to avoid matching phrases like "Câu hỏi này"
        matches = list(re.finditer(r"Câu hỏi:\s*", text))
        if matches:
            last_match = matches[-1]
            question = text[last_match.end():].strip()
            context = text[:last_match.start()].strip()
    
    return context, question


def load_questions(file_path: str) -> List[Question]:
    """
    Load questions from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of Question objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        qid = item.get('qid', '')
        question_text = item.get('question', '')
        choices = item.get('choices', [])
        answer = item.get('answer', None)
        
        # Extract context if present
        context, raw_question = extract_context_and_question(question_text)
        
        q = Question(
            qid=qid,
            question=question_text,
            choices=choices,
            answer=answer,
            context=context,
            raw_question=raw_question
        )
        questions.append(q)
    
    return questions


def load_val_data() -> List[Question]:
    """Load validation dataset"""
    file_path = Path(pipeline_config.DATA_DIR) / pipeline_config.VAL_FILE
    return load_questions(str(file_path))


def load_test_data() -> List[Question]:
    """Load test dataset"""
    file_path = Path(pipeline_config.DATA_DIR) / pipeline_config.TEST_FILE
    return load_questions(str(file_path))


def save_results(results: List[dict], output_file: str = None):
    """
    Save prediction results to JSON file.
    
    Args:
        results: List of dicts with qid and predicted_answer
        output_file: Output file path (default: from config)
    """
    if output_file is None:
        output_path = Path(pipeline_config.OUTPUT_DIR) / pipeline_config.RESULTS_FILE
    else:
        output_path = Path(output_file)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Test loading
    print("Loading validation data...")
    questions = load_val_data()
    print(f"Loaded {len(questions)} questions")
    
    # Show sample
    if questions:
        q = questions[0]
        print(f"\nSample question:")
        print(f"QID: {q.qid}")
        print(f"Has context: {q.has_context()}")
        print(f"Answer: {q.answer}")
        print(f"Choices: {q.format_choices()[:200]}...")
    
    # Statistics
    with_context = sum(1 for q in questions if q.has_context())
    print(f"\nStatistics:")
    print(f"Questions with context: {with_context}")
    print(f"Questions without context: {len(questions) - with_context}")
