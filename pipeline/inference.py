# Inference Module - LLM API for Question Answering
import time
import re
import json
import logging
from typing import List, Dict, Optional, Any

from config import api_config, pipeline_config
from data_loader import Question
from llm_wrapper import VNPTLLM
from react_agent import ReActAgent
import rag  # Import RAG module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInference:
    """
    Handles interactions with the VNPT AI LLM API using LangChain.
    Supports optional ReAct agent for step-by-step reasoning.
    """
    
    def __init__(self, use_large: bool = False, use_oss: bool = False, use_react: bool = None, use_cot: bool = None):
        self.use_large = use_large
        self.use_oss = use_oss
        self.llm = VNPTLLM(use_large=use_large, use_oss=use_oss)
        
        # Initialize ReAct agent if enabled
        if use_react is None:
            use_react = pipeline_config.USE_REACT_AGENT
        
        self.use_react = use_react
        self.react_agent = None
        
        # Chain of Thought setting
        if use_cot is None:
            use_cot = pipeline_config.USE_CHAIN_OF_THOUGHT
        self.use_cot = use_cot
        
        model_type = "OSS 20B" if use_oss else ("large" if use_large else "small")
        
        if self.use_react:
            self.react_agent = ReActAgent(
                use_large=use_large,
                use_oss=use_oss,
                max_steps=pipeline_config.REACT_MAX_STEPS
            )
            logger.info(f"ReAct agent initialized (model={model_type})")
        elif self.use_cot:
            logger.info(f"Chain of Thought prompting enabled (model={model_type})")
        else:
            logger.info(f"Direct prompting mode (model={model_type})")
    
    def create_prompt(
        self, 
        question: Question, 
        additional_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for the question.
        """
        prompt_parts = []
        
        # Add context if available
        if question.has_context():
            prompt_parts.append(f"Đoạn thông tin:\n{question.context}\n")
        elif additional_context:
            prompt_parts.append(f"Đoạn thông tin:\n{additional_context}\n")
        
        # Add question
        if question.raw_question:
            prompt_parts.append(f"Câu hỏi: {question.raw_question}")
        else:
            prompt_parts.append(f"Câu hỏi: {question.question}")
        
        # Add choices
        prompt_parts.append(f"\nCác lựa chọn:\n{question.format_choices()}")
        
        # Add instruction
        prompt_parts.append("\nHãy chọn đáp án đúng nhất. Chỉ trả lời bằng chữ cái của đáp án (A, B, C, D, ...).")
        
        return "\n".join(prompt_parts)
    
    def extract_answer(self, response: str) -> str:
        """
        Extract answer letter from LLM response.
        """
        if not response:
            return "A"  # Default fallback
        
        # Clean response
        response = response.strip().upper()
        
        # Try to find a single letter answer
        # Pattern 1: Just the letter
        if len(response) == 1 and response in 'ABCDEFGHIJ':
            return response
        
        # Pattern 2: "Đáp án: X" or "Đáp án là X" - Look for LAST occurrence
        # This is safer for CoT where "Đáp án A" might appear in reasoning ("Giải thích vì sao Đáp án A sai...")
        # Updated to handle markdown like "**B**" -> \W matches non-word chars including *
        # Also handles "Đáp án đúng là..." or "Đáp án cần chọn là..."
        matches = list(re.finditer(r'ĐÁP ÁN(?:.*?LÀ)?\W*([A-E])', response, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
        
        # Pattern 3: "X." at the beginning (Direct answering)
        # Check that it's just the letter and some punctuation/space
        match = re.match(r'^([A-E])[\.\)\s]', response)
        if match:
            return match.group(1).upper()
            
        # Pattern 4: Last occurrence of a single letter line? 
        # Or look for [A-E] at the very end
        match = re.search(r'([A-E])\s*$', response)
        if match:
            return match.group(1).upper()

        # DANGEROUS FALLBACK REMOVED: 
        # Determining answer by finding first single letter is prone to errors ("I", "A", etc in text)
        
        # Default fallback
        return "A"

    def _find_refusal_option(self, question: Question) -> Optional[str]:
        """
        Find choice that represents a refusal to answer due to safety/policy.
        """
        refusal_keywords = [
            "tôi không thể trả lời",
            "tôi không thể cung cấp",
            "từ chối trả lời",
            "không thể trả lời câu hỏi",
            "tôi không thể hỗ trợ",
            "xin lỗi, tôi không thể"
        ]
        
        import string
        letters = string.ascii_uppercase
        
        for i, choice in enumerate(question.choices):
            choice_lower = choice.lower()
            if any(k in choice_lower for k in refusal_keywords):
                if i < len(letters):
                    return letters[i]
        return None

    def answer_question(
        self, 
        question: Question, 
        additional_context: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        Answer a question using the LLM.
        If ReAct agent is enabled, uses step-by-step reasoning with tools.
        """
        # Determine context
        context_content = ""
        if question.has_context():
            context_content = question.context
        elif additional_context:
            context_content = additional_context
        elif pipeline_config.ENABLE_RAG:
            # Try RAG if no context provided using global function
            logger.info(f"No context for {question.qid}. Retrieving from knowledge base...")
            try:
                # Use raw question for retrieval
                query = question.raw_question if question.raw_question else question.question
                retrieved_context = rag.retrieve_context(query)
                if retrieved_context:
                    logger.info(f"Retrieved context for {question.qid} (len={len(retrieved_context)})")
                    context_content = retrieved_context
                else:
                    logger.warning(f"No context retrieved for {question.qid}")
            except Exception as e:
                logger.error(f"RAG retrieval failed for {question.qid}: {e}")
            
        # Truncate or Refine context if too long
        # Safe estimate: 1 token ~ 3.5 chars
        max_chars = int(pipeline_config.MAX_INPUT_TOKENS * 3.5)
        
        # Refinement Logic
        if pipeline_config.REFINE_CONTEXT and len(context_content) > pipeline_config.REFINE_CONTEXT_THRESHOLD:
            logger.info(f"Refining context for {question.qid} (Length: {len(context_content)} chars)")
            try:
                from embedding_wrapper import VNPTEmbeddings
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_community.vectorstores import FAISS
                
                # Split context
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=pipeline_config.REFINE_CHUNK_SIZE,
                    chunk_overlap=pipeline_config.REFINE_CHUNK_OVERLAP
                )
                chunks = splitter.split_text(context_content)
                logger.info(f"Split context into {len(chunks)} chunks")
                
                if len(chunks) > 1:
                    # Embed and retrieve
                    embeddings = VNPTEmbeddings()
                    
                    # Create metadatas with original index
                    metadatas = [{"index": i} for i in range(len(chunks))]
                    
                    vectorstore = FAISS.from_texts(
                        texts=chunks, 
                        embedding=embeddings,
                        metadatas=metadatas
                    )
                    
                    # Query is the question itself (and choices?)
                    query_text = question.raw_question if question.raw_question else question.question
                    # Adding choices might help if they contain keywords
                    query_text += f"\n{question.format_choices()}"
                    
                    # Get results
                    retrieved_docs = vectorstore.similarity_search(
                        query_text, 
                        k=min(len(chunks), pipeline_config.REFINE_TOP_K)
                    )
                    
                    # Sort by original index to maintain narrative flow
                    retrieved_docs.sort(key=lambda x: x.metadata.get("index", 0))
                    
                    refined_context = "\n\n...\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    logger.info(f"Refined context length: {len(refined_context)} chars")
                    context_content = refined_context
                    
            except Exception as e:
                logger.error(f"Context refinement failed for {question.qid}: {e}")
                # Fallback to truncation if refinement fails
        
        # Final safety truncation
        if len(context_content) > max_chars:
            logger.warning(f"Truncating context for {question.qid} from {len(context_content)} to {max_chars} chars")
            context_content = context_content[:max_chars] + "...(truncated)"
        
        # Get question text
        question_text = question.raw_question if question.raw_question else question.question
        choices_str = question.format_choices()
        
        # Use ReAct agent if enabled
        if self.react_agent:
            try:
                answer = self.react_agent.answer(
                    question=question_text,
                    choices=choices_str,
                    context=context_content if context_content else None,
                    verbose=verbose
                )
                return answer
            except Exception as e:
                logger.error(f"ReAct agent failed for {question.qid}: {e}")
                logger.info("Falling back to direct LLM call")
                # Fall through to direct LLM call
            
        # Direct LLM call (original method)
        context_str = ""
        if context_content:
            context_str = f"Đoạn thông tin:\n{context_content}\n"

        # Construct choices string
        choices_str = question.format_choices()
        
        # Get question text
        question_text = question.raw_question if question.raw_question else question.question

        # Combine system instruction and user prompt into one string
        # CHANGED: Context FIRST, then Question
        
        # Check if we should use Chain of Thought
        use_cot_for_this_request = self.use_cot
        if pipeline_config.AUTO_COT_FOR_CONTEXT and (question.has_context() or additional_context or pipeline_config.ENABLE_RAG):
            # Also auto-enable CoT if RAG is used (implied by ENABLE_RAG check above + auto flag)
            use_cot_for_this_request = True
            
        if use_cot_for_this_request:
            # Chain of Thought prompting - explicit step-by-step reasoning
            full_prompt = f"""Bạn là một trợ lý AI thông minh. Hãy trả lời câu hỏi trắc nghiệm bằng cách suy luận từng bước.

HƯỚNG DẪN TÍNH TOÁN:
Nếu cần thực hiện tính toán toán học, hãy viết biểu thức trong cặp dấu ngoặc nhọn đôi. Hệ thống sẽ tự động tính toán cho bạn.
Ví dụ: "Độ co giãn là {{ (80 - 100) / 100 }}." sẽ được hiển thị là "Độ co giãn là -0.2."
Hỗ trợ các phép tính: +, -, *, /, pow, sqrt, abs, round, min, max.

{context_str}

Câu hỏi: {question_text}

Các lựa chọn:
{choices_str}

HÃY SUY LUẬN TỪNG BƯỚC (Chain of Thought):

Bước 1: Phân tích yêu cầu câu hỏi.
Bước 2: (Nếu có đoạn thông tin) Tìm chi tiết liên quan trong đoạn thông tin. Trích dẫn ngắn gọn nếu cần.
Bước 3: Phân tích từng lựa chọn A, B, C, D.
Bước 4: Loại trừ phương án sai và xác định phương án đúng.
Bước 5: Kết luận.

Định dạng câu trả lời:
Suy luận: [Viết đầy đủ các bước suy luận]
Đáp án: [Chỉ ghi duy nhất một chữ cái in hoa (A, B, C, hoặc D) không kèm ký tự đặc biệt]"""
        else:
            # Direct prompting (default)
            full_prompt = f"""Bạn là một trợ lý AI thông minh. Hãy trả lời câu hỏi trắc nghiệm một cách chính xác dựa trên thông tin được cung cấp.

{context_str}

Câu hỏi: {question_text}

Các lựa chọn:
{choices_str}

Định dạng câu trả lời bắt buộc:
Giải thích: [Giải thích ngắn gọn]
Đáp án: [Chỉ ghi duy nhất một chữ cái in hoa (A, B, C, hoặc D) không kèm ký tự đặc biệt]"""
        
        try:
            # Call LangChain LLM Wrapper
            response = self.llm.invoke(full_prompt)
            
            # Checks for refusal/failure cases (empty response usually implies 400 Bad Request / Content Policy violation)
            if not response:
                logger.warning(f"Empty response from API (likely content policy violation). Checking for refusal answer for {question.qid}...")
                refusal_answer = self._find_refusal_option(question)
                if refusal_answer:
                    logger.info(f"Found refusal answer: {refusal_answer}")
                    return refusal_answer
                logger.warning("No refusal answer found. Defaulting to A.")

            answer = self.extract_answer(response)
            return answer
        except Exception as e:
            logger.error(f"Error answering question {question.qid}: {e}")
            
            # Try to recover if it's a refusal case
            refusal_answer = self._find_refusal_option(question)
            if refusal_answer:
                logger.info(f"Found refusal answer after error: {refusal_answer}")
                return refusal_answer
                
            return "A"
    
    def answer_questions_batch(
        self, 
        questions: List[Question],
        show_progress: bool = True
    ) -> List[str]:
        """
        Answer multiple questions.
        """
        results = []
        total = len(questions)
        
        for i, q in enumerate(questions):
            if show_progress:
                print(f"Processing {i + 1}/{total}: {q.qid}")
            
            answer = self.answer_question(q)
            results.append(answer)
            
            if show_progress and (i + 1) % 10 == 0:
                correct = sum(1 for j, a in enumerate(results) 
                            if questions[j].answer and a == questions[j].answer)
                print(f"  Current accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
        
        return results


# Create global inference instances
llm_small = LLMInference(use_large=False)
llm_large = LLMInference(use_large=True)
llm_oss = LLMInference(use_oss=True)


if __name__ == "__main__":
    # Test inference
    from data_loader import load_val_data
    
    print("Testing LLM inference with LangChain...")
    
    questions = load_val_data()
    
    if questions:
        q = questions[10]

        q.context = """
        Có các liên đoàn kinh tế tại Cuba, với số thành viên lên tới 98% nhân lực hòn đảo này. Các liên đoàn không đăng ký với bất kỳ một cơ quan nhà nước nào, và tự lấy kinh phí hoạt động từ nguồn đóng góp thành viên hàng tháng. Những người ủng hộ họ cho rằng các viên chức liên đoàn được bầu lên trên cơ sở tự do, và có nhiều quan điểm chính trị bên trong mỗi liên đoàn. Tuy nhiên, tất cả các liên đoàn đều là một phần của một tổ chức được gọi là Confederación de Trabajadores Cubanos (Hiệp hội Công nhân Cuba, CTC), hội này thực sự có những mối quan hệ mật thiết với nhà nước và Đảng Cộng sản. Những người ủng hộ cho rằng CTC cho phép công nhân chuyển ý kiến lên chính phủ
        """
        # print(f"\nQuestion: {q.qid}")
        print(f"Has context: {q.has_context()}")
        print(f"Ground truth: {q.answer}")
        print("Choise: ", q.format_choices())
        
        # Test with small model
        print("\nTesting with small model...")
        answer = llm_small.answer_question(q)
        print(f"Predicted: {answer}")
        print(f"Correct: {answer == q.answer}")
