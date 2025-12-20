"""
ReAct Agent Implementation
Reasoning + Acting agent that uses tools to answer questions step by step.
"""

import re
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

from config import pipeline_config
from tools import tool_registry
from llm_wrapper import VNPTLLM

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Represents a single step in the agent's reasoning."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent.
    Uses a chain-of-thought approach with tool usage.
    """
    
    REACT_PROMPT_TEMPLATE = """Bạn là một trợ lý AI thông minh. Hãy trả lời câu hỏi trắc nghiệm bằng cách suy luận từng bước.

Bạn có thể sử dụng các công cụ sau:

{tools}

Sử dụng định dạng sau:

Question: câu hỏi cần trả lời
Thought: suy nghĩ về cách giải quyết
Action: tên công cụ cần sử dụng (một trong [{tool_names}])
Action Input: input cho công cụ
Observation: kết quả từ công cụ
... (có thể lặp lại Thought/Action/Action Input/Observation nhiều lần)
Thought: Tôi đã có đủ thông tin để trả lời
Final Answer: đáp án cuối cùng (chỉ ghi một chữ cái A, B, C, hoặc D)

Lưu ý:
- Nếu câu hỏi yêu cầu tính toán, hãy dùng Calculator
- Nếu không cần công cụ, có thể đưa ra Final Answer trực tiếp
- Đáp án cuối cùng PHẢI là một chữ cái duy nhất (A, B, C, hoặc D)

{context}

Question: {question}

Các lựa chọn:
{choices}

Thought:"""

    def __init__(
        self, 
        llm: Optional[VNPTLLM] = None, 
        use_large: bool = True,
        use_oss: bool = False,
        max_steps: int = 5
    ):
        """
        Initialize ReAct Agent.
        
        Args:
            llm: LLM instance to use. If None, creates a new one.
            use_large: Whether to use large model (only if llm is None)
            use_oss: Whether to use OSS model via ngrok (only if llm is None)
            max_steps: Maximum reasoning steps before giving up
        """
        if llm is None:
            # Create LLM with stop sequence to pause at Observation
            self.llm = VNPTLLM(
                use_large=use_large,
                use_oss=use_oss,
                stop=["Observation:"]
            )
        else:
            self.llm = llm
            
        self.max_steps = max_steps
        self.steps: List[AgentStep] = []
    
    def _build_prompt(
        self, 
        question: str, 
        choices: str,
        context: Optional[str] = None
    ) -> str:
        """Build the initial prompt for the agent."""
        context_str = ""
        if context:
            context_str = f"Đoạn thông tin:\n{context}\n"
        
        return self.REACT_PROMPT_TEMPLATE.format(
            tools=tool_registry.get_tools_description(),
            tool_names=", ".join(tool_registry.get_tool_names()),
            context=context_str,
            question=question,
            choices=choices
        )
    
    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse Action and Action Input from LLM output.
        
        Returns:
            Tuple of (action_name, action_input) or (None, None)
        """
        # Pattern: Action: <name>\nAction Input: <input>
        action_match = re.search(
            r'Action:\s*(.+?)(?:\n|$)', 
            text, 
            re.IGNORECASE
        )
        input_match = re.search(
            r'Action Input:\s*(.+?)(?:\n|$)', 
            text, 
            re.IGNORECASE | re.DOTALL
        )
        
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            # Clean up action input - remove quotes and extra whitespace
            action_input = action_input.split('\n')[0].strip()
            # Remove surrounding quotes if present
            action_input = action_input.strip("'\"")
            return action, action_input
        
        return None, None
    
    def _parse_final_answer(self, text: str) -> Optional[str]:
        """
        Parse Final Answer from LLM output.
        
        Returns:
            The answer letter (A-D) or None
        """
        # Pattern: Final Answer: <answer>
        match = re.search(
            r'Final Answer:\s*([A-D])', 
            text, 
            re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        
        # Try to find just a letter if "Final Answer:" is present
        if "final answer" in text.lower():
            match = re.search(r'\b([A-D])\b', text[text.lower().find("final answer"):])
            if match:
                return match.group(1).upper()
        
        return None
    
    def _extract_thought(self, text: str) -> str:
        """Extract the Thought portion from LLM output."""
        # Find content before Action or Final Answer
        thought = text
        
        for marker in ["Action:", "Final Answer:"]:
            if marker in thought:
                thought = thought.split(marker)[0]
        
        return thought.strip()
    
    def answer(
        self, 
        question: str, 
        choices: str,
        context: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        Answer a question using ReAct reasoning.
        
        Args:
            question: The question text
            choices: Formatted choices string
            context: Optional context/passage
            verbose: Whether to print reasoning steps
            
        Returns:
            The answer letter (A, B, C, or D)
        """
        self.steps = []
        prompt = self._build_prompt(question, choices, context)
        
        if verbose:
            print("=" * 60)
            print("ReAct Agent Starting")
            print("=" * 60)
        
        for step_num in range(self.max_steps):
            if verbose:
                print(f"\n--- Step {step_num + 1} ---")
            
            try:
                # Call LLM
                response = self.llm.invoke(prompt)
                
                if verbose:
                    print(f"LLM Output:\n{response}")
                
            except Exception as e:
                logger.error(f"LLM call failed at step {step_num + 1}: {e}")
                return "A"  # Fallback
            
            # Check for Final Answer first
            final_answer = self._parse_final_answer(response)
            if final_answer:
                if verbose:
                    print(f"\nFinal Answer: {final_answer}")
                return final_answer
            
            # Parse Action
            action, action_input = self._parse_action(response)
            
            if action and action_input:
                # Execute tool
                if verbose:
                    print(f"\nExecuting: {action}({action_input})")
                
                observation = tool_registry.execute(action, action_input)
                
                if verbose:
                    print(f"Observation: {observation}")
                
                # Record step
                self.steps.append(AgentStep(
                    thought=self._extract_thought(response),
                    action=action,
                    action_input=action_input,
                    observation=observation
                ))
                
                # Append to prompt for next iteration
                prompt += response + f"\nObservation: {observation}\nThought:"
                
            else:
                # No action found, try to get answer from response
                # Maybe model just reasoned without using tools
                
                # Look for any answer pattern in response
                match = re.search(r'(?:đáp án|answer)[:\s]*([A-D])', response, re.IGNORECASE)
                if match:
                    return match.group(1).upper()
                
                # Just a bare letter?
                match = re.search(r'\b([A-D])\b', response)
                if match:
                    return match.group(1).upper()
                
                # Continue prompting
                prompt += response + "\nThought:"
        
        # Max steps reached - try to extract any answer
        logger.warning("ReAct agent reached max steps without final answer")
        
        # Look through steps for any indication
        full_text = prompt
        match = re.search(r'\b([A-D])\b', full_text[-500:])  # Check last 500 chars
        if match:
            return match.group(1).upper()
        
        return "A"  # Default fallback

    def get_reasoning_trace(self) -> str:
        """Get a formatted trace of the agent's reasoning."""
        trace = []
        for i, step in enumerate(self.steps):
            trace.append(f"Step {i+1}:")
            trace.append(f"  Thought: {step.thought[:100]}...")
            if step.action:
                trace.append(f"  Action: {step.action}")
                trace.append(f"  Action Input: {step.action_input}")
                trace.append(f"  Observation: {step.observation}")
        return "\n".join(trace)


# Convenience function
def create_react_agent(use_large: bool = True, max_steps: int = 5) -> ReActAgent:
    """Create a new ReAct agent instance."""
    return ReActAgent(use_large=use_large, max_steps=max_steps)


if __name__ == "__main__":
    # Test ReAct Agent
    print("Testing ReAct Agent...")
    
    agent = create_react_agent(use_large=True)
    
    # Test question requiring calculation
    question = "Nếu căn bậc hai của 144 cộng với 50 thì kết quả là bao nhiêu?"
    choices = "A. 52\nB. 62\nC. 72\nD. 82"
    
    print(f"\nQuestion: {question}")
    print(f"Choices:\n{choices}")
    
    answer = agent.answer(question, choices, verbose=True)
    print(f"\n=== Final Answer: {answer} ===")
    
    print("\n\nReasoning Trace:")
    print(agent.get_reasoning_trace())
