"""
Math Tool for VNPT AI Pipeline
Provides safe evaluation of mathematical expressions.
"""

import math
import re
from typing import Union, Optional

class MathTool:
    """
    A tool for evaluating mathematical expressions.
    Can be used by the LLM or as a standalone utility.
    """
    
    # Allowed functions and constants
    ALLOWED_NAMES = {
        k: v for k, v in math.__dict__.items() 
        if not k.startswith("__")
    }
    ALLOWED_NAMES.update({
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
    })
    
    def __init__(self):
        pass
    
    def calculate(self, expression: str) -> Union[float, int, str]:
        """
        Evaluate a mathematical expression safely.
        
        Args:
            expression: The math string to evaluate (e.g., "2 + 2", "sqrt(16)")
            
        Returns:
            The result as a number, or an error message string.
        """
        # Clean the expression
        expression = expression.strip()
        
        # Remove markdown code blocks if present
        expression = expression.replace("`", "")
        
        try:
            # Compile the code to check for unsafe operations
            code = compile(expression, "<string>", "eval")
            
            # Check for unsafe names in the code object
            for name in code.co_names:
                if name not in self.ALLOWED_NAMES:
                    return f"Error: Forbidden function or variable '{name}'"
            
            # Evaluate
            result = eval(code, {"__builtins__": {}}, self.ALLOWED_NAMES)
            return result
            
        except Exception as e:
            return f"Error: {str(e)}"

    def process_markdown(self, text: str) -> str:
        """
        Process markdown text and evaluate math expressions inside specific delimiters.
        Supported delimiters:
        - `{{ expression }}` -> result
        - `$$ expression $$` -> result (if it's a calculation)
        """
        
        def replace_match(match):
            expr = match.group(1)
            result = self.calculate(expr)
            return str(result)
        
        # Replace {{ ... }}
        text = re.sub(r'\{\{(.+?)\}\}', replace_match, text)
        
        return text

# Global instance
math_tool = MathTool()

if __name__ == "__main__":
    # Test
    tool = MathTool()
    print(tool.calculate("2 + 2"))
    print(tool.calculate("sqrt(16) * 2"))
    print(tool.calculate("pow(2, 3)"))
    print(tool.process_markdown("The result of 5 * 5 is {{ 5 * 5 }}."))
