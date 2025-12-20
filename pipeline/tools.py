"""
Tool System for ReAct Agent
Provides a registry of tools that the agent can use for reasoning.
"""

from typing import Callable, List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from math_tool import math_tool

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a tool that the agent can use."""
    name: str
    description: str
    func: Callable[[str], str]
    
    def execute(self, input_str: str) -> str:
        """Execute the tool with the given input."""
        try:
            result = self.func(input_str)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return f"Error: {str(e)}"


class ToolRegistry:
    """Registry of available tools for the agent."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools."""
        
        # Calculator tool
        self.register(Tool(
            name="Calculator",
            description="Tính toán biểu thức toán học. Input là một biểu thức Python hợp lệ. Ví dụ: 'sqrt(144) + 50' hoặc '(100 - 80) / 100'",
            func=lambda expr: math_tool.calculate(expr)
        ))
        
        # Context Analyzer tool  
        self.register(Tool(
            name="ContextAnalyzer",
            description="Phân tích và trích xuất thông tin từ đoạn văn được cung cấp. Input là câu hỏi cần tìm trong context.",
            func=self._analyze_context
        ))
    
    def _analyze_context(self, query: str) -> str:
        """Analyze context - this is a placeholder that will be enhanced."""
        return f"Đang tìm kiếm thông tin về: {query}"
    
    def register(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def execute(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
        return tool.execute(tool_input)
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"{name}: {tool.description}")
        return "\n".join(descriptions)
    
    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return list(self.tools.keys())


# Global tool registry
tool_registry = ToolRegistry()


def register_rag_tool(retrieve_func: Callable[[str, int], str]):
    """
    Register RAG search tool with custom retrieve function.
    This should be called after RAG engine is initialized.
    
    Args:
        retrieve_func: Function that takes (query, top_k) and returns context string
    """
    def rag_search(query: str) -> str:
        try:
            context = retrieve_func(query, 3)
            if context:
                return context
            return "Không tìm thấy thông tin liên quan."
        except Exception as e:
            return f"Lỗi tìm kiếm: {str(e)}"
    
    tool_registry.register(Tool(
        name="RAGSearch",
        description="Tìm kiếm thông tin từ knowledge base. Input là câu hỏi hoặc từ khóa cần tìm.",
        func=rag_search
    ))
    logger.info("RAGSearch tool registered")


if __name__ == "__main__":
    # Test tools
    print("Testing Tool Registry...")
    
    print("\n=== Available Tools ===")
    print(tool_registry.get_tools_description())
    
    print("\n=== Test Calculator ===")
    result = tool_registry.execute("Calculator", "sqrt(144) + 50")
    print(f"sqrt(144) + 50 = {result}")
    
    result = tool_registry.execute("Calculator", "(100 - 80) / 100")
    print(f"(100 - 80) / 100 = {result}")
    
    print("\n=== Test Unknown Tool ===")
    result = tool_registry.execute("UnknownTool", "test")
    print(result)
