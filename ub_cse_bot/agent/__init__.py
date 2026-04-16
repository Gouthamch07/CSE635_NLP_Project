__all__ = ["UBCSEAgent", "AgentResponse", "build_tool_registry", "ToolCall", "ToolResult"]


def __getattr__(name: str):
    if name in ("UBCSEAgent", "AgentResponse"):
        from .orchestrator import AgentResponse, UBCSEAgent
        return {"UBCSEAgent": UBCSEAgent, "AgentResponse": AgentResponse}[name]
    if name in ("build_tool_registry", "ToolCall", "ToolResult"):
        from .tools import ToolCall, ToolResult, build_tool_registry
        return {
            "build_tool_registry": build_tool_registry,
            "ToolCall": ToolCall,
            "ToolResult": ToolResult,
        }[name]
    raise AttributeError(name)
