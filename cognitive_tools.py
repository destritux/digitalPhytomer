from abc import ABC, abstractmethod

class CognitiveTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, arguments: dict) -> dict:
        """Executes the tool's logic and returns structured output."""
        pass

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: CognitiveTool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> CognitiveTool:
        return self.tools.get(name)

    def generate_system_prompt_declaration(self) -> str:
        if not self.tools:
            return ""
        declaration = "\nAvailable Tools:\n"
        for t in self.tools.values():
            declaration += f"- {t.name}: {t.description}\n"
        return declaration
