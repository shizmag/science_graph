import os
import re
from typing import Dict, Any

class PromptManager:
    """
    Loads and manages LLM prompts from template text files.
    Allows easy retrieval and formatting of prompts.
    """
    def __init__(self, base_dir: str = None) -> None:
        if base_dir is None:
            # By default, look for template files in the directory containing manager.py
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self._cache: Dict[str, str] = {}

    def get_prompt(self, category: str, template_name: str, **kwargs: Any) -> str:
        """
        Retrieves a prompt template and formats it with the provided keyword arguments.
        Uses {{variable}} syntax for formatting to avoid conflicts with JSON syntax braces.
        """
        key = f"{category}/{template_name}"
        if key not in self._cache:
            file_path = os.path.join(self.base_dir, category, f"{template_name}.txt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Prompt template not found at: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                self._cache[key] = f.read()

        template = self._cache[key]
        
        # Render the template using simple {{ var }} replacement
        rendered = template
        for k, v in kwargs.items():
            pattern = r"\{\{\s*" + re.escape(k) + r"\s*\}\}"
            val = str(v)
            rendered = re.sub(pattern, lambda m: val, rendered)
            
        return rendered.strip()
