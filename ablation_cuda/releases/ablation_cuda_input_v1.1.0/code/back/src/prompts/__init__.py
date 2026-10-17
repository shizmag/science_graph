from .manager import PromptManager

# Global default prompt manager instance
prompts = PromptManager()

__all__ = ["PromptManager", "prompts"]
