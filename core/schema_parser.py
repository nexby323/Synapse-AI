import re
import requests
import logging
from typing import Optional

logger = logging.getLogger("LLMCoder")

class LLMCodeGenerator:
    """
    Interfaces with a local LLM to translate natural language tasks 
    into executable Python scripts for the sandboxed agent.
    """
    def __init__(self, model_name: str = "llama3"):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        
        #this prompt enforces the rules of the sandbox.
        self.system_prompt = """
        You are an expert Python cybersecurity developer.
        The user will provide a monitoring or data collection task.
        Write a standalone Python script to perform this task.
        
        CRITICAL RULES:
        1. Output ONLY valid Python code. Do NOT output markdown, backticks, or explanations.
        2. You may use standard libraries and 'psutil' only.
        3. You MUST write your final output to a CSV file located exactly at: '/app/logs/collected_data.csv'.
        4. The script should run a short loop (e.g., 5 iterations with a 1-second sleep), write the data, and exit cleanly.
        """

    def generate_agent_code(self, user_prompt: str) -> Optional[str]:
        """
        Sends the task to the LLM and extracts the generated Python code.
        """
        payload = {
            "model": self.model_name,
            "prompt": f"{self.system_prompt}\n\nTask: {user_prompt}\n\n# Python Code:\n",
            "stream": False,
            "options": {"temperature": 0.1} # Low temperature for logical, predictable code
        }

        try:
            logger.info("Requesting agent code generation from local LLM...")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            raw_output = response.json().get("response", "")
            return self._clean_llm_output(raw_output)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with LLM: {e}")
            return None
    #TODO: actually check the code itself 
    def _clean_llm_output(self, text: str) -> str:
        """
        Strips markdown formatting if the LLM disobeys the prompt instructions.
        """
        # Remove ```python and ``` blocks
        text = re.sub(r'```python', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```', '', text)
        return text.strip()