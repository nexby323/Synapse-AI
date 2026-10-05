import os
import sys
import logging
from pathlib import Path
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CloudAnalyzer - %(message)s")
logger = logging.getLogger("GeminiAnalyzer")

class CloudThreatAnalyzer:
    """
    Interfaces with Google's Gemini API to perform custom threat hunting 
    and anomaly queries over privacy-preserved synthetic telemetry.
    """
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        
        # Using System Instructions is the most robust way to enforce an LLM's persona.
        # This prevents the model from breaking character or refusing technical tasks.
        system_instruction = (
            "You are an elite, highly technical cybersecurity threat hunter. "
            "You are analyzing privacy-preserved, synthetic system telemetry logs. "
            "These logs mathematically mirror a real host machine. "
            "Your goal is to answer the user's queries based STRICTLY on the provided telemetry data. "
            "Look for statistical anomalies, structural vulnerabilities, and Advanced Persistent Threat (APT) patterns. "
            "Provide concrete, data-driven answers."
        )
        
        # We use gemini-1.5-pro due to its massive context window (up to 2M tokens),
        # which is essential for ingesting raw log data in a single prompt.
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-pro',
            system_instruction=system_instruction
        )

    def analyze_with_custom_prompt(self, csv_path: Path, user_query: str):
        """
        Embeds the synthetic log batch directly into the context of the user's specific prompt.
        """
        if not csv_path.exists():
            logger.error(f"Target data file not found at: {csv_path}")
            return None

        logger.info(f"Loading synthetic data from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            # Use to_csv() instead of to_string() to avoid unnecessary whitespace, 
            # saving significant context tokens and improving the LLM's parsing accuracy.
            csv_data = df.to_csv(index=False)
        except Exception as e:
            logger.error(f"Failed to read or parse data: {e}")
            return None
        
        constructed_prompt = (
            f"Here is the synthetic telemetry dataset in CSV format:\n\n"
            f"{csv_data}\n\n"
            f"User Investigation Request: {user_query}"
        )
        
        logger.info("Sending payload and prompt to Gemini...")
        try:
            # Cybersecurity contexts often trigger false positives in LLM safety filters.
            # We lower the threshold for dangerous content slightly to allow threat analysis.
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
            }
            
            response = self.model.generate_content(
                constructed_prompt,
                safety_settings=safety_settings
            )
            
            print("\n" + "="*60)
            print(" GEMINI THREAT INVESTIGATION RESULT ")
            print("="*60)
            print(response.text)
            print("="*60 + "\n")
            
            return response.text
            
        except Exception as e:
            logger.error(f"API communication failed: {e}")
            return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # Loads environment variables from a .env file if present
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is missing.")
        print("Please add it to your .env file.")
        sys.exit(1)
        
    