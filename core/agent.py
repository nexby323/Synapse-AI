import logging
import sys
from schema_parser import LLMCodeGenerator
from agent_factory import DynamicAgentFactory

#configure basic logging for the orchestrator
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Orchestrator - %(message)s"
)
logger = logging.getLogger("SynapseOrchestrator")

class SynapseOrchestrator:
    """
    The main controller for Synapse-AI. 
    Manages the lifecycle: Prompt -> Code Generation -> Sandboxed Execution.
    """
    def __init__(self):
        self.code_generator = LLMCodeGenerator(model_name="llama3")
        self.agent_factory = DynamicAgentFactory(logs_dir="../collector/raw_data")

    def execute_task(self, user_task: str) -> bool:
        """
        Executes the full pipeline for a given user task.
        """
        logger.info(f"Processing new task: '{user_task}'")
        
        #generate the code via LLM
        generated_code = self.code_generator.generate_agent_code(user_task)
        if not generated_code:
            logger.error("Pipeline aborted: Failed to generate agent code.")
            return False
            
        logger.debug(f"Generated Code Preview:\n{generated_code[:200]}...\n")
        
        #execute the generated code in the Docker sandbox
        logger.info("Passing generated code to the Agent Factory...")
        success = self.agent_factory.generate_and_execute(generated_code)
        
        if success:
            logger.info("Pipeline completed successfully. Raw data is waiting for synthesis.")
        else:
            logger.error("Pipeline failed during agent execution.")
            
        return success

if __name__ == "__main__":
    #ensure Ollama is running locally before executing
    orchestrator = SynapseOrchestrator()
    
    print("Welcome to Synapse-AI Orchestrator.")
    user_input = input("Enter the monitoring task for the agent: ")
    
    if user_input.strip():
        orchestrator.execute_task(user_input)
    else:
        logger.warning("No task provided. Exiting.")
        sys.exit(1)