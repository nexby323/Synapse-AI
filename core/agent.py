# For logging the system
import logging
# To load file and folder from the project
import sys
# The normal use of pandas
import pandas as pd
# To deal easily with paths on the system
from pathlib import Path
# The class for generating the agent's code to get the resources 
from schema_parser import LLMCodeGenerator
# The agent's code runner (by Docker)
from agent_factory import DynamicAgentFactory
# To deal easily with paths on the system
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'synthesizer')))
from vae_model import TabularVAE
from validator import PrivacyValidator
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
        # The llama3 model is used (localy)
        self.code_generator = LLMCodeGenerator(model_name="llama3")
        # The path for the agents to write the raw_data collected is as the paramater 
        self.agent_factory = DynamicAgentFactory(logs_dir="../collector/raw_data")
        # The path to the raw data collected 
        self.raw_data_path = Path("../collector/raw_data/collected_data.csv")
        # The path to the safe obscure data  
        self.synthetic_data_path = Path("../synthesizer/synthetic_data/safe_output.csv")

        # Make a directory on the synthetic_data path for the data 
        self.synthetic_data_path.parent.mkdir(parents=True, exist_ok=True)
        
    def execute_task(self, user_task: str) -> bool:
        """
        Phase 1: Prompt -> Code Generation -> Data Collection
        """
        logger.info(f"Processing task: '{user_task}'")
        
        generated_code = self.code_generator.generate_agent_code(user_task)
        if not generated_code:
            return False
            
        success = self.agent_factory.generate_and_execute(generated_code)
        if not success or not self.raw_data_path.exists():
            logger.error("Data collection failed or file not found.")
            return False
            
        return self._synthesize_and_validate()
    
    def _synthesize_and_validate(self) -> bool: 
        """
        Phase 2: The "self healing loop" - VAE Training + Validation
        """
        # Using method from the tensorflow framework as in the link https://www.tensorflow.org/api_docs/python/tf/keras/Model
        logger.info("Starting synthesis and validation loop")
        # Load the raw data and handle an exception if occurs 
        try: 
            df = pd.read_csv(self.raw_data_path)
            
            real_data = df.values 
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            return False
        # The number of coulmns in the data collected, number of features 
        input_dim = real_data.shape[1]
        vae = TabularVAE(intput_dim = input_dim, latent_dim = 4)
        validator = PrivacyValidator() # Use the default values (ks_treshold = 0.1, privacy_distance_treshold = 1-e4 )
        
        max_attempts = 3 
        epochs_per_attempts = 50 
        vae.compile(optimizer='adam')
        for attemp in range(1,max_attempts+1):
            logger.info(f"--- Synthesis Attempt {attemp}/{max_attempts} ---")
            
            # Train the vae on the private data 
            logger.info(f"Training VAE for {epochs_per_attempts} epochs")
            vae.fit(real_data,epochs=epochs_per_attempts,batch_size=32,verbose=0)
            
            # Creating the synthetic data (with the same number of sample)
            num_samples = len(real_data)
            synthetic_data = vae.generate_synthetic(num_samples)
            
            # Validates the data (by similarity and privacy) 
            if validator.validate_data(real_data,synthetic_data):
                # If the synthetic data passes the validation - save the data
                
                #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html save the synthetic_data to a csv file
                pd.DataFrame(synthetic_data,columns=df.columns).to_csv(self.synthetic_data_path,index=False)
                
                logger.info(f"SUCCESS: Safe synthetic data exported to {self.synthetic_data_path}")
                
                self.raw_data_path.unlink() # Deleting the file on that path (to not save any private data)
                logger.info("Raw data securely destroyed.")
                return True
            else:
                logger.warning("Validation failed. Forcing VAE to learn deeper relationships")
                # The loop repeats until success or max attemps are passed
        logger.error("Model failed to converge on safe data after maximum attempts.")
        # Destroy the raw data when failure too
        self.raw_data_path.unlink()
        return False
                
            
def main(): 
    # Ensure Ollama is running locally before executing
        orchestrator = SynapseOrchestrator()
        
        print("Welcome to Synapse-AI Orchestrator.")
        user_input = input("Enter the monitoring task for the agent: ")
        
        if user_input.strip():
            orchestrator.execute_task(user_input)
        else:
            logger.warning("No task provided. Exiting.")
            sys.exit(1)
if __name__ == "__main__":
    main()