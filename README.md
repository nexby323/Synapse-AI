# Synapse-AI: Decentralized Privacy-Preserving Agent Platform

Synapse-AI is a secure, decentralized AI architecture engineered to autonomously collect host system telemetry, transform raw logs into statistically identical synthetic datasets via deep generative modeling, and mathematically validate privacy guarantees prior to external consumption.

---

## Core Concepts

* **Dynamic LLM-Driven Agents:** Transforms natural language monitoring intents into executable Python scripts using a local LLM (Ollama), enabling adaptable data collection without hardcoded parameters.
* **Docker-out-of-Docker (DooD):** Executes collection agents in isolated sibling containers by mounting the host's Docker socket. This provides secure, ephemeral sandboxing without the severe security risks associated with privileged Docker-in-Docker (DinD).
* **Variational Autoencoder (VAE) Synthesis:** Instead of masking or hashing data, a neural network learns the continuous statistical distribution (latent space) of the host's behavior. It generates entirely novel, mathematically sound data points that mimic the system without containing real user data.
* **Dual-Gate Validation:** A self-healing feedback loop that tests the synthetic data before the raw data is destroyed.
  * *Utility Gate (KS Test):* Ensures the synthetic data's cumulative distribution matches the real telemetry (Kolmogorov-Smirnov distance).
  * *Privacy Gate (DCR Test):* Measures the Euclidean Distance to Closest Record to guarantee the VAE did not overfit and memorize exact raw data.
* **Zero-Exposure Cloud Integration:** By exporting mathematically verified synthetic data, Synapse-AI safely feeds external commercial LLMs (like Gemini) or global Federated Learning networks without exposing raw, sensitive system logs.

---

## Architecture Overview

```text
                      +-----------------------------------+
                      |       User Monitoring Intent      |
                      +-----------------+-----------------+
                                        |
                                        v
                            +-----------------------+
                            | core/schema_parser.py | (Local Ollama / LLM)
                            +-----------+-----------+
                                        | Python Agent Code
                                        v
                            +-----------------------+
                            | core/agent_factory.py | (Sandboxed Docker Sibling)
                            +-----------+-----------+
                                        |
                                        v
                            [ Raw Host Telemetry ]  (collector/raw_data/)
                                        |
                                        v
                         +-----------------------------+
               +-------->|  synthesizer/vae_model.py   |<--------+
               |         +--------------+--------------+         |
               |                        | Synthetic Batch        | (Self-Healing
     Retrain on Failure                 v                        |  Feedback Loop)
               |         +-----------------------------+         |
               +---------|   synthesizer/validator.py  |---------+
                         +--------------+--------------+
                                        | Verified Safe Data
                                        v
                         [ Safe Synthetic CSV Output ]
                                        |
                                        +--------------------------------+
                                        |                                |
                                        v                                v
                        +-------------------------------+   +-----------------------+
                        | external_models/gemini_ai.py  |   | Federated Learning /  |
                        | (Zero-Exposure Threat Hunting)|   | LSTM Anomaly Nodes    |
                        +-------------------------------+   +-----------------------+
```
## Repository Structure
```text 
Synapse-AI/
├── core/
│   ├── agent.py               # Orchestrator & self-healing pipeline
│   ├── agent_factory.py       # Sandboxed Docker runner (DooD)
│   └── schema_parser.py       # Ollama code generation engine
├── collector/
│   └── raw_data/              # Ephemeral folder for raw logs (git-ignored)
├── synthesizer/
│   ├── vae_model.py           # Generative Tabular VAE model
│   ├── validator.py           # KS-test & DCR verification suite
│   └── synthetic_data/        # Verified, exported anonymized CSVs
├── external_models/
│   └── gemini_analyzer.py     # Interactive Cloud LLM analysis client
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .gitignore
``` 

## Installation and Prerequisites
### 1. System Requirements
- Docker Engine & Docker Compose installed and running.
- Python 3.9+
- Ollama running locally with the LLaMA 3 model:
```bash
ollama run llama3
``` 
### 2. Environment Setup
```bash 
git clone [https://github.com/nexby323/Synapse-AI.git](https://github.com/nexby323/Synapse-AI.git)
cd Synapse-AI
# Create and configure the environment variables file
touch .env
#Open the .env file in your text editor and add your Gemini API key:
GEMINI_API_KEY=your_actual_api_key_here

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
``` 
## End-to-End Pipeline Execution
Run the complete pipeline directly through the orchestrator:
```bash 
python core/agent.py 
``` 
### Execution Flow:
1. Prompt: Enter your collection task when prompted (e.g., "Monitor system memory, CPU usage per core, and active TCP connection count every second").

2. Collection: The orchestrator generates the agent script via Ollama, builds an ephemeral Docker container, and collects the raw metrics to the local volume.

3. Synthesis & Validation: The VAE trains on the raw metrics, samples synthetic equivalents, and validates them against the KS and DCR thresholds.

4. Secure Export: Raw data is immediately destroyed upon validation. Clean synthetic logs are saved to synthesizer/synthetic_data/safe_output.csv.


## External Cloud Analysis
Once the synthetic data is generated and verified, you can safely query it using external APIs without leaking host data. Ensure your .env file is populated, then run:
```bash
# If running locally via Python:
python external_models/gemini_analyzer.py

# Or if executing inside the running Docker container:
docker exec -it synapse_core python external_models/gemini_analyzer.py
``` 