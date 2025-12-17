import os
import json
import logging
import requests
import subprocess
import time
from typing import Optional, List

from dotenv import load_dotenv

import litellm
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.environments.local import LocalEnvironment

# Load .env variables
load_dotenv("/app/lumyn/.env")
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mini_swe_runner")

if os.path.exists("/app/lumyn"):
    os.chdir("/app/lumyn")
    logger.info("Changed working directory to /app/lumyn")



class InstrumentedLitellmModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.completion_latencies = []

    def query(self, messages: list[dict], **kwargs) -> dict:
        start_time = time.time()
        response = super().query(messages, **kwargs)
        end_time = time.time()
        self.completion_latencies.append(end_time - start_time)
        return response

# PrintingAgent class removed as functionality is now in DefaultAgent



def setup_environment():
    """Reads scenario data and sets up the environment (kubeconfig, etc)."""
    scenario_data_path = "/tmp/agent/scenario_data.json"
    if not os.path.exists(scenario_data_path):
        logging.warning(f"Scenario data not found at {scenario_data_path}")
        return {}

    with open(scenario_data_path, 'r') as f:
        data = json.load(f)

    kubeconfig_content = data.get("kubeconfig")
    if kubeconfig_content:
        kubeconfig_path = "/tmp/kubeconfig.yaml"
        with open(kubeconfig_path, 'w') as f:
            f.write(kubeconfig_content)
        os.environ["KUBECONFIG"] = kubeconfig_path
        logger.info(f"Kubeconfig written to {kubeconfig_path}")

    return data

def get_alerts(base_url: str, token: Optional[str] = None):
    """Fetches firing alerts from Prometheus."""
    if not base_url:
        return []
    
    url = f"{base_url}/prometheus/api/v1/alerts"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            alerts = data.get("data", {}).get("alerts", [])
            firing_alerts = [a for a in alerts if a.get("state") == "firing"]
            return firing_alerts
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}")
    
    return []

def run_agent():
    """Main execution loop."""
    logger.info("Starting mini-swe-agent runner...")
    global_start_time = time.time()
    
    # 1. Setup Environment
    scenario_data = setup_environment()
    
    # 2. Extract Observability Params
    obs_url = os.environ.get("OBSERVABILITY_STACK_URL")
    if not obs_url and "prometheus_url" in scenario_data:
        prom_url = scenario_data["prometheus_url"]
        if prom_url.endswith("/prometheus"):
            obs_url = prom_url[:-11]
        else:
            obs_url = prom_url

    token = os.environ.get("OBSERVABILITY_STACK_SERVICE_ACCOUNT_TOKEN")

    # 3. Fetch Alerts
    alerts = get_alerts(obs_url, token)
    logger.info(f"Refreshed alerts: {len(alerts)} found.")
    
    # 4. Construct Prompt
    system_prompt = (
        "You are a Site Reliability Engineer (SRE). "
        "Your goal is to diagnose and remediation issues in a Kubernetes environment. "
        "You have access to bash commands, including `kubectl`, `curl`, and standard linux utilities. "
        "Execute commands to investigate the issue and fix it."
    )
    
    task_prompt = (
        f"There are currently {len(alerts)} firing alerts in the system.\n\n"
        "Active Alerts:\n"
        f"```json\n{json.dumps(alerts, indent=2)}\n```\n\n"
        "Your Task:\n"
        "1. Diagnose the root cause of these alerts.\n"
        "2. Fix the issue if possible.\n"
        "3. Once you are confident the issue is resolved (or you have a diagnosis), create the following output files in the current directory:\n"
        "   - `diagnosis_struct_out.json`: A JSON block with diagnosis details (fields: `root_cause`, `evidence`).\n"
        "   - `remediation_struct_out.json`: A JSON block with remediation details (fields: `action_taken`, `result`).\n"
        "   - `agent_output.json`: A combined JSON with any other relevant info.\n\n"
        "When you are done and the files are created, reply with 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    
    # Initialize Agent with custom system prompt via kwargs
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    model = InstrumentedLitellmModel(model_name=model_name)
    env = LocalEnvironment()
    
    agent = DefaultAgent(model=model, env=env, system_template=system_prompt)
    
    execution_error = None
    try:
        agent.run(task_prompt)
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        execution_error = e
        with open("agent_output.json", "w") as f:
            json.dump({"error": str(e), "status": "failed"}, f)

    global_end_time = time.time()
    total_duration = global_end_time - global_start_time

    # --- Print Metrics Report ---
    print("\n" + "="*50)
    print("📊 FINAL AGENT METRICS REPORT")
    print("="*50)
    
    # End-to-End Latency
    print(f"⏱️  End-to-End Duration: {total_duration:.2f}s")
    
    # LLM Metrics
    avg_llm_lat = sum(model.completion_latencies) / len(model.completion_latencies) if model.completion_latencies else 0.0
    
    print(f"🤖 Total LLM Calls:      {model.n_calls}")
    print(f"⏳ Avg LLM Latency:      {avg_llm_lat:.4f}s")
    
    # Token Stats
    print(f"💰 Total Cost:           ${model.cost:.4f}")
    print(f"📥 Total Input Tokens:   {model.input_tokens}")
    print(f"📤 Total Output Tokens:  {model.output_tokens}")
    print(f"∑  Total Tokens:         {model.input_tokens + model.output_tokens}")
    print(f"🧠 Reasoning Tokens:     {model.reasoning_tokens}")
    
    planning_overhead = (model.reasoning_tokens / model.output_tokens * 100) if model.output_tokens > 0 else 0.0
    print(f"📊 Planning Overhead:    {planning_overhead:.2f}%")

    # Tool Metrics
    print(f"🛠️  Total Tool Calls:     {agent.tool_call_count}")
    print(f"❌ Tool Failures:        {agent.tool_error_count}")
    error_rate = (agent.tool_error_count / agent.tool_call_count * 100) if agent.tool_call_count > 0 else 0.0
    print(f"📉 Tool Error Rate:      {error_rate:.1f}%")
    
    avg_tool_lat = 0.0
    if agent.tool_latencies:
        avg_tool_lat = sum(agent.tool_latencies) / len(agent.tool_latencies)
        print(f"⏱️  Avg Tool Latency:     {avg_tool_lat:.4f}s")
    
    print("="*50 + "\n")

    # Save metrics to file
    metrics = {
        "duration_seconds": total_duration,
        "llm_calls": model.n_calls,
        "avg_llm_latency_seconds": avg_llm_lat,
        "total_cost": model.cost,
        "input_tokens": model.input_tokens,
        "output_tokens": model.output_tokens,
        "total_tokens": model.input_tokens + model.output_tokens,
        "reasoning_tokens": model.reasoning_tokens,
        "planning_overhead_percent": planning_overhead,
        "tool_calls": agent.tool_call_count,
        "tool_failures": agent.tool_error_count,
        "tool_error_rate_percent": error_rate,
        "avg_tool_latency_seconds": avg_tool_lat,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to metrics.json")



if __name__ == "__main__":
    run_agent()
