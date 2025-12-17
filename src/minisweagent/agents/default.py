"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
import time
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n"
        "{% if output | length < 10000 -%}\n"
        "<output>\n{{output}}\n</output>\n"
        "{%- else -%}\n"
        "<warning>Output was too long and has been truncated.</warning>\n"
        "<output_head>\n{{ output[:5000] }}\n</output_head>\n"
        "<elided_chars>{{ output | length - 10000 }} characters elided</elided_chars>\n"
        "<output_tail>\n{{ output[-5000:] }}\n</output_tail>\n"
        "{%- endif %}\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.tool_call_count = 0
        self.tool_error_count = 0
        self.tool_latencies = []

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        response = self.query()
        print(f"\n\033[92m🤖 Agent Action:\033[0m\n{response['content']}\n")
        output = self.get_observation(response)
        print(f"\n\033[93m👁️ Observation:\033[0m\n{output.get('output', '')}\n")
        return output

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        start_time = time.time()
        print(f"\n\033[94mTOOL CALL\033[0m\n{action['action']}\n")  
        is_error = False
        try:
            output = self.env.execute(action["action"])
            print(f"\n\033[93mOUTPUT\033[0m\n{output.get('output', '')}\n")
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            is_error = True
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except Exception:
            is_error = True
            raise
        else:
            # Check return code for error if available in output dict
            if isinstance(output, dict) and output.get("returncode", 0) != 0:
                is_error = True
            
            try:
                self.has_finished(output)
            except Submitted:
                # Submitted is a success
                is_error = False
                raise
            except Exception:
                is_error = True
                raise

            return output | {"action": action["action"]}
        finally:
            end_time = time.time()
            latency = end_time - start_time
            self.tool_call_count += 1
            if is_error:
                self.tool_error_count += 1
            self.tool_latencies.append(latency)
            
            # Print Metric
            status = "FAILED" if is_error else "SUCCESS"
            color = "\033[91m" if is_error else "\033[94m" # Red if failed, Blue if success
            print(f"\n{color}📊 Tool Metrics:\033[0m Status: {status}, Latency: {latency:.4f}s\n")

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
