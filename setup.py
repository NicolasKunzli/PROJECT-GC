# ONE-TIME SETUP — run once, save the IDs to your environment
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

environment = client.beta.environments.create(
    name="code-clean-env",
    config={
        "type": "cloud",
        "networking": {"type": "unrestricted"},
    },
)
print(f"ENV_ID={environment.id}")

agent = client.beta.agents.create(
    name="clean",
    model="claude-opus-4-7",
    system=(
        "You are a senior software architect specializing in code organization and readability. "
        "Your task is to thoroughly read a codebase, understand what each file and function does, "
        "then reorganize it so the structure is fluid and easy to navigate. This includes: renaming "
        "files and directories for clarity, moving functions and classes to the most logical location, "
        "splitting overly large files, grouping related concerns together, and ensuring all imports "
        "are updated to reflect every move. Never change the behavior of the code — only its structure."
    ),
    tools=[
        {
            "type": "agent_toolset_20260401",
            "default_config": {"enabled": False},
            "configs": [
                {"name": "read",  "enabled": True},
                {"name": "glob",  "enabled": True},
                {"name": "grep",  "enabled": True},
                {"name": "edit",  "enabled": True},
                {"name": "write", "enabled": True},
            ],
        }
    ],
)
print(f"AGENT_ID={agent.id}")
print(f"AGENT_VERSION={agent.version}")
