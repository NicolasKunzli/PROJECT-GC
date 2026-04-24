# RUNTIME — run on every invocation
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

AGENT_ID     = os.environ["AGENT_ID"]
ENV_ID       = os.environ["ENV_ID"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]  # PAT with Contents: Read and write

session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENV_ID,
    resources=[
        {
            "type": "github_repository",
            "url": "https://github.com/NicolasKunzli/PROJECT-GC",
            "authorization_token": GITHUB_TOKEN,
            "checkout": {"type": "branch", "name": "main"},
        }
    ],
)
print(f"Session: {session.id}")

# Open stream before sending — so no events are missed
with client.beta.sessions.stream(session_id=session.id) as stream:
    client.beta.sessions.events.send(
        session_id=session.id,
        events=[{
            "type": "user.message",
            "content": [{
                "type": "text",
                "text": (
                    "Explore the full repository at /workspace/PROJECT-GC. "
                    "Read every file and understand the codebase as a whole. "
                    "Then reorganize the structure so it is fluid and easy to navigate: "
                    "rename files and directories for clarity, move functions and classes "
                    "to the most logical location, split large files if needed, group "
                    "related concerns together, and update all imports accordingly. "
                    "Do not change any behavior — only the structure and layout. "
                    "When done, summarize every change you made."
                ),
            }],
        }],
    )

    for event in stream:
        if event.type == "agent.message":
            for block in event.content:
                if block.type == "text":
                    print(block.text, end="", flush=True)
        elif event.type == "session.status_idle":
            if event.stop_reason.type != "requires_action":
                print("\n--- Agent finished ---")
                break
        elif event.type == "session.status_terminated":
            print("\n--- Session terminated ---")
            break
