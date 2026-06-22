"""
Smoke test for the Ollama client.

Prerequisites:
  1. Ollama running locally: `ollama serve` (or installed as a service)
  2. Tool-capable model pulled: `ollama pull qwen3:8b`

Verifies tool calling works end-to-end with a fake weather tool.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.ollama_client import OllamaClient
from src.agent.llm_client import ToolSpec, ToolResult


def main():
    print("Connecting to Ollama (http://localhost:11434)...")
    client = OllamaClient()
    print(f"Model: {client.model}\n")

    weather_tool = ToolSpec(
        name="get_track_temp",
        description="Get the current track temperature in Celsius for a circuit.",
        parameters={
            "type": "object",
            "properties": {
                "circuit": {"type": "string", "description": "Circuit name"},
            },
            "required": ["circuit"],
        },
    )

    print("--- Turn 1: ask a question that should trigger the tool ---")
    response = client.generate(
        system_prompt=(
            "You are a concise F1 race engineer. Use tools when needed to "
            "get information you don't have."
        ),
        messages=[
            {"role": "user", "content": "What is the track temp at Baku right now?"},
        ],
        tools=[weather_tool],
    )

    print(f"Text: {response.text!r}")
    print(f"Tool calls: {response.tool_calls}\n")

    if not response.wants_tools:
        print("⚠️  Model did not call the tool. Some local models need prompt nudging.")
        return

    # Turn 2: feed back a fake tool result
    tc = response.tool_calls[0]
    tool_result = ToolResult(
        call_id=tc.call_id, name=tc.name,
        content='{"track_temp_c": 42.0}',
    )

    response2 = client.generate(
        system_prompt=(
            "You are a concise F1 race engineer. Use tools when needed."
        ),
        messages=[
            {"role": "user", "content": "What is the track temp at Baku right now?"},
            {"role": "assistant", "tool_calls": [tc]},
            {"role": "tool", "tool_result": tool_result},
        ],
        tools=[weather_tool],
    )

    print("--- Turn 2: after tool result ---")
    print(f"Text: {response2.text!r}")
    print(f"Tool calls: {response2.tool_calls}")


if __name__ == "__main__":
    main()
