"""
Smoke test: verify Gemini client works end-to-end with a simple tool call.

Run after you've set GEMINI_API_KEY in .env.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent.gemini_client import GeminiClient
from src.agent.llm_client import ToolSpec, ToolResult


def main():
    client = GeminiClient()

    # Define a trivial tool
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

    # Turn 1: ask a question that should trigger the tool
    response = client.generate(
        system_prompt="You are a concise F1 race engineer. Use tools when needed.",
        messages=[
            {"role": "user", "content": "What is the track temp at Baku right now?"},
        ],
        tools=[weather_tool],
    )

    print("--- Turn 1 ---")
    print(f"Text: {response.text}")
    print(f"Tool calls: {response.tool_calls}")

    if not response.wants_tools:
        print("Model did not call the tool. End of test.")
        return

    # Turn 2: feed back a fake tool result
    tc = response.tool_calls[0]
    tool_result = ToolResult(
        call_id=tc.call_id,
        name=tc.name,
        content='{"track_temp_c": 42.0}',
    )

    response2 = client.generate(
        system_prompt="You are a concise F1 race engineer. Use tools when needed.",
        messages=[
            {"role": "user", "content": "What is the track temp at Baku right now?"},
            {"role": "assistant", "tool_calls": [tc]},
            {"role": "tool", "tool_result": tool_result},
        ],
        tools=[weather_tool],
    )

    print("\n--- Turn 2 (after tool result) ---")
    print(f"Text: {response2.text}")
    print(f"Tool calls: {response2.tool_calls}")


if __name__ == "__main__":
    main()
