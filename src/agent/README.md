# Agent

Provider-agnostic race-engineer loop: one `step()` per lap, tools optional, radio call or silence as the output.

The loop never imports Gemini or Ollama. It talks to [`LLMClient`](llm_client.py). Swap the provider in the CLI; this package stays the same.

## Files

| File | Role |
|---|---|
| [`race_engineer.py`](race_engineer.py) | `RaceEngineerAgent.step(LapContext)` — prompt, tool loop, guardrails, `RadioCall` |
| [`llm_client.py`](llm_client.py) | Abstract `generate()` plus `ToolCall`, `ToolResult`, `LLMResponse`, `ToolSpec` |
| [`gemini_client.py`](gemini_client.py) | Gemini 2.5 Flash (`google-genai`), retries / backoff |
| [`ollama_client.py`](ollama_client.py) | Local Ollama (default `qwen3:8b`) |
| [`prompts.py`](prompts.py) | System prompt: persona, tools, silence-as-default, anti-repetition rules |

## Per-lap flow

```mermaid
flowchart TD
  ctx[LapContext]
  rec[memory.record_lap]
  prompt[system plus memory plus CURRENT LAP]
  llm[LLMClient.generate]
  tools{wants tools?}
  dispatch[dispatch_tool]
  silent[stay_silent]
  extract[extract last quoted radio line]
  jac[Jaccard vs last 3 calls]
  pit[box cooldown unless pit_stop]
  call[RadioCall]
  memOut[memory.record_call]

  ctx --> rec --> prompt --> llm --> tools
  tools -->|yes| dispatch
  dispatch -->|stay_silent| silent --> call
  dispatch -->|other| llm
  tools -->|text| extract --> jac --> pit --> call
  call --> memOut
```

Cap: **4** tool iterations per lap. After that the lap is silent (`iteration cap hit`).

## Guardrails (after a spoken line)

1. Prompt rules in `SYSTEM_PROMPT`.
2. Jaccard token similarity ≥ 0.75 vs the last 3 spoken calls → force silent.
3. Pit phrases (`box this lap`, etc.) within 5 laps of a prior box, and no `pit_stop` event since → force silent.

Radio extraction takes the last double-quoted string, drops preamble lines, caps at 200 characters.

## Adding a provider

Subclass `LLMClient`, translate `ToolSpec` / messages to that API, return `LLMResponse`. The agent loop does not change.
