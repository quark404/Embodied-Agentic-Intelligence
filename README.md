# Embodied Agentic Intelligence

Research scaffold for measuring direct action, tool use, and planner–tool–memory interaction under the same episode budget. The current executable is a **synthetic interface smoke test**, not a robotics benchmark or a VLM result.

```bash
python3 -m unittest discover -s tests -v
python3 -m embodied_agentic --output /tmp/embodied-smoke.json
```

The harness defines `Environment.reset/step` and `Policy.act` interfaces. Episode logs contain observations, actions, receipts, success and tool counts. `MemoryBank` accepts only successful training episodes; held-out episodes can retrieve them but cannot write to them. A failed retrieved fact triggers a lookup and retry. The deterministic policies test the plumbing and are **not model baselines**.

`ChatActionPolicy(endpoint, model)` accepts a local OpenAI-compatible `/v1` endpoint (for example, a Qwen model served by vLLM). Set `MODEL_API_KEY` only if that endpoint requires one. Its action schema is checked locally; model quality and vision input are not yet validated.

See [research plan](docs/research_plan.md) for benchmark selection, controls and the model/robot integration path. No model endpoint, simulator or GPU is required for the current smoke test.
