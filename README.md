# Embodied Agentic Intelligence

**项目任务规划（中文）：[我们要研究什么、如何实验、希望证明什么](docs/任务规划.md)。**

Research scaffold for measuring direct action, tool use, and planner–tool–memory interaction under the same episode budget. Its task-level RAG means **saving the initial image when a subtask starts and recalling that historical image later when needed**. It does not use vector retrieval. The executable examples are synthetic interface checks, not robotics benchmark or VLM results.

```bash
python3 -m unittest discover -s tests -v
python3 -m embodied_agentic
```

`AgenticHarness` runs a bounded plan/act/observe loop. `SubtaskImageMemory` is its task-level memory: it stores the first PNG/JPEG image for each started subtask and allows explicit later recall only within the same task and episode. The model adapter, `HarnessChatPolicy`, sends actual current and recalled image bytes to an OpenAI-compatible multimodal chat endpoint, labeling recalled images as historical views. The deterministic synthetic demo tests control flow; its policies are **not model baselines**.

`HarnessChatPlanner` and `HarnessChatPolicy` accept a local OpenAI-compatible `/v1` endpoint (for example, a Qwen VLM served by vLLM). Set `MODEL_API_KEY` only if that endpoint requires one. The planner returns ordered subtasks, and the policy chooses a tool, recall, or replan. Decision schema and tool names are checked locally; no real VLM endpoint or simulator has been validated yet.

To attach a simulator, implement `reset(task) -> observation` and `execute(tool, args) -> StepResult`. Every observation must contain `images: {camera_name: PNG_or_JPEG_bytes}`; other observation fields must be JSON-serializable. `StepResult.receipt` must include a boolean `ok`. The simulator adapter owns `subtask_done` and `task_done`; a model assertion cannot mark an episode successful. Supply a unique `episode_id` for each rollout, even when `task_id` and seed are reused. A new episode never reads another episode's images.

`HarnessConfig` bounds decisions, tool calls, recalls, replans and elapsed time, and its `allowed_tools` is an explicit allowlist. The synthetic run writes `memory/manifest.jsonl`, image artifacts and `trace.jsonl` under `--output-dir`; traces contain image hashes and sources rather than raw image bytes.

`evaluation.run_paired` runs each configured condition on the same task/seed list with the same numerical budgets, fresh environment instances and separate episode image banks. It writes per-condition outcomes and budgets to `results.json`. The choice of model, primitive equivalence and native benchmark split must still be controlled when a real simulator is attached.

See [research plan](docs/research_plan.md) for benchmark selection, controls and the model/robot integration path. No model endpoint, simulator or GPU is required for local harness tests. Real simulator and Qwen behavior remain to be validated when that environment is available.
