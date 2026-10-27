# Embodied Agentic Intelligence

## 研究目标

以**同一个通用开源 VLM（优先 Qwen 系列）**为起点，在同一批具身任务、相同底层动作原语和预算下，依次加入工具调用、子任务规划和任务级图像记忆。我们要测清楚模型原本能完成什么、每个 Agentic 组件实际带来多少提升，以及哪些失败仍然无法解决。这里的任务级检索是在子任务开始时保存初始图像，之后需要回顾该子任务时按任务 ID 取回；它不是向量 RAG。所有成功率由 benchmark 原生判定，相关结论须通过真实模型与仿真实验验证。

**当前主评测目标为 RoboDojo-Sim。** Qwen3-VL 骨干的训练型 VLA 已在 RoboDojo 单任务获得非零结果；公开的通用 Qwen Agent + RPent + OpenWAM 仅找到一个完整但失败的 episode，尚不能视为成功率基线。我们先复现原生评测链路和简单任务的 Qwen 存活性，再进行 A/B/C/D 消融。不同模型角色、证据强度和主要参考见 [RoboDojo 专项调研](research/benchmarks/robodojo_qwen.md)，其他 benchmark 见[早期横向调研](research/benchmarks/README.md)。

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

`evaluation.run_paired` runs each configured condition on the same task/seed list with the same numerical budgets, fresh environment instances and separate episode image banks. A condition may select its own `environment_factory`, allowing PyRUA and RoboICL runtime arms to use the same harness. It writes per-condition outcomes and budgets to `results.json`. The choice of model, primitive equivalence and native benchmark split must still be controlled when a real simulator is attached.

For A/B/C/D ablations, `make_ablation_conditions` fixes the component matrix; `GoalOnlyPlanner` represents an episode-wide task in A/B, while C/D share a model planner and B/C/D share the same tool allowlist. Each `Condition` can disable image capture/recall and replanning independently, and `HarnessChatPolicy.allowed_decisions` restricts what the model may request. See the [benchmark research and experiment route](research/benchmarks/README.md) for the pilot gate and scientific controls.

The [upstream integration guide](docs/上游框架集成.md) explains the initial bridges in `upstream_adapters.py`: PyRUA-Lean `ArmToolkit` cells plus robot image artifacts and native episode verdicts; RoboICL's initialized RoboDojo observations and actions via injected conversion, guarded execution, fresh-camera reading and native scoring. They are alternative runtime arms, not yet a single shared simulator comparison. Adapter tests use fakes; neither upstream stack has been validated live here.

See [research plan](docs/research_plan.md) for benchmark selection, controls and the model/robot integration path. No model endpoint, simulator or GPU is required for local harness tests. Real simulator and Qwen behavior remain to be validated when that environment is available.
