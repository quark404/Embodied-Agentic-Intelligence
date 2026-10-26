# Research plan and benchmark selection

## Questions and controlled comparisons

1. **Capability boundary:** Compare a direct action VLM, the same model with bounded tool calls, and code-as-policy execution using identical observations, robot primitives, task seeds and wall-clock/action budgets. Report task success, partial progress, intervention count, action errors, tool calls, latency and token/GPU cost. Keep safety limits enforced by the environment.
2. **Memory contribution:** Compare no memory, recent-history only, and explicit subtask-image recall. Capture the initial image when each subtask begins; later recall that image by subtask ID within the same task and episode. Freeze the base model and planner prompt. Test missing, stale, and wrong-task images and whether feedback leads to correction. No vector retrieval is planned.
3. **Planning contribution:** Ablate explicit subgoals, tool inventory and memory independently; use the same model and feedback. Report recovery after injected failures, not only final success.

## Benchmark order

| Stage | Benchmark | Why | Constraint |
| --- | --- | --- | --- |
| 1 | LIBERO / LIBERO-PRO | Established tabletop manipulation, many task variants; PyRUA-Lean already supports LIBERO-PRO through RPent. Good first controlled model and memory ablations. | RPent/robot stack, assets, GPU; check split and task licenses. |
| 2 | RoboDojo-Sim | Shared target of PyRUA-Lean and RoboICL; dexterity, dual arm and explicit feedback. Best direct comparison of code-as-policy and demonstration/history memory. | Isaac Sim 5.1 compatible GPU/Vulkan stack, XPolicyLab and benchmark assets; costly setup. |
| 3 | RoboTwin 2.0 | Dual-arm and scene variation to test transfer of plans/skills. PyRUA-Lean adapter exists. | RPent and LingBot-VLA stack; greater compute. |
| 4 | RoboCasa365 | Mobile manipulation and composite tasks for long-horizon state tracking and memory. PyRUA-Lean adapter exists. | RPent, navigation/VLA stack; use after simpler controls are validated. |

**Recommended first real experiment:** LIBERO-PRO with 20–50 paired episodes per condition across a small fixed task subset and held-out seeds, followed by RoboDojo when the simulator is available. Report confidence intervals over paired task/seed outcomes; do not compare published aggregate numbers from different protocols as if they were a controlled experiment.

## Integration path

- Implement a simulator adapter for `Environment` that maps benchmark observations and native success verdicts without exposing privileged state to the policy. Use PyRUA-Lean's supported OpenAI-compatible runtime and RPent adapter for the LIBERO-PRO code/tool arms. Preserve native run artifacts and task IDs.
- Connect `HarnessChatPlanner` and `HarnessChatPolicy` to a locally served Qwen VLM; their request/response contracts have synthetic tests, but a live endpoint has not been validated. Pin exact checkpoint, tokenizer, prompt, quantization and endpoint version. Keep closed models, if used, as a separately labeled tool/oracle arm, never the sole main baseline.
- Use `AgenticHarness` to run the plan/act/observe loop and `SubtaskImageMemory` to keep the first image of each subtask with task ID, episode ID, subtask ID and provenance. A later explicit recall tool call returns the original image labeled as a subtask-start view, never as the current observation. Log capture and recall events, missing IDs, and the effect of stale images. Isolate episodes; cross-episode experience retrieval is not implemented in this version.
- The harness already limits decisions, tool calls, recalls, replans and elapsed time. Before claiming model gains, add matched experiment conditions, deterministic seed/split manifests, token and GPU accounting, and native benchmark results. The present image memory is episode-local; any future cross-episode bank must use train-only sources and avoid demonstration/test overlap.

## Source framework notes

- [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean): robot primitives, Python cells, OpenAI-compatible runtime; reports LIBERO-PRO, RoboTwin 2.0, RoboCasa365 and RoboDojo support. Its published model comparisons use particular planners and robot stacks, so they are not interchangeable with new Qwen results.
- [RoboICL](https://github.com/Mosi-AI/RoboICL): RoboDojo interaction and demonstration memory, 8 zero-shot Open and 22 one-shot non-Open tasks in its documented setting. Its runner requires separate policy/simulator environments and Isaac Sim assets.

The repository's simulator-independent harness can validate control flow and image-memory bookkeeping with synthetic environments. Real benchmark scores require installing those upstream stacks and running their native evaluators.
