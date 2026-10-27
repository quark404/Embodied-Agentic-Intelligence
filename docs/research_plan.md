# Research plan and benchmark selection

## Questions and controlled comparisons

1. **Capability boundary:** Compare a direct action VLM, the same model with bounded tool calls, and code-as-policy execution using identical observations, robot primitives, task seeds and wall-clock/action budgets. Report task success, partial progress, intervention count, action errors, tool calls, latency and token/GPU cost. Keep safety limits enforced by the environment.
2. **Memory contribution:** Compare no memory, recent-history only, and explicit subtask-image recall. Capture the initial image when each subtask begins; later recall that image by subtask ID within the same task and episode. Freeze the base model and planner prompt. Test missing, stale, and wrong-task images and whether feedback leads to correction. No vector retrieval is planned.
3. **Planning contribution:** Ablate explicit subgoals, tool inventory and memory independently; use the same model and feedback. Report recovery after injected failures, not only final success.

## Benchmark order

| Stage | Benchmark | Why | Constraint |
| --- | --- | --- | --- |
| 1 | RoboDojo-Sim | The project's primary benchmark: official five-dimension evaluation, bimanual control, and explicit Memory/Long-Horizon tasks. See the [Qwen-specific evidence audit](../research/benchmarks/robodojo_qwen.md). | Isaac Sim 5.1 compatible GPU/Vulkan stack, XPolicyLab and benchmark assets; first establish that a Qwen agent can execute valid actions and obtain a native verdict. |
| 2 | LIBERO / LIBERO-PRO | Qwen base models have nonzero results in PyRUA-Lean on this benchmark, making it a useful diagnostic and cross-platform comparison. | RPent/robot stack, assets, GPU; do not treat LIBERO scores as RoboDojo scores. |
| 3 | RoboTwin 2.0 | Dual-arm and scene variation to test transfer of plans/skills. PyRUA-Lean adapter exists. | RPent and LingBot-VLA stack; greater compute. |
| 4 | RoboCasa365 | Mobile manipulation and composite tasks for long-horizon state tracking and memory. PyRUA-Lean adapter exists. | RPent, navigation/VLA stack; use after simpler controls are validated. |

**Recommended first real experiment:** Connect the harness to RoboDojo's native XPolicyLab evaluation path; perform a small, fixed Qwen-agent survival pilot on simple Open tasks before testing memory on `cover_blocks` and planning on Long-Horizon tasks. A Qwen-backed trained VLA has published nonzero RoboDojo results, but no general Qwen VLM agent success-rate sweep has been verified here. Report paired task/seed outcomes only after a functioning model/action baseline is established; do not compare published aggregate numbers from different protocols as if they were controlled experiments.

## Integration path

- Implement an XPolicyLab policy server bridge for RoboDojo's official environment client. Map benchmark observations and native success verdicts without exposing privileged state to the policy. Preserve native run artifacts and task IDs; the current `RoboICLAdapter` handles an initialized environment but is not itself a complete official evaluation entry.
- Connect `HarnessChatPlanner` and `HarnessChatPolicy` to a locally served Qwen VLM; their request/response contracts have synthetic tests, but a live endpoint has not been validated. Pin exact checkpoint, tokenizer, prompt, quantization and endpoint version. Keep closed models, if used, as a separately labeled tool/oracle arm, never the sole main baseline.
- Use `AgenticHarness` to run the plan/act/observe loop and `SubtaskImageMemory` to keep the first image of each subtask with task ID, episode ID, subtask ID and provenance. A later explicit recall tool call returns the original image labeled as a subtask-start view, never as the current observation. Log capture and recall events, missing IDs, and the effect of stale images. Isolate episodes; cross-episode experience retrieval is not implemented in this version.
- The harness already limits decisions, tool calls, recalls, replans and elapsed time. Before claiming model gains, add matched experiment conditions, deterministic seed/split manifests, token and GPU accounting, and native benchmark results. The present image memory is episode-local; any future cross-episode bank must use train-only sources and avoid demonstration/test overlap.

## Source framework notes

- [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean): robot primitives, Python cells, OpenAI-compatible runtime; reports LIBERO-PRO, RoboTwin 2.0, RoboCasa365 and RoboDojo support. Its published model comparisons use particular planners and robot stacks, so they are not interchangeable with new Qwen results.
- [RoboICL](https://github.com/Mosi-AI/RoboICL): RoboDojo interaction and demonstration memory, 8 zero-shot Open and 22 one-shot non-Open tasks in its documented setting. Its runner requires separate policy/simulator environments and Isaac Sim assets.

The repository's simulator-independent harness can validate control flow and image-memory bookkeeping with synthetic environments. Real benchmark scores require installing those upstream stacks and running their native evaluators.
