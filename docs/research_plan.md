# RoboDojo mixed-agent research protocol

The primary question is **which gaps between an open Qwen agent and a frontier closed model remain in embodied manipulation, and which can be narrowed by calling GPT-6 Astra only as a bounded planning or diagnosis tool?** In hybrid arms, Qwen makes every robot action decision. A full Astra agent is a small, separately labeled reference arm for the same-interface model gap. Published RoboICL/PyRUA/HexaShell scores are contextual evidence, not paired results for this project.

See [the Chinese task plan](任务规划.md), [the detailed experimental design](../research/benchmarks/mixed_agent_robodojo.md), [the subsequent Qwen Agentic RL roadmap](../research/benchmarks/agentic_rl_roadmap.md), and [the RoboDojo/Qwen source audit](../research/benchmarks/robodojo_qwen.md).

## Conditions and contrasts

| Arm | Planner | Robot actor | Bounded Astra advice | Subtask-start images |
| --- | --- | --- | --- | --- |
| q_only | Qwen | Qwen | none | off |
| astra_reference | Astra | Astra | full-agent reference, not a hybrid | off |
| hybrid_plan | Astra first plan; Qwen later replans | Qwen | one planning call | off |
| hybrid_diagnose | Qwen | Qwen | automatic on rejected action, or Qwen-requested consult after a prior action | off |
| hybrid_both | Astra | Qwen | planning and diagnosis | off |
| hybrid_both_memory | Astra | Qwen | same budgets as hybrid_both | on |

Primary contrasts are q_only→hybrid_plan (planning), q_only→hybrid_diagnose (failure diagnosis), hybrid_plan→hybrid_both (additional diagnosis), hybrid_both→hybrid_both_memory (task-image memory), and q_only↔astra_reference (limited full-agent reference gap). Fix Qwen checkpoint, three RGB cameras, proprioception, robot primitives, action conversion/safety guards, demonstration regime, native task/seed/layout list, physical-step budget and task scoring. The primary protocol allows at most **one Astra plan and one Astra diagnosis per episode, two closed calls total**; Qwen handles any later replan. A rejected-action receipt can automatically trigger diagnosis; after any prior robot action Qwen can explicitly issue a consult even if the receipt says execution succeeded but observed task progress appears stalled. Consult without a prior action is rejected. Validate the structured response and prevent advisory outputs from containing robot actions or success verdicts. Track separate closed-tool tokens, latency and cost; compare matched physical budget first and report compute tradeoffs alongside success.

## Native RoboDojo path

RoboDojo's Isaac Sim/IsaacLab client owns action execution, termination and scoring; XPolicyLab owns the policy-server contract. Pin their revisions and assets, then run installation doctor, camera preflight, action-contract tests and one native episode before any sweep. **RoboICL-style ee_delta with the native IK-continuity guard requires Mosi-AI's pinned RoboDojo fork `00ef04cd245bfa76e6f1e0bd8383c82f7964f942`; official main `f57e1633ffed59761bc8e568cfe7530c4b33381c` lacks `src.eval_client.ik_continuity`. Joint-action mode can use official main.** Every compared arm must use the same fork and action mode. Start with simple Open tasks such as general_pickup. Build_tower probes Precision, fill_pen_holder Long-Horizon, and cover_blocks Memory. Imitate_sorting_sequence is a boundary case for start-image memory because the critical demonstration unfolds during the episode. A 20–50 episode pilot is an integration check; claims about stable performance need frozen splits and native per-task episode counts.

The image bank captures raw camera bytes once at subtask entry and permits explicit same-task/same-episode recall by subtask ID. No vector index or future/cross-episode lookup exists. RoboICL's LIVE trajectory memory and one-shot demonstration context are separate variables: either disable them across arms or hold them identical. RoboDojo has no general native subtask-done predicate; a policy-requested advance is only **unverified plan progression** and must never count as task success. If a trained VLA executes low-level control, use the same checkpoint across compared arms and label the result planner-over-VLA rather than pure VLM control.

## Acceptance before GPU evaluation

- Tests verify q_only cannot request Astra; plan-only cannot request diagnosis; closed responses cannot send robot commands or claim native success; per-episode calls and time budgets are enforced.
- A rejected action can trigger diagnosis automatically; explicit consult needs a prior action, works after an ok=true receipt, and respects the same diagnosis cap. An unsuccessful tool receipt cannot mark the episode successful.
- Memory tests verify immutable subtask-start bytes, historical labels, episode isolation, recall on request and no raw image bytes in traces.
- Adapter contracts verify fresh three-camera observations, correct 14-D action ordering and conversion, guarded execution, and native terminal/score extraction. Fake-environment checks are not real RoboDojo validation.
- Every real episode records task/seed/layout, model and provider IDs, tool schema/prompt revisions, physical steps, native verdict, each Astra invocation reason, model tokens, elapsed time and cost. A missing native verdict is unknown, not silently recorded as a model failure.

On 2026-10-29, the local Qwen3.5-27B BF16/non-thinking/32K endpoint passed text, three-image, native-tool and Harness planner/actor checks; see the [serving guide](Qwen3.5推理环境.md). RoboDojo GPU simulation and the Astra endpoint remain unvalidated. These interface checks and source tests do not establish a robot performance gap. Upstream integration caveats remain in [the adapter guide](上游框架集成.md).

## Subsequent learning phase

Alignment on 2026-10-29: both stages are research objectives. First produce a capability map showing which gaps benefit most from bounded closed-model advice, including same-input probes, targeted interventions, Qwen self-advice controls, paired native outcomes and costs. The existing six arms provide coarse functional contrasts, not complete single-capability causal attribution; finer diagnostics require separately versioned conditions. See the [capability diagnosis and execution plan](能力诊断与推进计划.md). Do not assume verification or recovery is the bottleneck before collecting evidence.

Keep these six untrained conditions frozen. Once diagnosis identifies worthwhile targets, build executable lightweight environments with state transitions, partial observations, failures and verifiable task outcomes. Train the original Qwen checkpoint directly with multi-turn agentic RL; compare untrained, single-capability and joint-capability policies under new checkpoint names. SFT is outside the current route, not a prerequisite or required baseline; any supplementary SFT study is deferred until after the main work. Evaluate the same trained checkpoint with and without Astra, including any later routing-trained policy. Measure success at fixed total cost and total cost to a prespecified development target, not just update throughput. Final claims require held-out RoboDojo closed-loop episodes, native verdicts, and a full data/expert/training/evaluation cost ledger. If physical RL is added, separate its contribution from lightweight-environment RL. The paper triage in the [roadmap](../research/benchmarks/agentic_rl_roadmap.md) remains based on user-provided material and has not been rechecked against the papers in this update.
