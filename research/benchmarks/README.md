# 近期 Qwen 具身评测证据与首轮 benchmark 选择

资料核查：2026-10-28。这里记录**公开来源报告的成绩**，不是本项目复现结果。沿 [RoboICL 论文页面与代码](https://github.com/Mosi-AI/RoboICL) 的 RoboDojo 路线，核查其模型与协议；再查 [PyRUA-Lean 2026-10 技术报告配套代码](https://github.com/DAGroup-PKU/PyRUA-Lean) 中同类 Agent 框架是否已有 Qwen 分数。论文 PDF 本次无法从 arXiv 访问，RoboICL 的具体协议和数字只采用已核对的发布仓库材料；没有把未读的论文正文当作证据。两仓库在本次核查的 HEAD 分别为 `17b379a6deea33f71bd31e05bd60d1daf627acfb`、`fdf6e870654254e2c0c891ee407313fa48bdd0f9`。上游更新后应重查。

## 关键判断

**Qwen 在机器人 Agent 框架下并非普遍 0 分。** PyRUA-Lean 2026-10-13 的技术报告和 2026-10-15 发布的模型，在相同 PyRUA-Lean 框架内报告了 Qwen3.5-4B、Qwen3.5-27B 与基于 Qwen3.5-4B 训练的 Coder 成绩。LIBERO-PRO 未见任务上的原始 Qwen3.5-4B 为 **13.5%**，Qwen3.5-27B 为 **38.1%**。这足以把 LIBERO-PRO 作为首轮验证入口，却**不能证明**本项目的 prompt、模型服务、动作转换和 harness 能复现同一分数，也不能证明裸 VLM 直接输出底层动作就能成功。

RoboICL 2026-09-29 论文是较新的 RoboDojo 多轮交互/示范记忆路线，但发布的 runner 接的是 **GPT-6 Astra**，不是 Qwen。其成功例和公开回放不能充作“Qwen 在 RoboDojo 非零”的证据。RoboDojo 仍有价值：它提供长程与记忆任务，也有可用的原生评测器；但应在 Qwen 先导测试通过后再作为主实验。

## 可核查的公开数字

以下均为 **PyRUA-Lean 框架中的 planner/代码单元 + 机器人原语**成绩，单位为成功率百分比。训练列为 45 个 LIBERO-PRO 实例；“未见 655”包含其余 155 个 LIBERO-PRO 实例、250 个 RoboTwin 2.0、90 个 RoboCasa365 atomic 和 160 个 RoboCasa365 composite。数据来自 [PyRUA-Lean README 的 Performance 表](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/fdf6e870654254e2c0c891ee407313fa48bdd0f9/README.md#-performance) 与 [模型说明](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/fdf6e870654254e2c0c891ee407313fa48bdd0f9/docs/models/coder.md)。

| 模型/条件 | LIBERO-PRO 未见 155 | RoboTwin 2.0 250 | RoboCasa365 atomic 90 | RoboCasa365 composite 160 | 未见 655 汇总 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B 基座 | 13.5 | 14.0 | 52.2 | 12.5 | 18.8 |
| Qwen3.5-27B 基座 | 38.1 | 20.8 | 65.6 | 13.1 | 29.2 |
| PyRUA-Lean Coder v0.1-4B（Qwen3.5-4B 经 GRPO） | 37.4 | 52.4 | 76.7 | 19.4 | 44.1 |

**不可误读：**上表不是 RoboICL 的 RoboDojo 分数，不是本项目的 A/B/C/D 对照，也不是纯粹“给一张图直接预测机械臂动作”的 VLM 分数。PyRUA-Lean 的 LIBERO 路径还可能使用 Pi0.5 VLA/SAM3 等原语；若研究“通用 VLM 自身能力”，必须另设 `no-vla` 条件或明确标注原语能力。训练后的 Coder 是可选的上界/参照，不能和未训练基座混称一个模型的 harness 增益。

PyRUA-Lean 的 GPT-6 Astra 表还报告 LIBERO-PRO 工具调用 **83.0%**、代码单元 **94.0%**，以及 RoboDojo 官方评估 **50.35%**；它们只说明平台和原语可以跑通，**不作为本项目以开源 Qwen 为主的实验基线**。见同一 [Performance 表](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/fdf6e870654254e2c0c891ee407313fa48bdd0f9/README.md#-performance) 与 [RoboDojo 官方结果](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/fdf6e870654254e2c0c891ee407313fa48bdd0f9/integrations/xpolicylab/evaluation/robodojo_summary_official.md)。

## 沿 RoboICL 论文看 RoboDojo

- [RoboICL 论文（arXiv:2609.34261）](https://arxiv.org/abs/2609.34261) 与 [发布仓库](https://github.com/Mosi-AI/RoboICL/tree/17b379a6deea33f71bd31e05bd60d1daf627acfb) 描述冻结通用多模态模型、示范轨迹和当前 rollout 的交互记忆；发布 runner 的模型为 GPT-6 Astra。
- [零样本协议](https://github.com/Mosi-AI/RoboICL/blob/17b379a6deea33f71bd31e05bd60d1daf627acfb/configs/protocols/zero_shot_b25.json) 为 0 示范、25 个 LIVE 记忆块；[单样本协议](https://github.com/Mosi-AI/RoboICL/blob/17b379a6deea33f71bd31e05bd60d1daf627acfb/configs/protocols/one_shot_j12_b12.json) 为 1 示范、12 TRAIN 块和 12 LIVE 块。论文评测覆盖 8 个 Open 零样本任务和 22 个非 Open 单样本任务，见[仓库 Data preparation](https://github.com/Mosi-AI/RoboICL/blob/17b379a6deea33f71bd31e05bd60d1daf627acfb/README.md#one-shot-references)。
- RoboICL 的动作是双臂 `[T,14]` 动作块，需要匹配的转换、安全守卫和原生成功判定。我们的子任务**初始图像回忆**与 RoboICL 的 LIVE 轨迹记忆不是同一种机制；比较时须分别开关，保留示范数量、相机、动作 horizon、预算和评分协议。
- 本次在 RoboICL 发布配置中**没有找到可核查的 Qwen RoboDojo 成功率**。这是“证据尚缺”，不是断言 Qwen 得分为 0。若后续发现同协议公开结果，应补充精确版本、任务数、seed、动作接口和来源。

## benchmark 排序与决策

| 顺序 | benchmark | 现有 Qwen 非零依据 | 本项目下一步 |
| --- | --- | --- | --- |
| 1 | **LIBERO-PRO** | 有：基座 Qwen3.5-4B/27B 在 PyRUA-Lean 原语路径的公开分数 | 首先复现少量未见任务与原生判定；冻结模型、原语、任务、seed，然后做 harness 消融 |
| 2 | **RoboTwin 2.0 / RoboCasa365** | 有：同一报告给出 Qwen 基座分数 | 用于未见平台与长程任务外推；当前仓库还缺各自的启动适配 |
| 3 | **RoboDojo-Sim** | 本次未找到同协议 Qwen 分数；RoboICL/PyRUA 公开成绩使用 GPT-6 Astra | 先接 Qwen 做存活性测试；再研究记忆与双臂长任务，避免直接承诺完整 A/B/C/D |

优先使用 Qwen3.5-4B 基座与 Qwen3.5-27B 基座作开源参照；PyRUA-Lean Coder v0.1-4B 单列为经过训练的模型参照。可选新 checkpoint 必须先完成同一小样本先导测试，不能依据模型名称推断机器人成功率。

## 可执行的实验路线与代码对应

1. **先导复现。** 选 LIBERO-PRO 未见任务，固定任务、seed、观察、模型 checkpoint、推理参数、原语集、运行时间与原生成功判定。先跑 Qwen + 上游 PyRUA-Lean 原生入口，确认我们使用的环境能出现非零成功；再用本仓库 adapter 跑相同条件，核对动作回执与原生 verdict。20–50 个 episode 可作工程先导，不作为最终统计结论。
2. **若先导全失败。** 先看模型是否能产生合法工具/动作、相机是否有效、转换是否执行、是否因时间/动作预算停止；再按任务难度分层。0/20 只能说明该小样本尚未成功，不能证明真实成功率为零，也不能直接判定“harness 无效”。若同环境上游也为零，暂缓做记忆增益结论，换更简单任务/原语或模型；若上游非零而本项目为零，先修接口与控制流。
3. **固定模型后配对消融。** 本仓库 `make_ablation_conditions` 生成 A/B/C/D 固定组件矩阵，`run_paired` 对每个任务/seed 创建独立 episode。`GoalOnlyPlanner` 用于不显式拆解的 A/B，`Condition.enable_image_memory` 与 `enable_replanning` 在代码层面关闭组件，`HarnessChatPolicy.allowed_decisions` 同时约束模型输出。A/B 策略应只允许 `tool`；C 允许 `tool,replan`；D 允许 `tool,recall,replan`。

| 条件 | Planner | 动作/工具 | 重新规划 | 子任务初始图像回忆 | 检验 |
| --- | --- | --- | --- | --- | --- |
| A 直接操作 | `GoalOnlyPlanner` | benchmark 允许的直接动作接口 | 关 | 关 | 同一模型的最小可运行基线 |
| B 工具调用 | `GoalOnlyPlanner` | 明确的机器人原语/工具 | 关 | 关 | 工具带来的变化 |
| C 规划 + 工具 | 模型 Planner | 与 B 相同 | 开 | 关 | 子任务规划与纠错的变化 |
| D 规划 + 工具 + 记忆 | 与 C 相同 | 与 C 相同 | 开 | 开 | 任务级图像回忆的增量 |

A 与 B 只有在**同一环境中定义了可比的底层动作接口**后才能归因于“工具”。B/C/D 必须保持相同模型、工具原语与预算；C/D 应保持相同 planner。当前 `run_paired` 负责调度和记录，不会自动证明这些科学控制已经满足。D 的回忆只查本 episode 中已经开始的子任务，旧图像必须与当前观察并列标明。RoboICL 原生 LIVE 记忆/示范另设因子，不能混进 D 的效果。

4. **报告。** 保存每个 episode 的 native verdict、任务 ID、seed、模型与原语版本、决策/动作/回忆次数、耗时及失败原因。分别报告各任务族成功率、恢复率与配对差值；样本不足时只称工程先导。特别区分“基座模型成绩”“训练过的 Qwen Coder 成绩”“harness 组件消融成绩”。

### 尚需完成的真实环境工作

本仓库现有 PyRUA-Lean 与 RoboICL adapter 只经过合成环境/上游假环境契约检查；尚未接通 GPU 仿真、官方评测主循环或本项目的 Qwen endpoint。RoboICL runner 的 GPT-6 Astra 提示词、Responses API 和动作语法不能直接当成 Qwen 的已验证实现。真实跑分前还要冻结动作 schema、接原生 task/subtask 判定，并保留上游版本与资产清单。本文件的优先级基于上游公开材料，不是本项目的 benchmark 实验结果。
