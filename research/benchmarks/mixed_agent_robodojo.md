# RoboDojo 混合 Agent：Qwen 主控制、Astra 受限认知工具

协议草案日期：2026-10-28。此文件规定**将来实际运行前应冻结的比较关系**，不是已完成的 RoboDojo 成绩。主问题是：通用开源 Agent 与闭源前沿模型在机器人任务中差多少，差在何处，能否仅用少量闭源规划/诊断工具缩小差距？[任务目标](../../docs/任务规划.md)给出通俗版本；[RoboDojo 环境与论文核查](robodojo_qwen.md)给出已核查的公开依据。

## 为什么选择受限工具范式

[RoboICL（2026-09）](https://github.com/Mosi-AI/RoboICL/tree/17b379a6deea33f71bd31e05bd60d1daf627acfb) 将 GPT-6 Astra 放进完整的多轮动作控制与示范/交互记忆流程；[PyRUA-Lean（2026-10）](https://github.com/DAGroup-PKU/PyRUA-Lean/tree/fdf6e870654254e2c0c891ee407313fa48bdd0f9) 让模型写机器人 Python cells；[HexaShell（2026-10）](https://github.com/HexaFuture/HexaShell/tree/d1a2a24dc4c528a652b359fd689398530ef941d8) 用 GPT-6 Astra 的代码 Agent 与失败后工具修复，在 RoboDojo 官方评测报告 46.9%。三者展示了强模型与执行接口协同的可能性，但公开分数采用不同工具、模型、动作与任务协议，不能互作直接消融。

我们的主实验让 Qwen 决定每一次机器人动作，只在事先规定的决策点向 Astra 请求**结构化计划**或**故障诊断**。这样 Qwen 的动作接口、低层能力和仿真条件保持一致，可以估计“多给一次高质量规划/诊断”带来的增量。为了知道差距本身有多大，在**一小组同接口任务**上保留完整 Astra Agent 参考臂；它不属于混合方案，其计算成本必须另报。若无这个参考臂，就只能说混合组相对 Qwen 改进多少，不能直接给出“闭源模型与开源模型的差距”。

## 论文启发与不能照搬之处

| 来源 | 借鉴 | 不照搬的原因 |
| --- | --- | --- |
| RoboICL：arXiv [2609.34261](https://arxiv.org/abs/2609.34261)；[零样本/单示范配置](https://github.com/Mosi-AI/RoboICL/tree/17b379a6deea33f71bd31e05bd60d1daf627acfb/configs/protocols) | 三相机实时观察、双臂动作格式、执行回执、安全守卫、0/1-shot 协议、同一 rollout 的 LIVE 记忆 | 其 0-shot 8 Open、1-shot 22 非 Open 及 Astra 完整动作预测不是我们的闭源工具条件；原生 LIVE 记忆需单独固定/关闭 |
| PyRUA-Lean：arXiv [2610.01939](https://arxiv.org/abs/2610.01939)；[RoboDojo 接口](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/fdf6e870654254e2c0c891ee407313fa48bdd0f9/docs/robots/robodojo.md) | 可组合机器人原语、短程序、动作回执、XPolicyLab 接入 | 其 Qwen3.5 基座的非零分数来自其他 benchmark；RoboDojo 公开 Agent 分数主要是 Astra，不是我们 Qwen 的预计分数 |
| HexaShell：arXiv [2610.23479](https://arxiv.org/abs/2610.23479)；[论文 PDF 和源码](https://github.com/HexaFuture/HexaShell/tree/d1a2a24dc4c528a652b359fd689398530ef941d8) | 统一 RGB+状态接口、审计动作工具、从失败轨迹定位控制瓶颈 | 其工具在跨 episode 开发并冻结；我们首轮的图像记忆仅限同一 episode，不能引入测试任务专属工具脚本 |
| [RoboDojo 论文 2607.04434](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/f57e1633ffed59761bc8e568cfe7530c4b33381c) 与 [XPolicyLab 2608.09892](https://github.com/XPolicyLab/XPolicyLab/tree/c43a74653bec480ebe82c84770664e8035bc56c9) | 原生五维任务/评分、Isaac 仿真、模型服务与仿真 client 分离 | 自定义 Harness 日志不能替代原生结果，也不能把独立任务与其配置变体混用 |

RoboICL、PyRUA-Lean 等 arXiv PDF 本次无法取得；上表的配置细节据固定提交的发布仓库核查。HexaShell 仓库随附 PDF 已核读。论文来源和公开分数的证据强度见 [RoboDojo × Qwen 调研](robodojo_qwen.md)。

## 冻结条件矩阵

| 条件名 | 初始/重规划 | 失败诊断 | 机器人动作决策 | 子任务初始图像回忆 |
| --- | --- | --- | --- | --- |
| q_only | Qwen | Qwen 自行处理回执 | Qwen | 关 |
| astra_reference | Astra | Astra | Astra | 关；仅小样本参照 |
| hybrid_plan | Astra 计划工具 | Qwen 自行处理回执 | Qwen | 关 |
| hybrid_diagnose | Qwen | Astra 诊断工具 | Qwen | 关 |
| hybrid_both | Astra 计划工具 | Astra 诊断工具 | Qwen | 关 |
| hybrid_both_memory | 同 hybrid_both | 同 hybrid_both | Qwen | 开 |

主要对照及对应结论上限：q_only 与 astra_reference 给出**同接口小任务集**上的模型差距；q_only 与 hybrid_plan 检验规划帮助；q_only 与 hybrid_diagnose 检验故障理解；hybrid_plan 与 hybrid_both 检验增加诊断；hybrid_both 与 hybrid_both_memory 检验历史初始图像。交互效应另比较 hybrid_both 与单项工具组。任何差异都只能在相同任务/seed、原语、预算和模型版本下解释。纯 Astra 参考组的机器人决策由 Astra 完成，它不是“Qwen + tool”，论文展示时必须单列。

冻结的首轮工具规则：每 episode 最多 **1 次 Astra plan、1 次 Astra diagnosis、合计 2 次闭源调用**。首次计划可由 Astra 给出；后续重新规划由 Qwen 完成。动作回执被拒绝（ok=false）时可自动触发诊断；Qwen 在至少执行过一次机器人动作后，也可显式输出 consult 请求诊断，**即使最近回执 ok=true**，因为“命令执行了”不代表“任务有进展”。无先前动作的 consult 必须拒绝；用完一次诊断后后续错误由 Qwen 自行处理。两种触发路径及其成本分别记录，不能看见测试分数后修改。closed 工具输出只含子任务或故障类别、可观察依据、建议动作方向，不含机械臂动作数组、仿真隐藏状态或自报成功。相同字段格式可让 q_only 由 Qwen 自己输出；这样变的是认知模型/工具调用，而非问题表述。

## 数据流与成功归属

```text
RoboDojo 原生相机/机器人状态
  → Qwen Agent（观察、动作选择、回执）
  → 按白名单条件可调用 Astra 计划或诊断（建议，不操作机械臂）
  → Qwen 选择动作 → 安全守卫/动作转换 → RoboDojo 原生执行/评分
  → 最新观察与执行回执 → 下一轮 Qwen
```

进入子任务时，Memory Bank 捕获三相机初始图像和 task/episode/subtask ID；回忆只检索本 episode 中已发生的子任务 ID，把**历史初始图像**与**当前图像**明确标注后并列。不能把历史图像假装为最新观察。RoboDojo 不提供通用原生子任务完成谓词；策略的“进入下一子任务”只表示**未经原生验证的计划进度**，即使触发新快照也不等于任务成功。原生成功和终止只由 RoboDojo 决定。RoboICL 的示范片段与 LIVE 交互轨迹分别开关并保持各臂一致，避免将其收益误算进本项目初始图像记忆。

## 任务、分母与阶段

运行前先冻结动作模式与代码版本：使用 RoboICL 风格 ee_delta 和原生 IK 连续性保护时必须选 Mosi-AI 固定的 RoboDojo fork `00ef04cd245bfa76e6f1e0bd8383c82f7964f942`，因为官方 main `f57e1633ffed59761bc8e568cfe7530c4b33381c` 没有 `src.eval_client.ik_continuity`；使用官方 main 可走 joint 模式。**所有对照组须共享同一 fork、控制模式、动作转换与安全守卫**，不能把不兼容环境造成的差异归因于模型或 Astra 建议。

1. **端到端存活性：**Open 类 general_pickup 等简单任务各若干布局；先看图像是否新鲜、动作是否合法、官方结果是否保存、Qwen 是否出现非零成功。若 0/20，继续定位接线与任务难度，不能推断真实成功率恒为零。建议此阶段在同一小清单跑 q_only/hybrid_plan/少量 astra_reference。
2. **差距诊断：**选 Precision 的 build_tower、Long-Horizon 的 fill_pen_holder、Generalization 的标准/扰动布局，按相同 seed 跑完整条件，逐任务保留原生进度而不只看二值成功。这里的类别依据 RoboICL 发布的任务设置，正式任务身份仍以冻结 RoboDojo 官方清单为准。
3. **记忆机制：**先在 Memory 类 cover_blocks（起始可见颜色后来被遮挡）跑 hybrid_both 与 hybrid_both_memory；再用 imitate_sorting_sequence 测边界，因为需要记住运行中演示，单张子任务初始图未必含答案。加入错 ID、过时图像和无历史图像对照，但不得将测试集未来帧注入记忆。
4. **正式扩大：**只有端到端链路和模型角色核实后才覆盖官方 42 任务及原生每任务 episode 数；报告五维平均同时保留任务级分母。训练型 StarVLA 的 build_tower 非零分数仅作执行环境参照，不能直接与混合 Agent 组作因果比较。

20–50 个任务/seed episode 是先导量级，不是正式统计保证。完整 Astra 参考组先限制在预先选定的小任务集，以控制闭源使用量；正式五维报告如果缺少 F 组，应明确写“未测全量闭源差距”，而不能把外部论文总分拼上去。

## 指标、归因和审计

- **主要结果：**RoboDojo 原生 success、原生部分进度、失败后恢复；分 Open/Precision/Long-Horizon/Memory/Generalization 及标准/扰动布局。配对报告成功率差及不确定区间，注明任务和 seed 分母。
- **机制结果：**计划是否可执行、动作/工具 schema 有效率、物理动作被安全守卫拒绝率、重复失败动作、记忆调用的准确子任务、检索后是否纠错。失败归因为视觉理解、任务规划、动作选择/执行、反馈诊断、历史状态、环境/服务异常；抽样轨迹由人工盲审复核。
- **资源结果：**每模型调用数、输入/输出 token、闭源工具调用上限命中、墙钟时长、GPU/模型显存、费用；在成功率旁同时展示开销。服务不可用/超时/缺少原生 verdict 单列，不能记为机器人任务失败。
- **公平边界：**不同组不能偷偷获得物体真值、隐藏评分、未来观察或其他测试 episode 的记忆；策略端只接公开观察和合法回执。冻结相机布局、状态顺序、动作时域、控制频率、原语和安全限制。若用训练型 VLA 作低层 executor，各组必须使用同一 checkpoint，结论只落在上层 Agent。

首轮验收从代码层面证明：各条件的 Astra 权限/配额被真实拒绝而非只写在 prompt；闭源工具返回无效 schema 不会触发动作；记忆不会跨 episode 泄漏；未经验证的子任务推进不算原生成功；原生终止/成功从仿真器透传；日志可定位每次建议到后续动作。合成/假环境只能验接口和安全属性，**不能验证性能差距**。GPU 仿真、模型权重与服务环境就绪后再写 RoboDojo 结果。
