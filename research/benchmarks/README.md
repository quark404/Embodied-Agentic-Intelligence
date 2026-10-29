# RoboDojo 研究资料与证据边界

两个研究重点依次推进：**先分析通用 Qwen Agent 哪些具身能力最值得闭源模型补强，再依据证据设计可执行轻量环境上的低成本、直接多轮 Agentic RL。** 第一阶段独立产出能力地图、单项建议的成功率收益与成本，第二阶段验证自主能力与具身迁移；当前不考虑 SFT。首轮实验见[混合 Agent 协议](mixed_agent_robodojo.md)，能力归因与下一步交付物见[能力诊断与推进计划](../../docs/能力诊断与推进计划.md)，训练路线及论文优先级见[Agentic RL 扩展规划](agentic_rl_roadmap.md)，通俗版见[中文任务规划](../../docs/任务规划.md)。本目录的论文和公开成绩只用于选择任务与设定对照，**不是本项目的实测分数**。

[RoboDojo × Qwen 证据核查](robodojo_qwen.md)说明：RoboICL 发布的 RoboDojo runner 使用 GPT-6 Astra，公开资料尚未给出可直接复用的通用 Qwen Agent 全任务成功率；StarVLA 的 Qwen3-VL 骨干是经过机器人动作训练的 VLA，不能充当裸 Qwen Agent 成绩。PyRUA-Lean 的 Qwen 非零成绩来自 LIBERO-PRO、RoboTwin 和 RoboCasa 等其他 benchmark，说明其工具框架值得参考，却不能推断本项目的 Qwen/RoboDojo 成绩。

首轮固定 RoboDojo 原生任务/布局/seed、相机、动作模式、示范数量、物理预算和评分器。`general_pickup` 用于存活性先导；`build_tower` 检查精细操作；`fill_pen_holder` 检查长程流程；`cover_blocks` 与 `imitate_sorting_sequence` 检查起始图像记忆的有效范围。比较 `q_only`、`astra_reference`、`hybrid_plan`、`hybrid_diagnose`、`hybrid_both` 和 `hybrid_both_memory`，其中完整 Astra Agent 仅作小样本同接口参考。正式结论必须从本项目的原生 `_result.json` 和配对轨迹产生。

上游实现主要参考 [RoboICL](https://github.com/Mosi-AI/RoboICL)、[PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean)、[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) 和 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)。各论文/代码版本、公开数值与适用限制记录在上述两个专项文件中。
