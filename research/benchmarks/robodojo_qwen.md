# RoboDojo × Qwen：环境、论文与可用 harness

核查日期：2026-10-28。本项目现在以 **RoboDojo-Sim 为主评测**。下文严格区分：① 通用 Qwen VLM 通过 Agent harness 规划/调用工具；② 用 Qwen 初始化、再训练动作头的 VLA；③ 使用闭源模型的 Agent harness。三者都可以在 RoboDojo 上运行，但不能混成一个“Qwen Agent 成功率”。本次没有 GPU 仿真，所列成绩均来自标注的上游材料，不是本项目结果。

## 结论先行

**目前最接近我们目标的公开 Qwen Agent 实验是 `small-vlm-openwam-robodojo`：Qwen3.5-2B + RPent + OpenWAM，跑完 RoboDojo `fill_pen_holder` 单个 episode，原生判定失败。** 它证明 Qwen、Agent 工具循环、动作模型和官方仿真可接通，尚未证明这种 Agent 有非零成功率。它没有对应正式论文，应作为可复查的工程先例，而非论文基线。

**存在 Qwen 系模型在 RoboDojo 的非零成绩，但来自训练型 VLA。** StarVLA 以 Qwen3-VL-4B-Instruct 为骨干，在 `build_tower` 官方 50 episode 参考中报告 QwenOFT 36%、QwenGR00T 14%、QwenPI_v3 56%。这说明 benchmark 和 Qwen 系视觉骨干并非天然不兼容，却不能推出通用 Qwen VLM 独立规划、直接操作或调用本项目工具也能达到这些成绩。

因此主实验仍应做 **Qwen Agent 存活性门槛**：先在简单 Open 任务用原生评分跑通并观察能否成功；若失败，定位是图像、动作格式、机器人控制、时间预算还是模型决策；确认可运行后再加入任务级图像记忆。不要在 0/1 的失败案例上推导“加记忆也必定无效”，也不要用 VLA 的非零成绩替代 Agent 基线。

## 官方环境与评测入口

| 组成 | 本次核对的上游 | 我们要沿用的接口 |
| --- | --- | --- |
| [RoboDojo 论文与仿真仓库](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/f57e1633ffed59761bc8e568cfe7530c4b33381c)（[arXiv:2607.04434](https://arxiv.org/abs/2607.04434)） | 42 个仿真任务、五类能力；ARX X5 双臂、Isaac Sim；仓库提供任务、资产、原生终止/评分，不承载模型服务 | 使用其官方 `scripts/robodojo.sh doctor/eval/summarize` 和任务评分；评测结果不能由模型口头判断 |
| [XPolicyLab 论文与策略接口](https://github.com/XPolicyLab/XPolicyLab/tree/c43a74653bec480ebe82c84770664e8035bc56c9)（[arXiv:2608.09892](https://arxiv.org/abs/2608.09892)） | 一个策略目录对应 `eval.sh`、`deploy.yml`、模型服务；RoboDojo client 通过 WebSocket 请求动作 | 本项目要包装成 XPolicyLab policy，交由官方 client 推进仿真与记分 |
| [官方 Docker 说明](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/f57e1633ffed59761bc8e568cfe7530c4b33381c/docker/README.md) | 仿真侧 Isaac Sim 5.1、IsaacLab 2.3、Python 3.11；模型服务可在容器外 | Linux/NVIDIA GPU、Vulkan/CUDA 和完整 Assets；官方 Docker 镜像约 200 GB，建议预留 300 GB 以上磁盘；本云环境未安装这些资源 |

官方仓库将 42 个任务分为 Generalization、Precision、Long-Horizon、Memory、Open 五类；任务变体/配置数可大于独立任务数，不能把“42 tasks”与某些 harness 的“54 configurations”直接当成同一分母。开发阶段可运行 `doctor`、单任务 `eval --eval-num 1`；正式结果应按固定任务、原生 episode 数、seed、评分规则运行和汇总。`--dry-run` 只检查接线命令，不代表成功。官方认可的排行榜还需其提交/核查流程。

## 与本课题直接相关的论文和框架

| 论文/实现 | RoboDojo 上使用的模型 | 对本课题的作用 | Qwen RoboDojo 证据 |
| --- | --- | --- | --- |
| [RoboICL: Embodied In-Context Learning with GPT-6 Astra](https://arxiv.org/abs/2609.34261)；[代码](https://github.com/Mosi-AI/RoboICL/tree/17b379a6deea33f71bd31e05bd60d1daf627acfb) | GPT-6 Astra；0-shot/1-shot，示范和 LIVE 轨迹记忆 | **主要 harness 参考**：三相机观察、双臂动作转换、安全执行、交互反馈、示范/历史消融 | 无：发布 runner 的模型不是 Qwen |
| [HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents](https://arxiv.org/abs/2610.23479)；[仓库内论文 PDF 与代码](https://github.com/HexaFuture/HexaShell/tree/d1a2a24dc4c528a652b359fd689398530ef941d8) | Codex + GPT-6 Astra；官方 42 任务 46.9% 成功 | **主要工具 harness 参考**：薄命令行、可复用工具、失败后改工具；论文 PDF 已核读 | 无：公开实验使用 GPT-6 Astra |
| [PyRUA-Lean: Python for Lean Robot-Use Agents](https://arxiv.org/abs/2610.01939)；[RoboDojo 接口](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/fdf6e870654254e2c0c891ee407313fa48bdd0f9/docs/robots/robodojo.md) | RoboDojo 公开分数使用 GPT-6 Astra（官方 50.35% 平均成功率） | **主要代码/原语参考**：机器人 Python cells、原语回执、XPolicyLab 接口 | 其 Qwen3.5 基座分数在 LIBERO-PRO/RoboTwin/RoboCasa，不能移植成 RoboDojo 分数 |
| [StarVLA: A Versatile Vision-Language-Action Model](https://arxiv.org/abs/2604.05014)；[XPolicyLab RoboDojo 适配及结果](https://github.com/XPolicyLab/XPolicyLab/blob/c43a74653bec480ebe82c84770664e8035bc56c9/policy/StarVLA/README.md) | Qwen3-VL-4B-Instruct 骨干 + 在机器人数据上训练的 OFT/GR00T/PI_v3 动作头 | **非零控制参照**：验证 Qwen 系骨干 + 正确动作接口能完成任务；可作为单独的低层执行臂 | 有，仅 `build_tower` 任务的发布成绩；不是通用 Qwen Agent |
| [XPolicyLab 的 GPT-6 Astra Direct EEF / RoboProbe](https://github.com/XPolicyLab/XPolicyLab/blob/c43a74653bec480ebe82c84770664e8035bc56c9/policy/GPT_6_Astra_Direct_EEF/README.md)；[论文](https://arxiv.org/abs/2609.24170) | GPT-6 Astra，RGB + `move_eef` | 直接操作的动作接口参照，便于设计同原语 Qwen A 组 | 无 |

HexaShell 论文第 4 节以同一 GPT-6 Astra 的无工具基线约 22.5% 对比其工具 harness 的 46.9%；这是**工具机制与能力边界的研究范式**，不能当作 Qwen 的提升量。RoboICL 的记忆是动作/回执/观察的交互轨迹；我们 D 组是子任务开始时的原始图像，作用和信息量不同，必须单独消融。

## 已实际用 Qwen 接入 RoboDojo 的公开项目

| 项目 | 模型角色与动作接口 | 可核对的结果 | 作为参考的等级 |
| --- | --- | --- | --- |
| [small-vlm-openwam-robodojo](https://github.com/MmMaiIIi/small-vlm-openwam-robodojo/tree/f6e7f983d21403e8147e68ec1640c332f17ab62f) | 本地 Qwen3.5-2B 做 RPent planner/tool loop，OpenWAM 执行低层动作 | [`fill_pen_holder`、seed 0 的完整记录](https://github.com/MmMaiIIi/small-vlm-openwam-robodojo/blob/f6e7f983d21403e8147e68ec1640c332f17ab62f/docs/RPENT_UNRESTRICTED_REPORT.md)：18 次有效工具调用、1100/1100 原生动作步，最终失败。**只有一个 episode，无成功率 sweep，也无论文。** | **最直接的 Qwen Agent 工程先例**：复用它的接口审计与失败定位；不能作为非零成功基线 |
| [StarVLA / XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/c43a74653bec480ebe82c84770664e8035bc56c9/policy/StarVLA/README.md#released-hugging-face-robodojo-checkpoints) | Qwen3-VL-4B 初始化，训练型 VLA 直接输出 14 维关节动作块 | `build_tower` 发布的官方 50 episode 参考：OFT 36%、GR00T 14%、PI_v3 56%；同一 README 给出本地 10-layout 复测 4/10、1/10、5/10。**仅此任务，非全榜单平均。** | **主要非零控制参照**：不能当成通用 VLM Agent 结果 |
| [Embodify](https://github.com/YidaYang/embodify/tree/6be154471801f4cea47f3df1f46b000476b3eeb6) | RoboDojo MCP 接口可连包括 Qwen 在内的模型 | README 宣称兼容；没有查到 Qwen 在 RoboDojo 的公开实测分数或论文 | 接口备选，不是结果基线 |

**排除误判：** [Spotter 论文/代码](https://github.com/zqc3117/Spotter/tree/84f5cf75b5ad05d335dfec9ec6d47b52b269dcdb) 确有本地 Qwen 和机器人干预循环，也存在 RoboDojo 相关代码，但 README 公开成绩来自 RoboCasa/RoboTwin 2.0，不能称为“Qwen 已在 RoboDojo 得分”。XPolicyLab 中的 ABot-M0、Awomo-0.5、InternVLA-A1 等也使用 Qwen 骨干，但属于训练型 VLA；它们的存在不证明通用 Qwen Agent 在 RoboDojo 有可复现非零分数。不能把“代码可配置 Qwen”“Qwen 骨干动作模型有成绩”“Qwen Agent 有成绩”混为一谈。

## 我们的 RoboDojo 首轮设计

1. **先打通官方原生入口。** 固定 RoboDojo/XPolicyLab 提交、资产 revision、Isaac 环境、三个 RGB 相机、14 维状态/动作顺序、动作 horizon 与原生评分；先执行 `doctor` 和单任务 `eval`，保存 `result.json`/原生 trace。本仓库现有 `RoboICLAdapter` 只接已初始化环境，没有组成官方 XPolicyLab policy server；这项接线仍待 GPU 环境。
2. **先验证 Qwen Agent，而非先测记忆。** 用 `general_pickup` 之类的 Open 类简单任务检查有效动作、是否能抓到目标、是否出现至少一个原生成功；把 `build_tower` 保留为训练型 StarVLA 的控制参照，不让不同动作头成绩和 Agent 成绩直接比较。最初 20–50 个固定任务/seed episode 只作工程试跑；0/20 仍不足以证明真实成功率恒为零。
3. **再测规划与记忆。** 同一 Qwen、同一底层原语与执行预算下配对 A/B/C/D。记忆优先选择 `cover_blocks`：初始可见的三色方块后来被杯子遮住，任务要求按颜色顺序揭开，因此子任务初始图像有明确可检验用途。`imitate_sorting_sequence` 的关键顺序来自**运行中**的对方机器人示范，单张子任务初始图像可能不含答案，应列为记忆机制的边界案例。`fill_pen_holder` 等 Long-Horizon 任务用于看规划和纠错，但它在现有 Qwen3.5-2B/RPent 单例中失败，不能预设可成功。
4. **归因门槛。** 若 Qwen Agent 所有简单任务都没有有效动作或原生成功，先修动作转换、提示词/输出格式、相机顺序与控制原语，必要时采用有成绩的训练型 VLA 作**同一执行工具**，由 Qwen 负责上层规划。该条件须明示为“Qwen planner + VLA executor”，不能声称 Qwen 独立具身操作成功。只在可运行的固定基线上评估新增规划/记忆的增量。

论文正文可直接核读的是 HexaShell 仓库随附 PDF。RoboDojo、XPolicyLab、RoboICL、PyRUA-Lean、StarVLA 的论文条目和 arXiv 链接已核对，但本次 arXiv PDF 站点不可访问；有关它们的具体协议和数字均引用对应发布仓库/结果表，未将未读取的论文正文当作独立证据。若日后取得 PDF，应补录论文版本、表号和与代码发布结果的差异。
