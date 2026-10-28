# Embodied Agentic Intelligence

## 研究目标

在**相同 RoboDojo 任务、观察、机器人动作接口和物理预算**下，找出通用开源 Qwen Agent 与前沿闭源模型 GPT-6 Astra 的具身能力差距主要出现在哪里。随后检验：让 Astra 每个 episode 只作为少量**规划或失败诊断工具**，能否帮助仍由 Qwen 操作机器人的 Agent 缩小差距；增加的延迟、token 和费用是否值得。我们不预设混合方案会成功。

混合组的每一步机器人动作由 Qwen 选择。Astra 的结构化建议只能包含子任务计划或诊断文本，不能提交动作、读取隐藏评分，也不能宣称成功。另设一个**小样本完整 Astra Agent 参考组**，在相同接口下估计模型整体差距；它不算混合组。最终成功和部分分数只读 RoboDojo 原生 `_result.json`。

通俗的[中文任务规划](docs/任务规划.md)说明要证明什么；[详细实验协议](research/benchmarks/mixed_agent_robodojo.md)列出任务、归因边界和指标；[RoboDojo/Qwen 公开证据](research/benchmarks/robodojo_qwen.md)区分训练型 Qwen VLA 的成绩与尚未确立的通用 Qwen Agent 成绩。公开论文的分数不是本项目跑出的结果。

需要汇报时可直接打开[逐页 HTML 幻灯片](presentation/index.html)：含 Harness 流程、六组对照、任务级图像记忆与两阶段研究路线示意，也可在浏览器中导出 PDF。

后续研究将探索 **Qwen 具身 Agentic RL**：用未训练对照的失败轨迹定位值得学的能力，先做示范学习与有标准答案的局部训练，再训练“何时请求 Astra”，最后在独立 RoboDojo 闭环任务中测试**撤除 Astra 后是否仍有提升**。[第二阶段路线与论文借鉴](research/benchmarks/agentic_rl_roadmap.md)已单独规划；当前六组和代码仍是未训练基线，尚无训练后成绩。该路线不要求新建物理模拟器，但离线指标不能替代原生机器人成功率。

| 条件 | 规划 | 机器人动作 | Astra 介入 | 子任务初始图像记忆 |
| --- | --- | --- | --- | --- |
| `q_only` | Qwen | Qwen | 无 | 关 |
| `astra_reference` | Astra | Astra | 完整 Agent，仅小样本参考 | 关 |
| `hybrid_plan` | Astra 建议一次 | Qwen | 规划 | 关 |
| `hybrid_diagnose` | Qwen | Qwen | 动作后诊断一次 | 关 |
| `hybrid_both` | Astra 建议一次 | Qwen | 规划与诊断，各最多一次 | 关 |
| `hybrid_both_memory` | 同上 | Qwen | 同上 | 开 |

计划工具先于 Qwen 动作调用；诊断工具在动作被拒绝时自动触发，也允许 Qwen 在已有动作后主动请求，即使执行回执为成功。每个混合 episode 最多 **1 次计划、1 次诊断、合计 2 次 Astra 调用**，失败请求也占预算。后续重新规划由 Qwen 完成。记忆是在**每个子任务开始保存初始相机图像**，之后按同一 episode 的子任务 ID 显式取回；没有向量检索，也不读取别的 episode。RoboDojo 没有通用子任务完成谓词，策略的 `advance` 只是未验证的计划推进，绝不算原生成功。

## 首轮实验和验证

先用 `general_pickup` 等 Open 任务做端到端存活性测试，检查合法动作、相机新鲜度、原生结果和 Qwen 是否可能取得非零成绩。然后在相同 seed/布局上比较 `build_tower`（精细操作）、`fill_pen_holder`（长程任务）和 `cover_blocks`（记忆），后者再测试 `imitate_sorting_sequence`：运行中演示未必能靠一张子任务起始图记住。先导 20–50 个 episode 只作工程筛选；正式结论需冻结任务清单、官方每任务 episode 数、模型 checkpoint、版本、动作模式和统计分母。

主要报告相同任务/seed/布局的**配对成功率差、原生进度分、恢复率、动作拒绝率、记忆调用与模型调用成本**。`q_only` 对 `hybrid_plan` 定位规划，`q_only` 对 `hybrid_diagnose` 定位纠错，`hybrid_plan` 对 `hybrid_both` 看二者叠加，`hybrid_both` 对 `hybrid_both_memory` 看图像记忆；`q_only` 对 `astra_reference` 测小样本整体差距。即使 Astra 的计划正确，Qwen 的低层操作也可能失败，结论必须据轨迹分类和原生评分。各组使用同一个动作转换、安全守卫、相机、公开状态、物理步数和任务列表。

## 代码与本地契约测试

- [`embodied_agentic/harness.py`](embodied_agentic/harness.py)：有预算的规划—动作—观察循环，原生成功归环境所有。
- [`embodied_agentic/advisory.py`](embodied_agentic/advisory.py)、[`experiment.py`](embodied_agentic/experiment.py)：受限 Astra 规划/诊断、六个冻结条件、Qwen 动作端。
- [`embodied_agentic/image_memory.py`](embodied_agentic/image_memory.py)：按 task/episode/subtask 隔离的起始图像。
- [`embodied_agentic/robodojo_native.py`](embodied_agentic/robodojo_native.py)、[`integrations/xpolicylab/Embodied_Agentic`](integrations/xpolicylab/Embodied_Agentic)：RoboDojo 原生评测进程桥接、双臂动作转换和安全检查。
- [`embodied_agentic/schedule.py`](embodied_agentic/schedule.py)、[`reporting.py`](embodied_agentic/reporting.py)：冻结矩阵启动与原生结果配对汇总；缺失结果保留为缺失，不记作失败。

```bash
cd /path/to/Embodied-Agentic-Intelligence
python3 -m unittest discover -s tests -v
python3 -m embodied_agentic.experiment --config configs/robodojo_mixed.example.json
```

这些测试使用合成观察或假环境，验证条件权限、动作契约、预算、记忆隔离和评分归属；它们**不能证明模型的 RoboDojo 成功率**。

## 接入真实 RoboDojo

先按 [RoboICL 安装说明](https://github.com/Mosi-AI/RoboICL)配置 RoboDojo/XPolicyLab、Isaac 环境、机器人资产及模型服务。首选 RoboICL 风格的 `ee_delta` 控制，它要求 Mosi-AI 固定的 RoboDojo fork 提交 `00ef04cd245bfa76e6f1e0bd8383c82f7964f942`：代码会检查 fork 的原生 IK 连续性保护并在缺失时停止。官方 main 可选 `joint` 控制；**同一实验的所有组必须使用同一 fork 和模式**。此处未提供可运行的 GPU 仿真或真实 Qwen/Astra 端点，实际分数仍待实验。

将本仓库 policy 目录以符号链接安装到 RoboDojo 的 XPolicyLab；仿真环境中还需有 Python `numpy` 和 `Pillow`。`configs/robodojo_mixed.example.json` 的 URL、模型名、密钥环境变量要换成真实服务，冻结副本后再运行。示例：

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
ln -s "$(pwd)/integrations/xpolicylab/Embodied_Agentic" \
  "$ROBODOJO_ROOT/XPolicyLab/policy/Embodied_Agentic"
python3 -m embodied_agentic.schedule \
  --robodojo-root "$ROBODOJO_ROOT" \
  --config configs/robodojo_mixed.example.json \
  --tasks configs/robodojo_pilot.example.json \
  --policy-env /path/to/policy/python --sim-env /path/to/simulator/python \
  --run-id pilot01 --action-mode ee_delta \
  --output runs/pilot01/schedule.json
```

上面只生成计划。确认模型服务与仿真环境后，在相同命令尾部添加 `--resume --execute` 串行执行已冻结计划；完成的原生 run 会跳过，未完成的单 run 要先按 RoboDojo 自身的续跑流程处理。每个任务/条件使用独立原生结果目录。若用官方 main，改为 `--action-mode joint`。评测后运行：

```bash
python3 -m embodied_agentic.reporting \
  --schedule runs/pilot01/schedule.json \
  --robodojo-root "$ROBODOJO_ROOT" \
  --output runs/pilot01/report.json
```

报告同时保留所有原生计分记录和主比较的有效样本：缺失 harness 轨迹、未完成 native run、服务或仿真故障不进入主比较分母；模型输出/决策违反协议则计为模型失败。可在每个模型配置中填入 `input_usd_per_million_tokens` 和 `output_usd_per_million_tokens` 估算美元费用；缺少单价时金额标为未知，token 与时长仍会记录。源论文、上游接口与适配器历史见[研究目录](research/benchmarks/README.md)及[上游集成说明](docs/上游框架集成.md)。
