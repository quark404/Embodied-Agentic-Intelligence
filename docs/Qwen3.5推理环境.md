# Qwen3.5-27B 本地多模态推理

配置日期：2026-10-29。按项目要求使用 **Qwen3.5**，不部署 Qwen3.8；不涉及 SFT 或 RL 训练，也不更改六组实验的工具权限。

## 本机验收状态

2026-10-29 已在物理 GPU 6 的单张 H20 上启动服务，tmux 会话为 `qwen35-27b`。文本 JSON、三图像识别、native function call、现有 Harness planner/actor 四项真实端点检查全部通过，结果保存在 `runs/qwen35/smoke.json`；本地 40 项测试通过。权重约 51.75 GiB，独立运行环境约 9.2 GiB，未下载 Qwen3.8 或第二个大模型。

这是 **BF16 / non-thinking / 32K / eager 的接口先导验收**，不是机器人任务成功率或正式吞吐基准；Astra 与 RoboDojo 原生仿真仍未接通。服务占用 GPU，后续不使用时按下文方式正常停止。

## 模型选择

首个主基线选择 `Qwen/Qwen3.5-27B` 的 **BF16 原始权重**，不是量化版、Base 版或 VLA。理由是先用单张 H20 建立可重复的多模态/工具接口，为后续能力诊断保留资源；不预设其机器人成功率。大模型可在链路稳定后作为规模对照，但不能把不同规模结果混在同一个 Qwen 条件里。

官方 Qwen3.5 主力型号包括 27B dense、35B-A3B、122B-A10B、397B-A17B；没有对应的官方 50–100B 主力型号。35B/122B 等 MoE 名称中的总参数与激活参数不能混为一谈。后续若要明显更大的同系列对照，再评估 122B-A10B 的多卡预算，本次不同时下载。

- 权重 revision：`fc05daec18b0a78c049392ed2e771dde82bdf654`。
- 官方权重文件总计约 55.56 GB（51.75 GiB），另需运行环境和缓存空间；实际显存还包含视觉编码、激活和 KV cache。
- 独立 Conda 环境：`$HOME/miniconda3/envs/qwen35-vllm`，Python 3.12。
- 固定依赖：`vllm==0.23.0`、`transformers==5.8.1`；完整安装版本可保存到 `runs/qwen35/pip-freeze.txt`。
- 本用户共用权重缓存：`$HOME/.cache/huggingface/hub`，不把权重复制进 Git 仓库；不修改 `juice` 环境。

## 重建环境与下载

在仓库根目录执行；以下命令会下载运行依赖和约 55.6 GB 权重。下载脚本检查剩余空间、固定 revision，并复用已有 HF 缓存。安装不用额外 wheel 缓存。

```bash
conda create -n qwen35-vllm --override-channels -c conda-forge python=3.12 pip -y
conda activate qwen35-vllm
PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL='' \
  python -m pip install --no-cache-dir uv
UV_DEFAULT_INDEX=https://pypi.org/simple uv pip install --python "$CONDA_PREFIX/bin/python" \
  --no-cache -r requirements/qwen35-serving.txt
HF_HUB_DOWNLOAD_TIMEOUT=120 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  python scripts/download_qwen35.py
```

本机默认包镜像曾返回 503，因此这里对这些安装命令单独指定官方 PyPI，不修改用户全局配置，不关闭 TLS 校验。下载脚本可重复执行以复用/补齐缓存，不运行第二份权重复制。

## 启动与连接

```bash
conda activate qwen35-vllm
export QWEN_GPU=0
bash scripts/serve_qwen35.sh
```

默认绑定本机 `127.0.0.1:18000`，API base 为 `http://127.0.0.1:18000/v1`，服务模型名为 `Qwen/Qwen3.5-27B`。不对外网开放，也不配置闭源 API 凭证。**必须显式设置已分配给自己的 `QWEN_GPU`；上面的 0 只是示例，不再默认选择原服务器的 GPU6。** 脚本设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 并使用数字编号；若继承了 `CUDA_VISIBLE_DEVICES`，不允许选取其范围之外的设备，UUID 掩码无法与数字安全对应时拒绝覆盖。重启前要重新检查分配和空闲情况。首次候选 GPU1 在启动期间被其他任务使用，因此未强行共享。显存占用不可读或明显占用时拒绝启动，不停止其他人的作业；检查不等于集群调度预约。可用 `QWEN_GPU`、`QWEN_PORT`、`QWEN_ENV` 适配新机器，不影响已经启动的原服务。

独立终端中运行，或用命名 tmux 会话保留服务：

```bash
mkdir -p runs/qwen35
tmux new-session -d -s qwen35-27b \
  "QWEN_GPU=${QWEN_GPU:?先选择已分配的GPU} bash scripts/serve_qwen35.sh >runs/qwen35/server.log 2>&1"
tail -f runs/qwen35/server.log
```

不要在已有同名服务运行时重复启动。需停服务时，先进入 `tmux attach -t qwen35-27b`，按 Ctrl-C 正常结束，再确认对应 GPU 进程退出；不要批量杀掉其他用户的 Python 进程。

## 首轮服务参数与研究边界

| 参数 | 本次值与用途 |
| --- | --- |
| 精度与并行 | BF16、TP=1；不引入量化误差作为额外变量 |
| 上下文 | 32,768 tokens，包括输入与输出；这是工程先导设置，不是模型最大能力配置 |
| 并发 | 最多 2 条序列，显存利用率 0.85；后续需实测吞吐后再调 |
| 图像 | 每次最多 12 张，每图 `max_pixels=1048576`；禁用视频，本轮聚焦相机帧 |
| 思考模式 | 默认 `enable_thinking=false`，先打通低延迟 JSON/工具协议；不是“该模型不支持思考” |
| 工具接口 | `qwen3_coder` native tool parser；现有 Harness 仍走 JSON 消息协议，不自动改成 native function calling |
| 执行模式 | `--enforce-eager` 用于先导接线，尚未优化 CUDA Graph/编译吞吐 |

Qwen3.5 默认支持 thinking，官方对长推理建议更长上下文；因此本服务只能称 **27B/non-thinking/32K 先导基线**。正式能力差距实验要另行冻结思考开关、采样、输出预算、上下文和图像处理，必要时加 thinking 对照，不能用先导配置代表开源模型上限。当前 Harness 固定 `temperature=0`，也不是照搬模型卡的通用采样建议。提高性能前先记录这些变量，不能在配对实验中悄悄改变。

现有 Harness 每次发送当前相机和显式回忆图像；图像尺寸和数量受服务上限约束。输入超过预算时应报告配置/协议问题，不静默丢相机或截断任务历史。

## 验收与接入项目

```bash
conda activate qwen35-vllm
python -m scripts.check_qwen35
python -B -m unittest discover -s tests -v
python -m embodied_agentic.experiment \
  --config configs/robodojo_qwen35_27b.local.example.json --condition q_only
```

`runs/qwen35/smoke.json` 记录真实端点的文本 JSON、三图像、native function call 和现有 Harness planner/actor 检查结果、耗时及 token。只有报告 `passed=true` 才算这些接口检查通过；`experiment` 命令仅验证配置，不启动机器人环境。`runs/` 已在 Git 忽略规则中。

接线时发现模型会给规划 JSON 加 Markdown 代码围栏，与现有严格解析器冲突。因此在通用 planner/actor 提示中明确要求 raw JSON、禁止围栏与额外文字；该提示对开闭源适配器相同，不通过剥除围栏、静默修复或重试掩盖错误。正式实验要固定更新后的提示版本。首次 CUDA 内核编译属于部署冷启动开销，应与预热后的每次请求延迟分开记录。

新配置的 Qwen 端指向本地服务；Astra 部分仍是占位符，未配置前只运行 `q_only`。这些检查不等于 RoboDojo 成功率，接下来仍需按原生环境接线流程运行独立 episode。

## 官方核对来源

- 模型卡：`https://huggingface.co/Qwen/Qwen3.5-27B`
- 固定权重信息：`https://huggingface.co/api/models/Qwen/Qwen3.5-27B`
- vLLM 固定版本模型注册：`https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/registry.py`
- 依赖版本：`https://pypi.org/pypi/vllm/0.23.0/json`

本次以实际固定版本安装和本机检查为准，不把模型卡早期的 nightly 安装示例当作当前唯一部署方式。
