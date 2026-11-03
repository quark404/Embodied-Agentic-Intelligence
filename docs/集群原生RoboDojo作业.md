# 集群原生 RoboDojo + Qwen Harness 作业

更新：2026-10-30。**完整入口已经提交，但真实原生成功仍未验收。** 不把环境预检、示例方块渲染、CPU 测试或 endpoint 握手算成任务成功。

**15:49 更新**：不再把开发机代理作为后续前提。AIDC 已有内网 wheel 实际读取证据，预置 RoboDojo 镜像正核验，模型镜像内容身份锁定已实现；详见[无代理资源复用](集群无代理资源复用.md)。本文其余作业表和公网安装条件保留为历史记录，不代表仍按同一路线重复提交。

## 当前三条作业

| 队列 | 作业 ID | 实际工作 | GPU / 最长运行时间 | 平台接受时间（北京时间） |
| --- | --- | --- | --- | --- |
| AIDC | `aidc01-dd6fc78aa006` | 自身预检 → 隔离安装 → bridge → RGB → Qwen → 原生 episode | 3 / 240 分钟 | 2026-10-30 14:18:33 |
| 北京 debug | `bcloud-bj-zone1-b9422ca13f3e` | 自身预检 → 隔离安装 → 三视角诊断渲染 | 1 / 120 分钟 | 2026-10-30 14:18:58 |
| 廊坊 | `acloud-langfang-6bea5a5ecc93` | GPU、库、存储与网络预检 | 1 / 15 分钟 | 2026-10-30 14:20:01 |

14:24 平台回查均为 `Queuing`。全部 low、零平台自动重试、每个队列只有一条本项目在途作业。三条工作量与卡数不同，**结束时间不能直接用于比较集群速度**，应分别报告排队和执行耗时。

AIDC 原纯预检 `aidc01-a34d901df807` 已于 13:55:10 自然失败，未被停止；真实日志定位到 `/job_data` 路径解析后的边界校验。北京和廊坊的旧包在分别核对回执、当前用户、队列及未开始状态后撤换，平台确认 `Stopped` 才提交新包。所有旧包与审计保留，未修改任何在途冻结包。

## 存储修复不是绕过隔离

AIDC 的真实诊断显示目标路径位于平台 `/running_root` 的 `fuse.juicefs` 挂载内。最初 `job_local` 实现要求 `/job_data/...` 和其规范路径逐字一致，因此把平台路径映射当成越界，尚未创建结果就退出。

现在仅允许 `/job_data` 本身解析到真实挂载的 `/running_root` 或其目录之下；任意其他根别名、未挂载的同名目录、run 内部软链接、目录复用或跨 run 路径仍拒绝。代码只独占创建 `eai_<run_id>` 新目录，并验证上传包与结果目录的同一原始标记。不会改平台链接、挂载或权限，也不操作其他人的路径。

`job_local` 表示本 run 独立使用，不保证底层一定是本机临时磁盘；现有冻结包的 `storage_ephemeral` 字段是保守使用假设，不是持久化探测结论。方法不依赖跨作业复用这些路径。结果通过有 run 身份和内容校验的日志协议返回控制端，控制端仍仅保存本用户的小型记录。

## 完整入口做什么

入口：`cluster_eval/native_job.py`，模板：`configs/aidi_5090_native.example.json`。

1. 仅在显式确认的上传包中运行；要求三个实际可见的 RTX 5090，当前入口只接受索引 `0,1,2`。先检查新 run 路径及至少 240 GiB 空闲空间，不能在开发机误执行。
2. 在该 run 私有目录建立仿真与 bridge venv，使用固定 Python、Torch、Isaac、CuRobo/IsaacLab 源码。不执行上游 sudo、Conda 初始化、系统驱动安装或 `submodule --remote`。
3. 给仿真环境补齐 XPolicyLab 导入所需的 `h5py`。视频使用 `imageio-ffmpeg` 在该 venv 中自带的可执行文件，只向本 run 的 `bin/ffmpeg` 建链接，不安装系统 FFmpeg。
4. 真正执行 bridge 的握手/reset 检查，再在卡 0 做实际三视角 RGB 诊断。没有通过渲染，不继续加载模型。
5. 新建独立 Qwen venv，使用 `requirements/qwen35-serving-cu129.txt`。先检查依赖和两卡 CUDA 运算，再下载固定 revision 的资产和 Qwen3.5-27B。资产必须与固定清单的 15,365 个文件、41,269,513,111 字节吻合；不下载其他数据目录。
6. 卡 1、2 使用 BF16 张量并行；模型不量化、不替换为小模型。当前上下文 16,384、并发 1，服务只绑定作业内 `127.0.0.1` 的空闲端口。运行真实文本、三图、工具调用及 harness planner/actor endpoint 检查。
7. 通过上游原生启动脚本，执行 `general_pickup / seed=0 / eval_num=1 / q_only / ee_delta`。Qwen 输出由本仓库 harness 和动作守卫执行，不使用固定动作轨迹、假模型、闭源模型替代或合成环境。
8. 读取上游实际生成的 `_result.json`，并与 harness 摘要、完整动作/模型调用 trace、模型 revision 和三视角末次观测对应。失败原样记录，不改任务、奖励或成功判定。

全流程设置约 220 分钟的执行截止时间，给导出与清理留余量；命令超时只终止本入口创建的进程组。平台最长 240 分钟，不会因为轮询超时重提作业。没有 SFT 或 RL 训练。

## 与上游接入的关键约定

- 显式 venv 通过其中的 `python`/`python3` 调用固定 RoboDojo 的 `scripts/eval_policy.sh`，避开上游强制 `conda activate` 的包装层，但不改原生任务和评分代码。
- `ROBOICL_DATA_ROOT=<runtime>/assets` 对应真正下载的 `Assets/`；`ROBOICL_RESULTS_ROOT=<runtime>/native-results` 对应原生 `eval_result/`。不能假定结果总在源码根目录。
- bridge 和仿真分别使用 WebSockets 14.2 与 12.0；Qwen 的 Torch/CUDA 依赖不进入仿真 venv。两者使用不同 GPU，Qwen 编译缓存与仿真缓存分开。
- Bash 自动重试为 0。固定上游版本的内部 fatal restart 上限为 3，入口通过其已有计数变量将起始值设为 3，使首次 fatal 错误直接退出，不修改上游文件。
- 当容器进程为 root 时，只设置 IsaacLab 官方容器配置使用的 `OMNI_KIT_ALLOW_ROOT=1`。不关闭驱动检查，不用 sudo。
- `q_only` 不调用闭源顾问，原有实验定义中的图像记忆增强仍属于单独的对照条件。本次额外保存末次相机帧仅供审计，不会改变模型输入或偷偷加入额外感知工具。

## 返回哪些真实证据

`EAI_NATIVE_V1` 使用有大小上限的分块记录及最后的 commit，绑定本 run 与原始标记 SHA-256。控制端缺块、内容冲突、跨 run 或路径穿越均不接收；它不把任意 JSON 中的 `success=true` 当作成功。

通过校验后，控制端结果目录的 `native-artifacts/` 包含：

- `_result.json`：上游原始评分文件。
- `summary.json`、`trace.jsonl`、`experiment.json`：本次 harness 摘要、完整结构化调用/动作记录与条件。
- `schedule.json`：唯一的任务、seed、动作类型与 native run 身份。
- `model-download.json`、`model-check.json`：固定模型来源与实际 endpoint 检查。
- `cameras/cam_head.png`、`cameras/cam_left_wrist.png`、`cameras/cam_right_wrist.png`：harness 最后观测的原始 PNG，校验 CRC、尺寸、RGB 格式和摘要。

日志中额外记录实际包版本。单个文件最大 4 MiB，所列文件合计最大 10 MiB，编码后的 native 记录最大 16 MiB；超额明确失败，不静默截断。原生视频留在作业内，没有声称通过该小型协议完整归档视频。

评分文件、harness 任务/布局、Qwen planner/actor 调用以及非零物理执行步数必须互相一致。**原生 episode 完成与任务成功分开记录**：正常结束但未完成任务，仍不能达到此次目标。若运行失败，能收集到的白名单 trace 等以 `diagnostic_files` 返回，仅用于排错，不能让成功门槛通过。

CPU 单测通过只验证这些约束。最终还要确认平台作业成功、实际返回的原生结果成功、查看真实图像，并核对完整 trace；单个任务通关不是跨任务成功率，也不是 agentic RL 增益证据。

## 集群出网修复（2026-10-30）

北京 `bcloud-bj-zone1-b9422ca13f3e` 于 **14:35:59** 启动、**14:38:33** 失败。实际证据已区分两类问题：

- 已解决：平台 `/job_data` 映射到 `/running_root` 的合法路径被接受；5090、图形库、私有目录验证和 `inventory` 校验取回通过，安装真正进入 `isolated_uv`。
- 当前故障：PyPI、NVIDIA PyPI、GitHub、Hugging Face 的直连探测均报 `URLError`；pip 对 `/simple/uv/` 连续读超时，最终的“找不到 uv 版本”是索引不可达的结果，不据此修改版本。
- 修复：新增显式 `--egress-proxy`，只把无凭据的 HTTP(S) 代理地址和受限 `no_proxy` 加入作业白名单。保留 localhost 与平台内部域名直连，不转发账户 token、不继承开发机全部环境、不自动改用 Hugging Face 镜像、不关闭 TLS 校验。
- 失败早停：自身预检未取得依赖入口的 HTTP 可达证据，就不创建安装环境、不等待 pip 重试。新包还对固定 uv wheel 的实际 `files.pythonhosted.org` URL 做 HEAD，要求返回 200；仅索引页面能打开不再算通过。根页面 HEAD 返回 HTTP 错误仅代表请求可达，不证明下载授权、其他 wheel 或资产可用，仍由后续真实安装验收。

北京修正版 `render_proxy_20261030_144549` / **`bcloud-bj-zone1-55c355bc13fa`** 于 **14:46:12** 接受、**14:46:23** 启动、**14:47:08** 失败：配置代理后四个公网入口仍不可达，安装前早停生效，没有开始创建 venv。这证明该代理设置尚不能解决北京出网，不能宣称连接修复。没有启动或修改代理服务、终止他人进程或修改冻结包。

后续只读网络诊断 `network_diag_20261030_145451` / **`bcloud-bj-zone1-2e677e871273`** 于 **15:09:25** 启动、**15:10:35** Succeeded：取回并校验了诊断，不代表网络通过。代理变量确实已生效，但到代理的 TCP 连接 5 秒超时；公网直连、内网 PyPI 也超时。它没有安装任何依赖。诊断仅返回无凭据 host/port、异常类型/errno/CONNECT 状态与耗时，不打印原始异常中的 URL、凭据或完整环境。

### 实际集群差异与下一轮条件

| 证据 | 北京 | AIDC |
|---|---|---|
| GPU 与本 run 存储验证 | 1×5090，通过 | 3×5090，通过 |
| 开发机已有代理 | 变量已生效；TCP 超时 | 变量已生效；TCP 超时 |
| 内网 `pypi.hobot.cc` 直连 | 超时 | HTTP 200 |
| 公网 PyPI 索引直连 | 超时 | HTTP 200 |
| 公网实际 wheel 下载 | 无可达证据 | 无代理时 `files.pythonhosted.org` 连接重置；加代理后超时 |
| Isaac 渲染／Qwen／原生成功 episode | 尚未执行 | 尚未执行 |

AIDC 原生直连版 `aidc01-dd6fc78aa006` 于 **15:09:00** 启动、**15:11:12** 失败，已取得三卡真实证据，失败阶段为 `isolated_uv`。追加的代理原生版 `native_proxy_20261030_151258` / **`aidc01-3ae875a396a4`** 于 **15:13:36** 接受、**15:13:47** 启动、**15:15:00** 失败，网络早停在创建环境之前触发。两份都没有伪造 episode 或降低验收标准；回执、脱敏失败摘录和取回文件保留在各自 run / result 目录。

后续条件明确如下，避免继续提交相同失败配置：

1. 北京、AIDC 需要一个**从集群作业可达且获授权的出网代理**，或已经完整准备好的内网依赖／模型／RoboDojo 资产源；控制机能访问公网不能替代这一条件。不能只把 PyPI 换成内网源，就声称 GitHub、NVIDIA、HF 及实际文件下载已解决。
2. 获得新入口后，先以同一 profile 的新 run 验证索引、真实 wheel、NVIDIA、GitHub、HF 请求与后续下载。不要修改冻结包、关闭证书校验、启动控制机转发服务或修改平台网络/驱动。
3. 廊坊 `acloud-langfang-6bea5a5ecc93` 在 **15:18:49** 仍 Queuing；保留原排队位置与监督，不根据另外两地的失败推断它的网络情况。
4. 只读核对历史本项目缓存：仿真 venv 不完整，资产约 70 MiB，远小于固定完整资产清单；不能将这些残留当成现成离线环境，也未恢复本机安装/下载。

最新工作树的 Qwen／仿真子进程保留内部域名的 `NO_PROXY` 并确保 loopback 直连，避免后续又把内网源送到外部代理；这一小修复在最后一份冻结包之后，仅对以后新包生效。**149 项 CPU 回归测试通过，不等于 GPU 或原生任务验收。**

只读核对本用户已成功的 AIDC 训练作业 `aidc01-3fd55865d784`，其实际依赖下载域名是 `pypi.hobot.cc`，不是公网 PyPI；这只是内网源候选依据，不能证明北京可达或镜像包含 Isaac/CUDA/GitHub 资产。开发机在 15:00 的 HEAD 检查能经既有代理访问公网 PyPI/GitHub，也不能代替集群结果。未启动本机安装或资产下载。

跨服务器使用当前服务器获授权的代理地址，不能照搬私人内网地址：

```bash
"$AIDI_PYTHON" -B -m cluster_eval.prepare \
  --run-id "$NEW_RUN_ID" --image "$APPROVED_IMAGE" \
  --result-root "$PRIVATE_RESULT_ROOT" \
  --queue project-5090-robot-lab-debug-bcloud-bj \
  --profile render --storage-mode job_local \
  --egress-proxy "$APPROVED_EGRESS_PROXY"
"$AIDI_PYTHON" -B -m cluster_eval.submit_preflight \
  --run-dir "runs/cluster_eval/$NEW_RUN_ID"
"$AIDI_PYTHON" -B -m cluster_eval.submit_preflight \
  --run-dir "runs/cluster_eval/$NEW_RUN_ID" --execute
```

不需要代理的集群省略该参数；不能在已提交的私有配置或 workspace 内补参数。修正版通过网络、依赖、渲染后，三卡 `native` profile 使用相同参数接入；单卡诊断不等于 Qwen 或原生任务成功。

15:18 实测原三作业监督 PID `2698388` 仍存活，继续等待廊坊；北京网络诊断和 AIDC 代理版的独立监督均已取回终态后正常退出，没有把监督退出视为目标完成。确认监督存活须检查 `/proc` 与新快照，而非只读 PID 记录。

## 查询与后续复用

```bash
"$AIDI_PYTHON" -B -m cluster_eval.status \
  --run-dir runs/cluster_eval/native_alias_20261030_141545
```

三个新作业共用监督进程，PID 和路径在各 run 的 `supervision-launch.json`。当前为 PID `2698388`，输出目录为 `runs/cluster_eval/supervision_alias_20261030_142427/`；检查 `/proc` 与最新平台状态后才能认定它仍在运行。

新服务器只需控制端提交 SDK 与私有凭据，模型、资产和仿真安装由获分配的集群作业完成。使用新的 run ID 准备 `--profile native --storage-mode job_local`，先 dry-run 再显式提交；不能复制他人的 venv，也不能修改已提交包。`native_20261030_140856` 是发现路径问题前的未提交旧包，已标记不用，不要再提交。
