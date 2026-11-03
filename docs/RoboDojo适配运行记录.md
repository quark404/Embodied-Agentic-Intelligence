# RoboDojo + Qwen Harness：适配与监督

开始日期：2026-10-29。**目标仍未完成**；队列接受、依赖安装、CPU 单测和接口自检都不能替代真实 RoboDojo 验收。

**2026-11-03 镜像验证更新**：三个只读清单作业已返回。`starvla-robodojo-client:v4.6` 在 AIDC 上确认含 IsaacSim 5.1 / IsaacLab 0.54.3 / torch 2.7.0+cu128，适合先作为仿真基线；`qwen35:m_n` 在北京 debug 上确认含 `/opt/eai/qwen` 的 vLLM 0.17.1 / torch 2.10 / transformers 4.57.6，但没有仿真栈；`yaoting.huang-test:robodojo-5090-20261030` 因旧版清单未探 `/opt/eai/*`，结论待重探。详见[镜像验证_20261103](镜像验证_20261103.md)。

**2026-10-30 14:24 最新执行记录**：已根据 AIDC 真实日志修复平台 `/job_data` → `/running_root` 的路径校验问题，提交包含完整 Qwen + 原生 episode 的 AIDC 三卡作业 `aidc01-dd6fc78aa006`；北京更新为 `bcloud-bj-zone1-b9422ca13f3e`，廊坊更新为 `acloud-langfang-6bea5a5ecc93`。当前仍排队，目标未验收。新监督 PID `2698388`；详细证据、资源预算和成功判据见[集群原生 RoboDojo 作业](集群原生RoboDojo作业.md)。本轮 139 项 CPU 测试通过，不能替代集群实测。

**2026-10-30 13:44 更新**：已提交北京一体作业 `bcloud-bj-zone1-d44ef6f38ae0`，把自身预检、隔离安装及真实 RGB 渲染合并到一次排队中；原北京纯预检仅在确认属于当前用户且尚未开始后停止，AIDC 和廊坊候选保留。当前三条在途候选都仍为 `Queuing`，没有伪称远端安装或原生成功。JFS 根因修复、替换审计和结果校验协议见[无 JFS 接入修复](集群无JFS接入修复.md)。

排队期间又实际修复了两个后续必经入口：显式仿真 venv 不再被上游 Conda 包装层阻断；Qwen 启动器支持两张已分配 GPU 的 BF16 张量并行，并拒绝单张 5090 的不足显存配置、重复 GPU、越界选卡或占用中的 GPU。固定 CuRobo 的 CUDA extra 与构建依赖已进入提交的渲染包；venv/Qwen 启动器后续改动没有篡改冻结包，供后续原生作业使用。

本轮使用既有 `juice` Python 只读运行 **124 项 CPU 测试通过**，两份 Shell 入口语法检查和 `git diff --check` 通过。没有在开发机安装包、启动 GPU 模型或运行仿真；SDK 全局配置 SHA-256 与先前基线一致。系统 `/usr/bin/python3` 全套测试因缺少 NumPy 无法完成，未通过向系统环境安装依赖来绕过；专门的集群标准库测试可独立通过。这些都不是 GPU 验收。

**当前执行边界（2026-10-30）**：按用户要求，模型、仿真、安装和资产下载只在获分配的 5090 集群容器内继续。本机仅进行提交代码维护、轻量回归测试、状态/日志查询与私有回执记录；不恢复本机安装或 Qwen 服务，不修改 `juice`、训练目录或他人环境。

## 1. 完成标准

1. 三个已授权队列各一次单卡预检，记录最先结束、最先通过预检的作业，以及各自排队/执行耗时；不重复提交抢资源。
2. 在真实分配到的 RTX GPU 上启动固定版本 RoboDojo/Isaac，得到非空、正确颜色顺序的三相机图像。
3. Qwen3.5-27B 通过本仓库 planner/actor、工具守卫、记忆与执行反馈做决策；不能用固定动作脚本、假模型或合成环境代替。
4. 原生 `general_pickup` 至少完成一个真实 episode，保留原生 `_result.json`、harness 轨迹、模型调用记录和图像。先证明闭环正常，再核对原生任务成功判定；失败不能改成成功，奖励/判定函数不改。
5. 环境与启动步骤可复现，所有未通过项明确列出。先不训练、不做 SFT，不改共享环境或系统驱动，不使用 sudo。

## 2. 三队列监督

| 队列 | 平台 cluster | job ID | 接受时间（北京时间） |
|---|---|---|---|
| `project-5090-robot-lab-acloud-langfang` | `acloud-langfang` | `acloud-langfang-669fdf14eb85` | 2026-10-29 16:45:46 |
| `project-5090-robot-lab-debug-bcloud-bj` | `bcloud-bj-zone1` | `bcloud-bj-zone1-de2bee22d496` | 2026-10-29 18:50:10 |
| `project-5090-horizon-labs-aidc` | `hcloud` | `aidc01-bf51709b5980` | 2026-10-29 19:10:48 |

均为 1 GPU、low、15 分钟上限、零自动重试，只执行环境预检。19:26 的监督快照中三个作业均为 `Queuing`，未产生 `inventory.json`。这只是时间点记录，最新状态以平台及监督输出为准。

`cluster_eval.supervise` 每 120 秒查询显式回执对应的作业，最多 720 轮；只读，不提交、不停止作业、不提优先级。初始私有记录目录为 `runs/cluster_eval/supervision_20261029_1914/`：

- `monitor.json`：实际 PID、受监督的 run、启动时间和预算。
- `poll-*.json`：每次回查的白名单状态、预检结果和耗时。
- `first_terminal.json`：首个结束的作业，可能是失败，不能视为成功。
- `first_verified_inventory.json`：首个同时具有成功平台状态及真实单卡/JFS 预检结果的作业。
- `monitor-ended.json`：监督结束原因。监督达到观察上限不等于作业失败，也不允许因此重提。

提交时间不同，因此“最先结束”不等于严格受控的“最快队列”。比较时同时看接受到结束、排队时间、容器执行时间；原生适配尚需后续独立验收。

### 2026-10-30 平台回查与失败诊断

10:17（北京时间）直接查询平台和本项目日志，得到以下结果：

| 队列 | 状态 | 开始时间（北京时间） | 结束时间（北京时间） | 原生结果 |
|---|---|---|---|---|
| 廊坊 | `Queuing` | 未开始 | 未结束 | 无 |
| 北京 debug | `Failed` | 2026-10-30 00:27:00 | 2026-10-30 00:27:10 | 无 |
| AIDC | `Failed` | 2026-10-30 05:50:47 | 2026-10-30 05:50:56 | 无 |

- **北京 debug 最先结束，但没有通过预检**。两次失败都发生于 `check_context()`：`Real JFS output is required; no local fallback`，均无 `inventory.json`，更没有运行 Isaac/Qwen。北京排队约 5 小时 37 分钟、执行 10 秒；AIDC 排队约 10 小时 40 分钟、执行 9 秒。
- 原监督进程及工具句柄已不存在，旧记录停在 00:18，不能声称整夜连续监督。确认无存活副本后，10:18 启动新的原三任务只读监督，记录目录为 `runs/cluster_eval/supervision_20261030_101823/`；失败时间依据平台记录，而不是轮询发现时间。
- 日志尚不能区分 `/jfs-public` 不存在、未挂载，或容器只挂载其子目录而根目录 `ismount` 为假。平台没有提供可用的已结束 Pod 挂载明细；不能据此断言 GPU、驱动或仿真依赖有问题。
- `cluster_eval/probe.py` 新增失败时的只读标准输出诊断：JFS/结果目录存在性、原始标记摘要/身份匹配、相关挂载点与文件系统类型、GPU 型号及图形库可加载性。不输出挂载源、挂载参数、原始标记内容或环境变量；不改变 JFS/原始标记验收门槛，不写本地替代结果，不将失败改成成功。
- 本次 26 项集群相关回归测试通过，覆盖诊断不得在未确认的开发机执行、不得回退写盘、诊断异常不覆盖原始错误等。这些测试使用 mock，不代表集群挂载成功。

针对已定位的缺失诊断信息，单独提交一个新冻结快照，而非重试或篡改旧作业：

- run：`preflight_jfs_debug_20261030_102335`；job：`bcloud-bj-zone1-5644a376fd60`。
- 北京 debug 队列于 **2026-10-30 10:24:07** 接受，10:24:45 回查 `Queuing`。仍为 1 GPU、low、15 分钟、零自动重试，无 bucket 挂载、安装、模型加载或资产下载。
- 原因、冻结包和回执记录在该 run 的 `diagnostic-purpose.json`、`manifest.json`、`receipt.json`；未覆盖原三个回执或 `selected-preflight.json`。后续以新诊断的真实输出决定如何处理存储，不预先放宽检查。
- 诊断单独由独立进程会话监督，每 120 秒回查、最多 720 轮，记录在 `runs/cluster_eval/supervision_jfs_debug_20261030_102631/`。它只记录状态，不安装软件、不自动重提或启动评测；不能将存在 `monitor.json` 当作进程仍存活，恢复时仍须实际核对 PID/平台状态。

**11:38（北京时间）恢复监督后的阻塞复核：**连续三轮均实际查询平台，并通过进程身份和 PID 句柄等待确认监督存活。廊坊原预检与北京新诊断仍为 `Queuing`，`start_time` 为空、没有 `inventory.json`；11:25 回查的北京与 AIDC 原预检仍为 `Failed`。当前缺少已分配的集群运行环境及新的容器诊断，尚不能推进远端安装或原生闭环验收。

原三作业监督 PID `2156036`、新诊断监督 PID `2169028` 在 11:38 均确认存活；最新记录分别为 `poll-0039.json` 与 `poll-0035.json`，位于上述两个监督目录。目标因外部调度等待而受阻，不表示完成，也不停止这两个只读监督进程。没有重复提交、改变优先级、扩卡、放宽存储校验或恢复本机安装。下次继续时先查询同一 job ID 的实际状态；获得资源或新诊断后再确定操作，不能仅因等待超时就重启作业。

### 2026-10-30 12:06：北京诊断已运行，确认存储缺失

直接回查平台并读取新诊断作业的脱敏日志，确认 `bcloud-bj-zone1-5644a376fd60` 于 **11:39:36 开始、11:39:46 失败**，排队约 1 小时 15 分钟。不是持续排队，也不是 RoboDojo 或 Qwen 启动后的失败：

- 容器内 `/jfs-public` 目录不存在，结果目录和原始标记也不存在；相关挂载诊断仅见根文件系统 `overlay`。因此这一次不是“只挂子目录导致根目录 ismount 为假”，而是本次作业没有提供所需的 JFS 路径。其他队列的挂载情况仍须各自验证。
- `nvidia-smi` 成功识别一张 `NVIDIA GeForce RTX 5090`，驱动 `570.169`、显存 `32607 MiB`。`libvulkan.so.1`、`libGLX_nvidia.so.0`、`libEGL.so.1` 均可加载；这只证明 GPU 查询及库加载，尚未证明 Isaac 渲染或仿真兼容。
- 仍无 `inventory.json` 或原生 episode。经审阅的诊断保存在私有 run 目录的 `context-diagnostic-reviewed.json`，不含账户 token 或原始平台环境。
- 12:05 回查廊坊原预检仍为 `Queuing`。北京诊断的监督已按 `monitor-ended.json` 中的 `reason=all_terminal` 正常结束，未重启；原三作业监督 PID `2156036` 仍确认存活并继续等待廊坊。

该时间点尚未提交新的安装/评测作业，未安装主机驱动、手工挂载或使用 sudo。后续已实施的修复如下，不再等待容器自动出现开发机 JFS。

### 2026-10-30 12:50：存储接入已修改并重新提交验证

新增显式 `job_local` 模式，将工作目录置于本作业私有 `/job_data/eai_<run_id>`，以冻结原始标记约束身份，通过平台日志输出带 SHA-256 的小型预检 JSON，由控制端验证取回。旧 JFS 模式不变，新模式不是静默回退，也不冒充共享挂载已通过。资产下载器中的相同 JFS 限制已同步处理，独立集群安装入口也已准备。实现与验收边界见[集群无 JFS 接入修复](集群无JFS接入修复.md)。

- 北京修正版：run `preflight_local_20261030_122514`，job `bcloud-bj-zone1-d8af41833f56`，12:26:05 接受。
- AIDC 修正版：run `preflight_local_aidc_20261030_123228`，job `aidc01-a34d901df807`，12:37:17 接受。
- 两者均为 1 GPU、low、15 分钟、零自动重试，没有上传账户 token、安装软件或运行模型；12:50 回查均排队，尚不能声称远端存储往返已验证。
- 新监督 PID 分别为 `2416081`、`2483830`，对应 `supervision_local_20261030_122740/` 与 `supervision_local_aidc_20261030_125040/`；每 120 秒查询，只取回已校验的预检 JSON，不自动安装。
- 本次 42 项集群相关 CPU 测试通过，包括隔离路径、越界拒绝、结果篡改/冲突、结果取回、监督不误报和开发机禁止安装。冻结预检包不包含后续才新增的安装入口；安装入口尚未执行，依赖兼容和真实渲染仍待集群验证。

另一台机器启动新的只读监督实例时，先确认没有仍在运行的同一监督，使用新的输出目录：

```bash
"$AIDI_PYTHON" -B -m cluster_eval.supervise \
  --run-dir runs/cluster_eval/preflight_20261029_164441 \
  --run-dir runs/cluster_eval/preflight_debug_20261029_184948 \
  --run-dir runs/cluster_eval/preflight_aidc_20261029_191019 \
  --output-dir runs/cluster_eval/<新的监督目录> --interval 120 --max-polls 720
```

## 3. 隔离环境布局

历史准备根目录：`$MY_JFS_ROOT/embodied-agentic-runtime/robodojo_20261029_v1`。仅含本项目私有内容，根权限 `0700`。**用户要求只在 5090 上继续后，本机安装和资产下载已停止；以下是遗留状态，不是仍在后台安装，也不是可用的仿真环境。** 不复用或清空其他环境，不把资产下载进 home。

| 路径 | 作用与状态 |
|---|---|
| `bootstrap/` | 本次新建工具环境，已安装 `uv==0.9.3`；不是原 `juice` |
| `python/` | 已下载独立 CPython 3.11.14，使用 `uv python install --no-bin`，不改变用户默认 Python |
| `bridge/` | 已有 CPU 握手/reset 记录；独立 WebSocket 服务端依赖见 `requirements/robodojo-bridge.txt`，不等于仿真验收 |
| `sim/` | 不完整，安装已停止；Torch/Isaac 的集群安装与运行未验证，候选依赖见 `requirements/robodojo-sim.txt` |
| `RoboDojo/` | 按 `configs/robodojo_versions.json` 获取四个精确 revision，包括 IsaacLab/CuRobo；不执行上游一键安装 |
| `project/` | 本仓库运行模块与 policy 的小型副本，policy 链接不再指向开发机 home |
| `dataset/` | 仅部分资产，下载已停止、没有完整验收；固定 revision 的 `Assets/**` 预期 15,365 文件、41,269,513,111 字节 |
| `cache/`、`tmp/`、`logs/` | 本次独立缓存、临时文件、私有安装日志 |

此前的准备只针对本项目显式新环境；未运行上游含 sudo、全局 Conda 修改、`git submodule --remote` 的脚本。之后仅在集群容器内继续，不因为存在半成品而自动恢复本机进程。

### 依赖冲突处理顺序

- 上游安装脚本固定 WebSocket 12，而 XPolicyLab 服务端需要 ≥14：拆成两个进程/环境。10 月 29 日的真实 CPU 记录证明 WS14.2→14.2 和 WS12.0→14.2 握手/reset 通过，分别见 `runs/xpolicylab-bridge-20261029-ws14/result.json` 和 `runs/xpolicylab-bridge-20261029-ws12-to14/result.json`；未使用 GPU，不证明完整仿真协议或 episode 成功。
- 固定 IsaacLab 源码的 Starlette 要求与 RoboDojo 安装脚本中的固定版本不同。先检查真实 resolver/运行结果，不绕过错误宣称环境完整，不随意改 benchmark gitlink。
- 显卡驱动/图形库是否由平台容器运行时注入，必须等远端预检。应用启动后设置环境变量不能补回缺失驱动挂载；绝不在共享主机安装驱动。

## 4. 跨机器 Qwen 连接

新增候选方案 `scripts/qwen_jfs_relay.py`：集群端仅监听 loopback 的 HTTP 服务，将有界多模态请求写入私有 JFS；推理主机 worker 读取并转发给固定的本机 Qwen `/v1/chat/completions`，响应原样返回。无需公网监听、SSH 私钥或 AIDI token 转发。

- 每个 session 采用 `0700` 目录、`0600` 文件、独立临时 bearer token；它不是账户登录 token。
- 只允许指定 Qwen、非流式 chat completions、最多 8192 输出 token、内嵌 PNG/JPEG；禁止让推理机代取外部或本地文件 URL。
- 请求数量与存活时间有上限，claim 防止重复推理；超时/故障不自动重试。不自动申请 GPU或重启模型。
- JFS 转发会增加文件系统往返耗时，必须单独测量；不能把它称为已实现的低成本 RL 优化。

2026-10-29 19:20 本地转发实测未通过：Qwen 上游端口已不可达，返回 `URLError`。日志表明旧 Qwen 引擎于 **19:10:38** 意外退出，具体根因尚未确定；当时回查 tmux/监听端口不存在，GPU 6 已有其他用户进程。未停止该进程，也未自动抢占其他 GPU。**用户后续要求只在 5090 集群继续，因此不再采用“恢复本机 H20”的路线。** JFS relay 仍只是未通过真实模型验证的候选；存储不可达时同样不可用。27B BF16 不能直接装入单张 32 GiB 5090，后续需明确获分配的多卡/服务分离方案，不默默切换小模型或量化基线。

## 5. 后续执行顺序

1. 持续监督廊坊原任务与北京新诊断，区分结束、预检通过与原生任务成功；已失败作业仅查询历史日志，不自动重试。
2. 根据容器诊断确认共享存储可达性。若确实缺失，先解决获授权的集群挂载；不手工挂主机目录、不使用 sudo、不挂训练 bucket、不伪造同路径本地目录。如果只是挂载点判定不匹配，取得实际挂载与原始标记证据后再修正验证器并回归测试。
3. 存储与图形条件满足后，**在 5090 容器内**完成独立依赖/资产准备，记录精确版本并验证 Isaac 三相机图像；开发机不运行安装或仿真。
4. 为 Qwen3.5-27B BF16 明确合规的 5090 多卡/服务分离资源与限时预算，再运行真实模型＋harness＋RoboDojo episode。不得抢占他人 GPU、暗改精度或复用/篡改旧预检包。
5. 用真实轨迹逐项修复适配问题，直到本节完成标准均有证据。当前不具备标记目标完成的依据。
