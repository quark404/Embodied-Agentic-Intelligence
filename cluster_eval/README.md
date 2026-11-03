# AIDI 集群评测：隔离配置与分步接入

**2026-10-30 17:06 路线更新**：AIDC 无代理完整作业 `aidc01-205cc48a674d` 已通过三卡/四份评分源码检查、私有 CuRobo 安装、CUDA 编译运算、真实 bridge 握手和同作业三相机渲染，现进入 Qwen 推理依赖安装。共享资产与原 lock 并非完整一致；Qwen 完整下载、实际推理和原生成功尚未通过。细节见[无代理资源复用](../docs/集群无代理资源复用.md)，下方代理记录仅为历史排错。

**2026-10-30 15:19 状态更正**：北京和 AIDC 都已在实际 5090 作业中通过 `/job_data` 路径映射、GPU/存储检查与结果取回。剩余阻塞已定位到连接层：两地到开发机现有代理的 TCP 连接均超时；AIDC 直连内网 PyPI 正常，但公网 wheel 下载被重置。北京诊断成功只表示取到了这些证据，不是网络或 RoboDojo 成功；AIDC 代理版于 15:15:00 在安装前失败。暂不原样重提这两个队列，廊坊原作业仍排队并保留监督。详细证据、接入参数及下一轮启动条件见[出网修复](../docs/集群原生RoboDojo作业.md#集群出网修复2026-10-30)。

**2026-10-30 14:24 最新入口**：新增显式 `--profile native --storage-mode job_local`，3 GPU / low / 240 分钟，在同作业内执行自身预检、隔离安装、bridge、渲染、Qwen BF16 与一个真正 RoboDojo harness episode。已修复 AIDC 平台 `/job_data` 路径映射校验，新作业已接受但未验收。具体新作业 ID、文档和原生结果回传协议见[集群原生 RoboDojo 作业](../docs/集群原生RoboDojo作业.md)。下文旧运行记录仅作历史审计。

核验日期：2026-10-29。只读参考 `uflow/training_submit` 和 `juice` 的已安装 SDK；不修改训练代码、训练环境或现有全局登录配置。

**2026-10-30 修复更新**：北京诊断确认容器没有 `/jfs-public`。新增显式 `--storage-mode job_local`，使用本作业私有 `/job_data/eai_<run_id>/results`，通过平台日志取回带 run 身份与 SHA-256 校验的预检结果，不再要求容器挂载开发机 JFS。修正版已分别提交北京 `bcloud-bj-zone1-d8af41833f56` 与 AIDC `aidc01-a34d901df807`；12:50 回查均排队，远端往返尚未验证。旧 `jfs` 模式仍保留原检查，没有静默回退。详细证据见[适配运行记录](../docs/RoboDojo适配运行记录.md)。安装、模型和仿真仍只允许在获分配的集群容器内执行。

新机器从[跨服务器开发与部署](../docs/跨服务器开发与部署.md)开始；公共仓库不附带本机 `runs/`、账户凭据或环境。

**13:35 执行更新**：北京纯预检被本项目的一体安装渲染作业 `bcloud-bj-zone1-d44ef6f38ae0` 替换；旧北京作业已确认停止，AIDC 候选保留。`--profile render --storage-mode job_local` 在同一作业内依次完成自身预检、隔离安装和三视角 RGB 渲染，预算 1 GPU / low / 120 分钟。详细状态、替换审计及验收边界见[无 JFS 接入修复](../docs/集群无JFS接入修复.md)。这不是 Qwen 或 RoboDojo episode 已跑通的声明。

## 当前结论

- 指定队列 `project-5090-robot-lab-acloud-langfang` 可见，集群为 `acloud-langfang`，平台返回的 **project_id 是 `horizon-labs`**，不是根据队列名称猜出的 `robot-lab`。
- 另外已核验 `project-5090-robot-lab-debug-bcloud-bj` → `bcloud-bj-zone1`、`project-5090-horizon-labs-aidc` → `hcloud`，project_id 均为 `horizon-labs`。查询和准备入口支持 `--queue`，仅允许这三个队列，默认仍为廊坊队列。第三个队列实际 job ID 前缀是 `aidc01`，不能根据前缀猜查询 cluster。
- 用本项目私有 token 已成功查询队列；可见、可调度和实时空闲数不保证具有最终提交权限，也不保证立即获得资源。
- 已有 SDK 可以提交任意 Python 入口，因此可提交评测。平台可能仍把有 GPU 的作业归类为 `train`；实际是否训练取决于入口，我们使用 `python3`，不使用训练入口或 `torchrun`。
- 2026-10-29 16:45:46（北京时间）单卡预检已被平台接受，见文末运行记录。没有安装 Isaac、下载资产或修改 `juice`；提交成功和本地测试都不是集群仿真验收。

## 1. 登录与环境隔离

三个环境分开：提交 SDK、Qwen 推理、Isaac 仿真。AIDI 不要安装进 `qwen35-vllm`；Isaac 也不要装进 `juice`。

本机可先只读使用已有 SDK：

```bash
export AIDI_PYTHON="$HOME/miniconda3/envs/juice/bin/python"
```

这只是选择 Python 可执行文件，不执行 `conda activate`、不安装包；下面使用 `-B` 避免写入依赖目录的字节码。已核实版本：`robo_orchard_jobs==0.1.0.dev20261026204009`、`robo_orchard_core==0.7.0`、`aidisdk==0.20.23`，另见 `requirements/aidi-submit.txt`。

**本项目的凭据文件是 `$HOME/.config/embodied-agentic/aidi.json`**：目录权限 `0700`、文件权限 `0600`。它在仓库之外，不上传给集群，也不写入 `.bashrc`。本机已经配置，不要重复执行初始化。新服务器可在仓库根目录交互初始化：

```bash
python3 -B -m cluster_eval.credentials
```

提示符隐藏输入；禁止把 token 当命令行参数、写进 Git、提交 JSON、日志或镜像。初始化拒绝覆盖已有文件。轮换时先在平台撤销旧 token，再用受控方式更新此私有文件并保留 `0600`；不要在聊天里发送新的 token。

只读验证，输出路径必须是新路径：

```bash
"$AIDI_PYTHON" -B -m cluster_eval.query_queue \
  --output "runs/cluster_eval/queue-$(date +%Y%m%d-%H%M%S).json"
"$AIDI_PYTHON" -B -m cluster_eval.aidi_cli manager --help
```

`query_queue` 按进程 `AIDI_TOKEN` → 项目私有文件 → 原 SDK 配置的顺序选凭据；本机没有设置进程 token，已验证使用新私有文件。`aidi_cli` 明确使用项目私有文件，并仅在当前 Python 进程切换 SDK 配置路径：这是因为原 SDK 的全局配置可能优先于环境变量。不会覆盖 `$HOME/.aidisdk/config.yaml`。

`aidi_cli` 现在只允许查询，拒绝通用 submit、stop、delete 和 urgent；真正预检提交必须使用下文的冻结提交入口。不把管理命令默认开放给新服务器或实验脚本。

其他服务器若没有 SDK，应使用 **新的 Python 3.11 虚拟环境**和获授权的内部包源安装 `requirements/aidi-submit.txt`；不要克隆整个训练环境。当前对内部包源的探测返回 403/404，独立安装尚未验证；不能通过关闭 TLS 校验或修改共享 pip 配置绕过。现有 `juice` 的只读调用已验证可用，不等于独立环境已经建好。

## 2. 首个作业只做环境预检

模板：`configs/aidi_5090_preflight.example.json`。

| 项目 | 首轮设置 |
|---|---|
| 资源 | 1 节点 × 1 张 RTX 5090；8 CPU、CPU/内存比 8，最终以平台分配为准 |
| 预算 | 15 分钟，失败不自动重试 |
| 优先级 | 显式 `priority=1`，此版本 SDK 映射为 `low`；不使用 SDK 默认的 high、不调用 urgent |
| 入口 | `python3 cluster_eval/probe.py`，只读取机器与库的状态 |
| 执行开关 | `execute=false`、`EAI_CLUSTER_CONFIRMED=0`，默认不提交且禁止入口误运行 |
| 数据挂载 | 不挂训练 bucket；新 `job_local` 模式不要求远端 JFS |
| 上传范围 | 仅 `cluster_eval` 源码，不上传 home、token、权重、PDF 或整个仓库 |
| 结果 | 容器写私有结果，控制端通过日志校验取回；不称作 RoboDojo 成功率 |

`prepare.py` 只调用 SDK 的本地 `prepare_workspace()`，不调用提交、上传 API。它创建新 staging、新 JFS run 目录、开发机原始标记和随机作业密码；私有提交文件不进入 Git，不接受清空已有目录。只有你自己的结果根目录可以传入：

```bash
"$AIDI_PYTHON" -B -m cluster_eval.prepare \
  --run-id preflight_$(date +%Y%m%d_%H%M%S) \
  --image "$APPROVED_AIDI_IMAGE" \
  --storage-mode job_local \
  --result-root "$MY_JFS_ROOT/embodied-agentic-eval"
```

`MY_JFS_ROOT` 当前用于**控制端**保存小型回执和取回结果，应位于控制端真正挂载的 `/jfs-public` 下，不要求集群也有该挂载。`job_local` 模式把原始标记装入冻结代码包；容器先验证标记和自身上传包路径，再独占创建本 run 私有目录。原 `jfs` 模式只有显式选择或省略新参数时才使用，仍必须验证共享挂载。源码归档和结果目录独立，每个 run 都用新名字。

北京 debug 队列使用同一个模板及安全限制，不修改旧队列快照：

```bash
QUEUE=project-5090-robot-lab-debug-bcloud-bj
RUN_ID=preflight_debug_$(date +%Y%m%d_%H%M%S)
"$AIDI_PYTHON" -B -m cluster_eval.query_queue --queue "$QUEUE" \
  --output "runs/cluster_eval/queue-${RUN_ID}.json"
"$AIDI_PYTHON" -B -m cluster_eval.prepare --queue "$QUEUE" \
  --storage-mode job_local \
  --run-id "$RUN_ID" --image "$APPROVED_AIDI_IMAGE" \
  --result-root "$MY_JFS_ROOT/embodied-agentic-eval"
RUN_DIR="runs/cluster_eval/$RUN_ID"
"$AIDI_PYTHON" -B -m cluster_eval.submit_preflight --run-dir "$RUN_DIR"
```

这组命令只查询和本地准备，不提交。确认后再使用下文 `--execute`。队列及集群写入 manifest 和回执；提交使用冻结配置，不在提交时换队列。防重复检查限定为当前用户、当前队列，因此用户明确授权的两个队列各一次预检可以并存，但同一队列不重复创建本项目在途作业。跨队列并行仍须事先授权，不能为了抢调度自动批量提交。

**真正提交前必须确认资源授权、镜像、私有配置及结果路径。不要编辑冻结的私有快照，也不要使用通用 SDK submit 命令提交本预检。** 该 SDK 的 `new_job()` 默认把账户 token 注入作业环境；本项目入口在提交前重新设置严格的作业环境白名单，不修改 SDK 安装文件。

```bash
RUN_DIR=runs/cluster_eval/<新准备的run_id>
"$AIDI_PYTHON" -B -m cluster_eval.submit_preflight --run-dir "$RUN_DIR"
# 仅在确认申请这一个单卡作业后执行；会上传并申请资源。
"$AIDI_PYTHON" -B -m cluster_eval.submit_preflight --run-dir "$RUN_DIR" --execute
"$AIDI_PYTHON" -B -m cluster_eval.status --run-dir "$RUN_DIR"
```

第一条只核验快照，第三条只回查自己回执对应的 job，不返回可能含密码的原始平台 JSON。显式 `--execute` 才能提交，并只在内存中启用 `EAI_CLUSTER_CONFIRMED=1`；模板和私有预览配置始终保持 `execute=false`。检查包含上传包摘要、配置摘要、原始 JFS 标记、环境白名单、版本、同名/本项目在途作业及单次提交锁。提交报错后保留 attempt 文件，不自动重试；先回查平台，不能通过删锁重试制造重复作业。常规状态检查请间隔数分钟，不高频轮询。

## 3. 从预检到原生评测的门槛

1. **队列与存储**：实际分配卡型号、可见设备数、驱动、Python、私有目录和输出取回。`job_local` 额外记录可用磁盘和公共依赖站点 HEAD 连通性；不下载权重、资产或安装依赖。`probe.py` 不做 CUDA 分配或图像渲染。
2. **Isaac RGB 渲染**：在批准的镜像中，用同一张分配到的 5090 做单场景 headless 相机检查。CUDA 可用不等于 Vulkan/RTX 可用；训练镜像不是 Isaac 镜像。GPU 驱动由集群管理，不能在作业内升级。仅在 Python 启动后设置 `NVIDIA_DRIVER_CAPABILITIES` 不能补回容器启动时未注入的图形驱动库，需要核实平台容器运行配置。
3. **固定上游环境**：按 `configs/robodojo_versions.json` 固定 RoboDojo、XPolicyLab、IsaacLab、CuRobo 与资产 revision。Isaac 目标为 Python 3.11 / Isaac Sim 5.1.0，仍待远端验证。上游安装脚本的 WebSocket 12 与固定 XPolicyLab 的 ≥14 要求有冲突；不要直接运行全量安装脚本、更新子模块到 main 或改共享环境。
4. **Qwen 连通**：优先保留 H20 上已运行的 Qwen，集群只跑仿真。需验证集群到推理主机的网络与认证；作业内 `127.0.0.1` 不是开发机。授权的 SSH 隧道可以转发 loopback API，但不能假定集群能 SSH 到开发机，也不能上传个人私钥或把无认证模型接口暴露到公网。
5. **单 episode**：先 `q_only`、`general_pickup`、1 episode，验证三相机、动作守卫、原生终止、轨迹与 `_result.json`；以后才扩大条件矩阵。闭源端点仍未配置，不运行相关条件。

跨机器网络不通时，可以另评估同节点多卡部署：27B BF16 权重约 51.75 GiB，不应计划放进单张 32 GiB 的 5090。候选是两卡推理加独立仿真卡，但显存、TP、视觉编码及实际可分配拓扑都要再测，不能默认申请整机八卡，更不能悄悄换量化模型改变研究基线。

资产仅需固定版本的 `Assets/**`，已查询大小约 41.27 GB，不是整个多 TB 数据集。首次资产准备与镜像构建应单独安排；之后共享只读版本，避免每个 rollout 或每个评测作业重新安装和下载。

## 4. 验收和保护边界

- `inventory_passed` 只代表预检；`job_local` 的通过还要求图形库可加载，控制端监督必须验证日志结果的 run、原始标记摘要和内容摘要，才接受为已取回的预检。`native_evaluation_ready` 始终保持 false，直到另行完成真正仿真验收。
- 提交返回 job ID 只代表平台接受。结果必须同时保留代码/配置摘要、上游版本、模型 revision、任务 seed、原生结果、失败原因与时长。
- 不使用平台全局停止/删除命令；只在明确授权后管理本项目记录的 job ID。遇到排队不重复提交。
- SDK 快照可能含随机作业密码，只放私有 staging；token 文件绝不进入 `to_upload`。提交故障日志分享前先脱敏。
- 清理只针对本项目创建的具体 run 目录，不清理 JFS 公共数据、训练输出、`juice` 或 Hugging Face 共用缓存。

当前只读凭据验证记录：`runs/cluster_eval/queue-private-profile-20261029.json`；本地安全测试入口：`python3 -B -m unittest discover -s tests -p test_cluster_eval.py -v`。

## 5. 2026-10-29 实际运行记录

- run：`preflight_20261029_164441`；job ID：`acloud-langfang-669fdf14eb85`。
- 平台接受：**2026-10-29 16:45:46（北京时间）**；17:21:35 回查仍为 `Queuing`、`start_time=null`，尚无 `inventory.json`。这是当时快照，最新状态以 `cluster_eval.status` 为准。
- 确认低优先级、1 GPU、15 分钟上限。申请字段为 8 CPU、内存比 8；平台归一化回执为 `cpu=18, memory=174`，不能假定平台完全采用请求规格，也不能据此认为容器已启动。
- 使用训练案例中已有的 Ubuntu 22.04 / CUDA 12.8 / Torch 2.8 镜像作为**只读机器预检镜像**，不是已经验证的 Isaac 镜像。具体镜像字符串保留在私有快照，不修改原案例。
- 上传工作区约 61 KB，只含预检源码和 SDK 启动辅助文件，无 bucket 挂载、模型权重或登录 token。新作业返回的配置中也没有发现提交使用的 token。
- 私有提交快照、冻结包、摘要、attempt、平台回执和状态位于 `runs/cluster_eval/preflight_20261029_164441/`。每个已提交包独立冻结；后续编辑工作区文档或代码不会改变该包。
- 结果目录为本用户 JFS 命名空间下的 `embodied-agentic-eval/preflight_20261029_164441/`。原始标记已写入；远端必须实际读到同一标记，才会生成 `inventory.json`。目前没有原生 RoboDojo 成功率。
- 50 项本地测试通过；原 `uflow/training_submit` 的 9 个文件和共享 SDK 配置的摘要未变化，Qwen 健康端点仍返回 200。没有停止其他作业，也没有对 `juice` 执行 pip/conda 安装。
- 排队期间只读检查 CPU 通信依赖：`juice` 缺少 `msgpack-numpy` 和 OpenCV，不能把已通过的 SDK 查询当作 XPolicyLab WebSocket 联调通过。未往该环境补包、未启动桥接服务；下一步应在独立桥接环境安装并核验后再做 reset 测试。

## 6. 北京 debug 队列追加预检（2026-10-29）

- 用户明确要求在北京队列也运行一次；保留廊坊作业，不停止或提优先级。
- run：`preflight_debug_20261029_184948`；job ID：`bcloud-bj-zone1-de2bee22d496`。
- 平台接受：**2026-10-29 18:50:10（北京时间）**；18:50:23 回查为 `Queuing`，无开始时间、无 `inventory.json`。同期廊坊作业也仍在排队。空闲卡数量不等于立即分配；本记录不是测试通过声明。
- **18:52:44 再次回查仍为 `Queuing`**，尚无 `inventory.json`；没有重复提交或调整优先级。
- 请求 1 GPU、8 CPU、内存比 8、15 分钟、low 优先级、零自动重试。平台归一化规格为 `gpu=1, cpu=17, memory=102`，并非容器已经启动。
- 沿用原预检镜像，只读取机器、驱动库与包元数据、验证 JFS 标记。不训练、不安装依赖、不下载权重/资产、不运行 Isaac 或完整 Qwen 评测。
- 独立冻结包、私有配置、提交锁、回执及状态快照均在 `runs/cluster_eval/preflight_debug_20261029_184948/`；JFS 结果在本用户 `embodied-agentic-eval/preflight_debug_20261029_184948/`，不覆盖旧 run 或 `selected-preflight.json`。
- 新增队列白名单、真实集群映射及资源限制回归测试；本轮全部 **62 项 CPU 测试通过**。提交包经检查不含登录 token，SDK 自动注入凭据的防护仍保留。
- 提交后核验原训练目录 9 个文件及共享 SDK 配置，共 10 个文件摘要未变化；平台本作业元数据未发现提交 token，已有 Qwen 健康端点返回 HTTP 200。未安装或修改任何已有环境。

只读回查这次预检：

```bash
"$AIDI_PYTHON" -B -m cluster_eval.status \
  --run-dir runs/cluster_eval/preflight_debug_20261029_184948
```
