# 集群无 JFS 接入修复

**2026-10-30 14:24 更新**：AIDC 修正版预检的真实失败已定位并修复：平台把 `/job_data` 路径解析到 `/running_root` 的 JuiceFS 挂载，旧的逐字路径校验误拒绝。现仅放行这个已挂载的平台根映射，run 内软链接和跨 run 写入仍拒绝。新 AIDC 完整原生作业 `aidc01-dd6fc78aa006`、北京渲染 `bcloud-bj-zone1-b9422ca13f3e`、廊坊预检 `acloud-langfang-6bea5a5ecc93` 已接受，14:24 均排队。完整流程与替换记录见[集群原生 RoboDojo 作业](集群原生RoboDojo作业.md)。以下保留早期修复过程，不应再提交旧包。

## 修复原因与边界

2026-10-30 的北京容器日志证明 `/jfs-public` 不存在，同时能够识别 RTX 5090。此前把开发机共享路径当成集群必备条件是不适用的设计，而不是应该反复等待的条件。本次修改存储接入方式，不在容器里伪造同名 JFS，不手工挂载，不运行 sudo，不改变训练目录或共享 Python 环境。

## 已实现的链路

1. `cluster_eval.prepare --storage-mode job_local` 冻结源码、配置和本 run 原始标记。开发机的 JFS 只保存小型控制记录；模型和资产不在开发机下载。
2. 容器检查 `EAI_CLUSTER_CONFIRMED`、`WORKING_PATH`、上传包位置及标记摘要；只创建 `/job_data/eai_<run_id>/results`，拒绝目录复用、软链接或跨 run 路径。
3. `probe` 记录 GPU、库、磁盘和公共依赖站点连通性，将预检 JSON 写在本作业内，并输出有大小上限的 `EAI_INVENTORY_V1` 记录。
4. `cluster_eval.status` 使用控制端凭据查询**回执对应的作业**日志，验证 run、原始标记 SHA-256、JSON SHA-256，再原子保存 `inventory-export.json` 与 `inventory.json`。错误、缺失或冲突记录不通过，不能覆盖不同的既有文件。
5. 监督只有在平台成功结束、预检通过且取回校验通过时才接受该预检。容器没有收到登录 token；不通过把 SDK 凭据传进容器来上传结果。

旧 `jfs` 模式继续严格要求共享挂载。模式由提交时显式选择，不能运行失败后偷偷切换。日志传输现覆盖预检 JSON 和有大小上限的诊断渲染图片，**不替代原生 episode 的完整轨迹和图像归档**，也不证明 Qwen 或原生任务已跑通。

## 后续安装已准备，不在开发机执行

`cluster_eval.runtime_install` 默认只输出安装计划，只有显式 `--execute`、合法作业身份及该作业自身已通过的预检才允许执行。它拒绝复用既有 runtime，限制总时长、检查磁盘余量，将 HOME、缓存、Python、Torch 编译产物都重定向到该作业私有目录。

- 独立 bootstrap：固定 `uv==0.9.3` 和受管理的 Python `3.11.14`。
- 仿真与 WebSocket bridge 分为两个虚拟环境；仿真固定 Torch `2.7.0`/cu128、Isaac Sim `5.1.0`，bridge 不被仿真的 WebSocket 12 约束污染。
- 使用已固定 revision 的源码准备器，不执行上游 `install.sh`、`conda init`、`git submodule --remote` 或系统包安装。
- 约束文件按固定 IsaacLab 源码采用 `starlette==0.49.1`，不再照搬上游脚本冲突的 `0.45.3`；实际依赖解析及 GPU 渲染仍须在集群验证，遇到冲突会失败退出，不使用忽略依赖或静默改版本。
- 资产下载器新增显式 `--storage-mode job_local`，只允许在已核验的同一作业目录写入；通过 `python -m scripts.download_robodojo_assets` 调用。仍使用固定资产清单并逐文件检查，禁止开发机误运行该模式。

安装器已接入独立的 `render` 作业档位；2026-10-30 13:35 提交了第一份一体作业，见下节。不能把安装计划、提交回执或 CPU 回归测试当成环境安装成功。

## 本次提交

| 队列 | 新作业 | 接受时间（北京时间） |
| --- | --- | --- |
| 北京 debug | `bcloud-bj-zone1-d8af41833f56` | 2026-10-30 12:26:05 |
| AIDC | `aidc01-a34d901df807` | 2026-10-30 12:37:17 |

12:50 平台回查均为 `Queuing`，未产生可取回的预检；后续状态以平台及 `runs/cluster_eval` 私有记录为准。这两份包在提交前已冻结，之后的安装器修改不会改变在途作业。

## 13:35：改为一次排队完成预检、安装和渲染

不再强制采用“预检结束后重新排队安装，安装后重新排队渲染”的串行方式。新 `render` 档位在**同一个作业**内先验证自己的原始标记、GPU 和存储，通过后才安装私有环境并启动真实 Isaac RGB 渲染；预检不通过则立即退出。

- 新 run：`render_20261030_1340`；job：`bcloud-bj-zone1-d44ef6f38ae0`。平台接受时间为 **2026-10-30 13:35:00（北京时间）**，run 名只是标识，不替代平台时间。
- 1 GPU、low、最长 120 分钟、零自动重试；安装最多 90 分钟，渲染最多 15 分钟。禁止安装系统包或修改驱动。
- 原北京修正版 `bcloud-bj-zone1-d8af41833f56` 在替换前核验了当前用户、回执、队列、未开始及 `Queuing` 状态，13:34:34 平台确认 `Stopped` 后才提交新作业；不同时堆叠同队列的本项目任务。替换是执行范围改变，不是超时重提；原包、日志、回执完整保留。
- AIDC 的 `aidc01-a34d901df807` 未停止，继续作为独立预检候选。旧廊坊作业也未改动。
- 固定 CuRobo 的 `pyproject.toml` 把 CUDA 后端放在可选依赖中。安装入口现显式选择 `[cu12]`、约束 `cuda-core==0.7.0`，预先安装 `setuptools-scm==8.3.1` 满足无构建隔离要求；随后实际导入 `cuda.core` 并执行小型 Torch CUDA 运算。这些检查在集群作业内运行，尚未声称成功。
- 三张 96×96 PNG 通过 `EAI_RENDER_V1` 分块记录和最后的 commit 返回；校验 run、原始标记、整包 SHA-256、PNG CRC、RGB 格式、尺寸和视角差异。缺块或冲突拒绝通过。
- `scene=procedural_rgb_cubes` 仅用于诊断图形链路，**不是 RoboDojo 场景、机器人动作或基准成功率**。只有平台结束成功且图片校验通过才算渲染验收。

新监督 PID 与日志位置记录在该 run 的 `supervision-launch.json`；监督只查询和取回，不自动提交或停止。13:37:57 回查新作业为 `Queuing`，尚无 GPU 渲染证据。

```bash
"$AIDI_PYTHON" -B -m cluster_eval.status \
  --run-dir runs/cluster_eval/render_20261030_1340
```

`render_20261030_131807` 是更早的未提交冻结包，未包含此次 CuRobo 修正，不要提交它，也不要原地修改任一冻结包。
