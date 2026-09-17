# CastMind 复现事实库

> 用途：组会汇报的**本机复现**唯一依据。只记录工作区代码、配置、日志、产物里能核对的内容。  
> 对照论文事实见 `analysis/paper_facts.md`。  
> **本阶段不生成 PPT。**

## 0. 口径（先读）

### 0.1 一句话

本机工作是 **代码/流程级复现验证**，**不是**论文官方结果的严格数值复现，也**不宣称**「完整复现论文」或「超过论文 AlphaCast」。

### 0.2 为什么不是 Table 1 数值复现

| # | 事实 | 证据 |
|---|------|------|
| 1 | 论文主 LLM 为 GPT-5 | 论文 §4.1 |
| 2 | 本机 LLM 为 DeepSeek `deepseek-chat` | `.env` / `.env_template`：`MODEL=deepseek-chat`，`OPENAI_BASE_URL=https://api.deepseek.com/v1` |
| 3 | 无论文官方 A100 上的 DL 成品权重 | 仓库无作者 `.pth` 发布；`scripts/install_official_checkpoints.py` 只安装本地已有文件 |
| 4 | Windy Power / Sunny（Sundy）Power / MOPEX 无与作者一致的处理后数据 | `docs/数据缺口_Windy_Sundy_MOPEX.md`；MemCast Drive 核验见 `scripts/memcast_drive_inventory.json` |

开源实现仓库名为 **CastMind**（README）；论文脚注为 `AlphaCast_Official`。本工作区是 CastMind 的本地 fork。

### 0.3 标注约定

| 标记 | 含义 |
|------|------|
| 复现事实 | 代码/日志/产物可核对 |
| 【文档记载，原始产物未保留】 | 只出现在 `docs/` 旧报告中，当前 `outputs/` 已被覆盖 |
| 【本机文档互相冲突，需人工确认】 | 两份本地文档数字不一致 |
| 【分析/解释】 | 我对复现的理解 |
| 【论文未明确说明】 | 论文没写、实现里自行选择的超参 |

---

## 1. 环境与入口

### 1.1 推荐运行时（当前）

| 项 | 本机事实 |
|----|----------|
| 虚拟环境 | 项目根目录 `.venv` |
| transformers | **4.40.1**（`requirements.txt` 钉死；本机 `.venv` 实测 4.40.1） |
| 目的 | Sundial 需要 transformers 4.x cache API |
| 入口 | `bash scripts/setup_env.sh`；跑数 `bash scripts/run.sh` |
| 密钥 | 只放 `.env`，不写入本文件 |

旧文档曾描述 `.venv-sundial` 侧环境与 `scripts/setup_sundial_env.sh`。当前 git 状态显示这些脚本已删除/改名；**当前推荐路径是单一 `.venv` + `transformers==4.40.1`**（`RUNBOOK.md`）。

### 1.2 LLM

```text
MODEL=deepseek-chat
ORCHESTRATION_MODE=llm   # 或 deterministic
```

论文 GPT-5；本机 DeepSeek。`.env_template` 写明：「AlphaCast paper §4.1 uses GPT-5. This reproduction uses DeepSeek instead.」

### 1.3 协议配置（`config.yaml`）

**EPF（BE/DE/FR/NP/PJM）：**

```text
look_back: 168
predicted_window: 24
sliding_window: 168
frequency: h
```

**Long-term（ETTh1 / ETTm1 / windy_power / sunny_power / MOPEX）：**

```text
look_back: 96
predicted_window: 96
sliding_window: 96
```

ETTh1 / EPF：`frequency: h`；ETTm1 / windy / sunny：`frequency: 15min`；MOPEX：`frequency: D`。

论文列名 **ETTh** 对应本仓库数据集 **`ETTh1`**（`RUNBOOK.md`）。

---

## 2. 实现对照：四类信息源与三 Agent

### 2.1 Feature Library

实现：`castmind/features/extract.py`。

本机 ETTh1 `outputs/ETTh1/features.json` 实际键（20 个量级，与 EPF_NP 日志 “Target feature count: 20” 一致）：

- basic_count, basic_mean, basic_std, basic_min, basic_max, basic_skew, basic_kurt
- spectral_entropy
- crossing_points, flat_spots, lumpiness, entropy
- x_acf1, x_acf10, diff1_acf1, diff1_acf10, diff2_acf1, diff2_acf10, seas_acf1
- seasonal_strength（缺省时用 seas_acf1 代理）

最新 LLM 跑次 `selected_features.json` 只保留：

```text
basic_count, basic_mean, basic_std
```

权重各 1/3。这与论文 Case Study「偏好 mean/std/count」方向一致，但是 **DeepSeek 本机选择**，不能当成论文 Figure 6 的复现。

### 2.2 Knowledge Base / Contextual Repository

本机没有独立的「论文 Knowledge Base 数据库」。接近的材料是：

- `prompts/contextual_briefings/*.txt`（如 ETTh1：油温 OT、负荷列、日季节等）
- Investigator packet 里的 dataset briefing 字段（`castmind/agents/common.py`）

【分析/解释】briefing 文本同时承担论文里的 conceptual knowledge 与 contextual cues，工程上未严格拆成 Knowledge Base vs Contextual Repository。  
不要把它解释成「涨跌原因库」。

### 2.3 Case Library

实现：`castmind/tools/analysis.py` → `analyze_training()`。

| 项 | 本机事实 |
|----|----------|
| 聚类 | **写死 K-means**（`cluster_by_kmeans`，注释对齐论文 §3.3.5） |
| 默认 \(k\) | **6**（`num_clusters: Optional[int] = 6`） |
| 中心 | `sklearn.cluster.KMeans`，`random_state=0`，`n_init=10` |
| 窗口表示 | look-back 做 **z-score** 再入库 |
| 训练窗比武指标 | **MSE**（`np.mean((pred-fut)**2)`） |
| 簇内聚合 | `method="weighted"`：簇内各模型获胜次数全量计入；权重 \(w_i = n_i / \sum_j n_j\)（论文式 7）；**无「次数 ≤3 丢掉」过滤**（旧上游启发式已去掉） |
| 测试检索 | 当前窗 z-score 后与簇中心欧氏距离最近 |
| 辅助预测 | 簇内加权 `reference_prediction`（代码字段名） |
| 近邻 | `choose_neighbor_by_similarity` 取 top-1 look-back 与 pred |

【论文未明确说明】k=6、z-score——这是**本机/开源实现选择**（加权公式对齐式 7，无 count 阈值过滤）。

> 复现事实：当前预测 **不是**「全局历史最佳模型直接输出」。测试时先簇加权得到 `reference_prediction`，再交给 Generator；deterministic 模式才会把相似度建议模型的输出当作最终预测。

最新 ETTh1 `cluster_base.json`：**6 个簇**（k=6 得到验证）。

最新 `cases_stats.json`（2026-09-09 LLM 跑次，训练窗获胜计数，总和 **102** 窗）：

```text
PatchTST 33, TimesNet 22, Chronos 9, AutoCES 7, Prophet 5, DLinear 5,
AutoARIMA 4, TimeXer 4, iTransformer 3, Sundial 3, SeasonalNaive 2,
CrostonClassic 2, Autoformer 2, DynamicOptimizedTheta 1
```

季节：`memory.json` 中 `periodicity_lag=24`，`frequency: h`（ETTh1）。

### 2.4 三个 Agent（`ORCHESTRATION_MODE=llm` 默认）

| Agent | 本机实现 | 要点 |
|------|----------|------|
| Investigator | **LLM**（DeepSeek / `MODEL`）+ 工具 `gather_forecast_inputs` | `create_investigator_agent` → `Agent(model_name)`；打包 F / \(F_{\text{selected}}\) / K / E；案例检索由 Generator 侧合并 |
| Generator | **LLM**（DeepSeek） | `consult` → 读 packet → `record_chain_of_thought` → `emit_predictions` |
| Reflector | **LLM**（DeepSeek）+ 工具 `assess_forecast` / `scan_chain_of_thought` | 默认 LLM 审计；仅 `CASTMIND_RULES_REFLECTOR=1` 时退回规则 `FunctionModel` |

论文骨干为 GPT-5；本机三 Agent 骨干均为 DeepSeek → **流程对齐，系统行数字不宣称对齐**。  
`deterministic` 模式或 LLM 构建失败时走确定性回退（非默认全量评测路径）。

论文 §3.4.1 案例检索归属 Generator；本机 Investigator 不取案例，由 Generator `consult` 合并案例证据。  
Generator 把 `reference_prediction` 当作辅助基线，仅在证据充分时微调（`prompts/generator_agent.md`）。

---

## 3. 候选模型池（当前代码）

`castmind/models/base.py` → `get_default_models()`（**未提交 diff 已把 HoltWinters/Theta/ZeroModel 移出默认池**）：

**进入默认池（有依赖/权重才加入）：**

- 统计：SeasonalNaive, HistoricAverage, AutoARIMA, Prophet, AutoCES, CrostonClassic, DynamicOptimizedTheta（论文 Optimizers）
- DL（本地 `.pth` 存在才加入）：Autoformer, DLinear, PatchTST, TimesNet, iTransformer（论文 Transformer）, TimeXer
- 基础模型：Sundial（目录存在且 transformers major<5）、Chronos（`chronos-bolt-base` 目录存在）

**显式不进 Table 1 主池（类仍保留）：** HoltWinters, Theta, ZeroModel, TimesFM。

### Chronos

```text
amazon/chronos-bolt-base
本地目录：castmind/foundation_models/chronos-bolt-base
```

推理：`pipe.predict_quantiles(inputs=context, prediction_length=h, quantile_levels=[0.5])`。  
设备：`cuda if available else cpu`（不用 MPS）。

### Sundial

```text
本地目录：castmind/foundation_models/sundial-base-128m
环境：.venv + transformers==4.40.1
```

`generate()` 后按 shape 取最后 h 步；长度不符则截断或报错（output 整形）。

### DL 权重身份

当前 `config.yaml` 指向 `castmind/DeepLearningCheckpoints/<ds>/*.pth`（五模型：Autoformer / DLinear / iTransformer / PatchTST / TimesNet；**不跑 TimeXer**）。

**本机主用：TSLib 官方划分自训**（Train 拟合、Val 选模）——**不是**作者 Table 1 权重，但**不因** CastMind 曾把 Val 并进 `data/*/train.csv` 而失效。  
本仓 `scripts/train_checkpoints.py` 为可选补训入口；若历史上读的是合并后的 `train.csv`，那些 light 权重才与「纯 Train」不一致，可弃用或 `--force` 重训。  
EPF 若在 Price 列写反期间训过，仍须重训（与三分法无关）。

---

## 4. Bug 修复（影响能否跑通）

来源：git `12a4c10`、`docs/工作进展.md`、当前代码注释。均为**本机/开源工程修复**，不是论文内容。

### 4.1 Chronos API

- 问题：新 Chronos API 需要 `inputs=`，旧 `context=` 失效。
- 修复：`predict_quantiles(inputs=context, ...)`（`castmind/models/base.py`）。

### 4.2 async Reflector

- 问题：在同步 tool 内嵌套 `reflector_agent.run_sync` 会死锁。
- 修复：`emit_predictions` 改为 `async`，`await reflector_agent.run`。注释原文：`Nested run_sync deadlocks inside a sync tool during an agent run; use async.`

### 4.3 frequency / 时间特征维数

- 问题：推理时间特征用 `t/min`（5 维）与按 `h` 训练的 checkpoint（4 维）不一致，DL 无法加载。
- 修复：`timefeat_freq` / `args.freq` 对齐为 `'h'`。
- 另：`castmind/utils/time.py` 按 frequency 映射季节周期（小时→24），避免 SNaive 退化。

### 4.4 output / 预测长度

- Generator 若 LLM 输出长度 ≠ `predicted_window`：trim 或用末值 padding（`[warn] Normalizing LLM predictions length`）。
- Sundial：对 `generate` 输出 shape 做 2D/3D 分支，保证返回长度为 h。
- Reflector：避免把日期、`Step 0`、`window_offset` 误判为无依据数字或 horizon 冲突（`_WINDOW_CLAIM_PATTERN` / `_looks_like_date_fragment`）。

### 4.5 其他重要问题（代码里仍存在）

| 现象 | 事实 |
|------|------|
| `metadata.json` 的 `predicted_window` | 最新 LLM 产物写的是 **48**，来自最后一次 `emit_predictions` 的 H 参数，**不能**代表全程窗长 |
| 实际预测 CSV | 25 个窗 × 96 + **最后 1 窗 48 点** = **2448** 行（完整 26×96 应为 2496） |
| 论文 ETTh1 test 长度 | 2544（Table 5）；本机评测用对齐后的 2448 点 |
| HoltWinters 9.266 跑次 | 当时默认池**含** HoltWinters/Theta/ZeroModel，**不含** DL（见下节日志） |
| 当前默认池 | 已改为论文 Table 1 方法集（HoltWinters 不再入池） |

---

## 5. 真实实验（必须分跑次，禁止混成一张「最终表」）

下列数字均来自本机日志/JSON。不同跑次的模型池、是否含 Sundial、是否 LLM **不可横向当成同一实验的重复**。

### 5.1 用户点名的 ETTh1 deterministic（需与「当前 LLM」分开）

用户给定、且被 `docs/复现状态.md` + `docs/_archive/阶段D_指标记录.md` + 日志证实的一轮：

```text
数据集: ETTh1
模式: deterministic（无 LLM）
聚类: K-means k=6
MSE = 9.265714
MAE = 2.567678
sMAPE = 0.168206
选中模型: HoltWinters
```

日志：`outputs/_archive/ETTh1_paper_protocol_deterministic.log`

该日志还证实：

- `[info] Case-library clustering: K-means (k=6)`
- 使用 Sundial 侧环境 `.venv-sundial`，`transformers=4.40.1`
- 当时基线池（11）：`SeasonalNaive, HistoricAverage, AutoARIMA, HoltWinters, Theta, AutoCES, CrostonClassic, DynamicOptimizedTheta, ZeroModel, Sundial, Chronos`
- **没有** Autoformer/DLinear/PatchTST 等 DL
- `[info] Model selection: similarity suggested=HoltWinters`

**Case Library 是否包含 Sundial、是否 7 窗：**

- `docs/复现状态.md`：「案例库含 Sundial（本轮 **7** 窗）」
- `docs/_archive/阶段D_指标记录.md` / `docs/_archive/reproduction-report.md`：侧环境「案例库选中 **10** 窗」

【本机文档互相冲突，需人工确认】7 vs 10。  
当前磁盘 `outputs/ETTh1/cases_stats.json` 已被 2026-09-09 LLM 跑次覆盖为 `Sundial: 3`，**9.266 那次的 cases_stats 原始文件未保留**。

【分析/解释】9.266 是 deterministic 下相似度选出 HoltWinters 后、在测试窗上用该模型预测的分数，**不是** AlphaCast 三阶段 LLM 系统行。

### 5.2 当前 ETTh1 LLM 系统行（归档可复核）

产物：`outputs/_archive/ETTh1_llm_deepseek/`（与 `outputs/comparison/ETTh1_castmind_CastMind_llm_deepseek.json` 一致时可复核）。

| 项 | 值 |
|----|-----|
| 模式 | LLM，`chosen_model: LLM` |
| 骨干 | DeepSeek `deepseek-chat` |
| 预测点数 | 2448 |
| MSE | 7.516505 |
| MAE | 2.107981 |
| sMAPE | 0.216201 |
| 来源 | `outputs/comparison/ETTh1_castmind_CastMind_llm_deepseek.json`（2026-09-11 核对） |

论文 AlphaCast ETTh：**MSE 7.641 / MAE 2.017**。  
本机数字接近**不能宣称复现**（LLM≠GPT-5，DL 为 light 自训）。

历史文档中的 7.651 / 6.973 / 7.403 等为旧跑次或已被覆盖的产物，仅作历史引用。

### 5.3 十数据集 LLM 全量（2026-09-11 快照）

权重：`DeepLearningCheckpoints/<ds>/` light 五模型；**不跑 TimeXer**。命令见 [RUNBOOK.md](../RUNBOOK.md) §4。

| 数据集 | 归档 | n_points | MSE | MAE | 备注 |
|--------|------|--------:|----:|----:|------|
| EPF_NP / PJM / BE / FR / DE | 有（**失效**） | ≈432 | （修前列错目标） | | 2026-09-13 已修 `Price` 列；须重跑 LLM 才有有效系统行 |
| ETTh1 | 有 | 2448 | 7.517 | 2.108 | 主复现列 |
| ETTm1 | 有 | ≈4800 | 8.163 | 2.235 | 2026-09-11 续跑完成 |
| MOPEX | 有 | 2448 | 348624.284 | 331.190 | **代理数据**，不与论文比 |
| windy_power | 有 | 4800 | 1710.437 | 36.721 | **代理数据**，不与论文比 |
| sunny_power | 有 | 4800 | 259.604 | 8.768 | 同上 |

**纪律：** EPF 的 432 点是协议预期，不是失败；WP/SP/MOPEX 禁止与 Table 1 逐格对比；全体 DeepSeek 系统行不宣称复现 AlphaCast。  
EPF 修前「MSE 几十万」主因是 **Price/外生列对调**，不是模型崩了。

历史 deterministic 冒烟（ETTm1 9.215、EPF_NP 33 万等）保留在旧日志，**不再代表当前系统行**。

### 5.4 本机 Table 1 风格基线评测（全模型入口）

**全数据集 × 全基线评测入口是 `scripts/eval_baselines_table.py`（再 `assemble_comparison_table.py`），不是 `run_experiment.py`。**  
`--from-archive` 放命令末尾；十套 LLM 已齐后按 RUNBOOK §5 重跑。下表为历史 ETTh1 切片（light/旧权重口径可能与最新重跑略有出入）：

来源：`outputs/comparison/ETTh1_baselines.json`（look_back=96, predicted_window=96, sliding_window=96, season_length=24, n_points=2448）。

| 方法 | 本机 MSE | 本机 MAE | 论文 Table 1（ETTh）MSE/MAE |
|------|--------:|--------:|---------------------------|
| SeasonalNaive | 9.320 | 2.350 | 10.753 / 2.469 |
| HistoricAverage | 8.541 | 2.375 | 10.309 / 2.571 |
| AutoARIMA | 14.587 | 2.859 | 10.879 / 2.429 |
| Prophet | 47.257 | 4.835 | 46.697 / 4.592 |
| AutoCES | 12.867 | 2.772 | 24.218 / 3.646 |
| CrostonClassic | 8.652 | 2.255 | 9.719 / 2.331 |
| DynamicOptimizedTheta | 9.298 | 2.368 | 11.772 / 2.541 |
| Autoformer | 8.417 | 2.368 | 11.598 / 2.637 |
| DLinear | 8.473 | 2.311 | 8.506 / 2.239 |
| PatchTST | 7.447 | 2.108 | 8.396 / 2.169 |
| TimesNet | 8.926 | 2.281 | 7.940 / 2.079 |
| iTransformer | 7.703 | 2.194 | 8.307 / 2.136 |
| TimeXer | 7.310 | 2.172 | 8.765 / 2.204 |
| Sundial | 7.676 | 2.174 | 9.437 / 2.314 |
| Chronos | 9.822 | 2.389 | 9.397 / 2.250 |
| CastMind_llm_deepseek | 7.517 | 2.108 | AlphaCast 7.641 / 2.017 |

**读表纪律：**

- 公开权重（Chronos/Sundial）可谈数量级，**不宣称格级复现**。
- 自训 DL 行（含历史表中的 TimeXer）**禁止**写成「超过作者」；本轮全量跑**不评测 TimeXer**。
- CastMind 接近论文 7.641 是**巧合级接近 + 协议相近**，仍不得称复现成功。

---

## 6. 数据准备（与作者一致性）

`scripts/prepare_data.py`（**2026-09-16 起** `write_split` 写出独立 `train.csv` / `val.csv` / `test.csv`，不再把 Val 并进 train）：

| 数据集 | 划分常量 | 来源 |
|--------|----------|------|
| EPF×5 | (10224, 1584, 3024) | 与论文 Table 5 一致；Zenodo |
| ETTh1 | (8544, 1344, 2544) | 与论文 Table 5 一致；ETDataset |
| ETTm1 | (16896, 2496, 4896) | 与论文 Table 5 一致 |
| Windy/Sunny | 代码用 POWER_SPLITS 同上 | **序列为 Open-Meteo 代理，不可对齐作者** |
| MOPEX | 代码 `MOPEX_SPLITS = ETTH_SPLITS`，注释写「AlphaCast Table 6」 | 论文 Table 5 **无 MOPEX 行**；测站 `01022500` 为 stand-in |

系统 `training_csv` = **仅 Train**；案例库历史窗按纯 Train 重建（ETTh1 约 88 窗，不再是合并时的 102）。  
**DL：** TSLib 官方划分权重可继续用；Agent/LLM 须按新 Train 重跑。

**EPF 列顺序（2026-09-13 已修）：** Zenodo/epftoolbox 为 `Price, Exo1, Exo2`。旧 `prepare_epf` 误写成外生在前、`Price` 取第 3 列。  
**后果：** 修前 EPF LLM 归档与错误目标上的 EPF DL 权重作废；TSLib 若在修后数据上按官方划分重训的 EPF 权重可用。

主复现数据集：**ETTh1**（公开同源）。十套中 EPF/ETTm 可协议级复现；WP/SP/MOPEX 只保留流水线（代理数据），不纳入与论文逐格对比。

---

## 7. 硬件

文档记载：Apple M5，16 GB，MPS 可用，无 NVIDIA A100（`docs/工作进展.md`、`docs/汇报_AlphaCast论文与复现.md`）。
论文 baseline 在 A100 上跑。  
`PAPER_ALIGNMENT.md`：对公开 FM，设备差通常只造成次要浮点残差，不是对不齐主表的主因。

---

## 8. 复现结论（对外口径，写死）

1. 完成了 CastMind 三阶段流水线在 ETTh1 上的 **可运行验证**（案例库 K-means、DeepSeek Investigator/Generator/Reflector、Chronos bolt-base、Sundial in-pool）。  
2. **不是**论文官方结果的严格数值复现。  
3. 应称为：**代码/流程级复现验证**。  
4. 不应称为：完整复现论文、复现 Table 1、本机结果超过 AlphaCast。

原因仍是第 0.2 节四条，外加：DeepSeek≠GPT-5、DL 自训≠作者 `.pth`、WP/SP/MOPEX 代理数据、EPF 短覆盖等（**不是**「Reflector 仍是规则 FunctionModel」——默认 llm 下已是 LLM）。
