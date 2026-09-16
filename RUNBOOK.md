# CastMind 本机跑数手册

论文 Table 1 长序列列名 **ETTh** 对应本仓库数据集 **`ETTh1`**。  
唯一推荐环境：项目根目录 `.venv`（`transformers==4.40.1`，可跑 Sundial）。  
密钥只放在 `.env`，勿写入本文件。

**详细思维导图 + 逐步说明：**  
- 与论文对齐（推荐）：[PAPER_ALIGNMENT.md](./PAPER_ALIGNMENT.md)  
- 纯跑数操作树：[WORKFLOW.md](./WORKFLOW.md)  
- 组会分享讲稿：[docs/组会分享_AlphaCast.md](./docs/组会分享_AlphaCast.md)  
  - PPT（沿用模版版式）：`docs/组会分享_AlphaCast.pptx` ← `python scripts/build_group_talk_pptx.py`  
  - 模版文件：`docs/AlphaCast论文阅读与复现进展汇报.pptx`

## 1. 环境（一次）

```bash
cd /path/to/CastMind
bash scripts/setup_env.sh
source .venv/bin/activate
# 复制并填写密钥
cp -n .env_template .env   # 若尚无 .env
```

`.env` 常用项（本机**临时用 DeepSeek**；论文骨干是 GPT-5）：

- `ORCHESTRATION_MODE=llm`
- `OPENAI_BASE_URL=https://api.deepseek.com/v1` / `OPENAI_API_KEY` / `MODEL=deepseek-chat`

**允许例外：** (1) API Key (2) 自训 DL `.pth` (3) sunny/windy/MOPEX 代理 (4) **临时 DeepSeek≠GPT-5（须披露，不宣称系统行复现主表）**。其余协议不要为刷分改。有 GPT-5 后把 `MODEL`/`BASE_URL` 改回论文设置。

校验：

```bash
.venv/bin/python -c "import transformers; print(transformers.__version__)"
# 期望 4.40.x
```

## 2. DL 权重（默认用现成 light）

本轮默认使用 [`castmind/DeepLearningCheckpoints/<数据集>/`](castmind/DeepLearningCheckpoints/) 下已有权重（**不是**名为 `light.pth` 的单文件；**light** 指相对 `*_official/` 的训练预设）：

- `Autoformer.pth` / `DLinear.pth` / `iTransformer.pth` / `PatchTST.pth` / `TimesNet.pth`
- [`config.yaml`](config.yaml) 已指向上述路径
- **不跑 TimeXer**（无 `TimeXer.pth`，本轮不补训、不评测）
- **不强制** `train_checkpoints.py --preset official`

## 3. 单数据集冒烟 / 主复现列（ETTh1）

Deterministic（快，无 LLM；多为单窗冒烟，完整测试窗请用 llm）：

```bash
ORCHESTRATION_MODE=deterministic bash scripts/run.sh --dataset ETTh1
```

LLM 全测试窗（本机临时 DeepSeek；论文为 GPT-5）：

```bash
ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset ETTh1
```

产物：`outputs/ETTh1/` → 归档 `outputs/_archive/ETTh1_llm_deepseekchat/`（由 `MODEL` 生成 slug）。

## 3.1 Investigator \(F_{\text{selected}}\)（论文式 6）

[`config.yaml`](config.yaml)：`feature_selection: paper`（默认；可选 `rules` / `off`）。  
严格核对说明：[`docs/论文方法完整核对.md`](docs/论文方法完整核对.md)。

```bash
.venv/bin/python scripts/smoke_feature_selection.py --mode paper --dataset ETTh1
# 或 rules 冒烟（无需选特征 API）：
.venv/bin/python scripts/smoke_feature_selection.py --mode rules --dataset ETTh1
```

## 4. 十数据集全量跑（系统行）

`config.yaml` 中的十套：`EPF_NP EPF_PJM EPF_BE EPF_FR EPF_DE ETTh1 ETTm1 windy_power sunny_power MOPEX`。

**推荐：按优先级逐套跑**（半截失败会 exit 1；逐套更容易盯 Reflector / 续跑）。有 `outputs/<ds>/llm_resume_state.json` 时，**同命令**即从断点续。

优先级：EPF×5（Price 已修，旧归档失效）→ ETTh1 → ETTm1 → windy / sunny / MOPEX（代理，最后）。

```bash
cd /path/to/CastMind

# —— 单套（推荐；可反复续跑直到归档成功）——
ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset EPF_NP
# ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset EPF_PJM
# … 其余同理

# —— 或：EPF 五套串行（一套失败则 stop，修好/再续后再开下一批）——
for ds in EPF_NP EPF_PJM EPF_BE EPF_FR EPF_DE; do
  echo "===== $ds ====="
  ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset "$ds" || break
done

# —— 长序 / 代理（EPF 与主复现列跑稳后再开）——
for ds in ETTh1 ETTm1 windy_power sunny_power MOPEX; do
  echo "===== $ds ====="
  ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset "$ds" || break
done

# —— 十套一次性循环（仅在单套已能稳定跑通后使用；勿去掉 || break）——
for ds in EPF_NP EPF_PJM EPF_BE EPF_FR EPF_DE ETTh1 ETTm1 windy_power sunny_power MOPEX; do
  echo "===== $ds ====="
  ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset "$ds" || break
done
```

成功：非空 Experiment Summary + 新时间戳的 `outputs/_archive/<ds>_llm_<MODEL>/predictions.csv`。  
若要从零重跑某套（丢掉半截进度）：

```bash
rm -f outputs/<ds>/llm_resume_state.json outputs/<ds>/predictions.csv
ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset <ds>
```

- 中断 / Reflector 拒收 / 网络错误：会写 `outputs/<ds>/llm_resume_state.json`，**同命令重跑即可续跑**；未完整结束时 Experiment Summary 可能为空、**不会自动归档**，且进程 **exit 1**（`for … || break` 会停在该套）。
- 完整结束后：`outputs/_archive/<ds>_llm_<MODEL>/`（含 `predictions.csv`）。
- Reflector 默认仍是 **LLM**（论文路径）：工具 `deterministic_audit` 提供证据，**由 LLM 决定** `approved`。拒收后特征重选走 `prepare_investor_packet`。勿默认开 `CASTMIND_RULES_REFLECTOR=1`。
- **EPF（短序）：** `look_back=168`、`predicted_window=24`、`sliding_window=24`（day-ahead，与长序「stride=horizon」一致）。约 **2856** 点满测；旧配置 stride=168 只评 ~408 点，**已废弃**，勿再当论文对照。
- **长序**（ETTh1 / ETTm1 / MOPEX 等）：`sliding_window=96`，可接近满覆盖（如 ETTh1/MOPEX 2448 点；ETTm1 约 4800 点）。
- **WP / SP / MOPEX**：本机为代理数据，可跑通流水线，**禁止与论文 Table 1 逐格对比**。见 [`docs/数据缺口_Windy_Sundy_MOPEX.md`](docs/数据缺口_Windy_Sundy_MOPEX.md)。

### 4.1 进度快照（2026-09-13）

十套 LLM 均已归档：`outputs/_archive/<ds>_llm_deepseek/predictions.csv`。

| 数据集 | LLM 归档 | 备注 |
|--------|----------|------|
| EPF_NP / PJM / BE / FR / DE | **失效待重跑** | 2026-09-13 前归档在**错误 Price 列**上；数据已修，须重跑 LLM |
| ETTh1 | 有 | n=2448 |
| ETTm1 | 有 | n≈4800 |
| MOPEX | 有 | n=2448；代理数据 |
| windy_power / sunny_power | 有 | 代理数据；已完成 |

**EPF 数据修复（2026-09-13）：** 曾把 Zenodo `Price` 与外生列写反（`prepare_epf`），导致与论文差几个数量级；已按 epftoolbox 顺序纠正并重写 `data/EPF_*`。基线请用修后数据重评；**旧 `*_llm_deepseek` 归档不可作系统行验收。**

## 5. 基线表 + 对比表

**入口是 `eval_baselines_table.py`（各基线独立滑窗），不是 `run_experiment.py`。**  
`--from-archive` 须放在命令**末尾**（`nargs=*`，避免吞掉后面参数）。先修过共享 list 污染：每套数据集各自 `list(...)` 再发现归档路径。TimeXer 本机无权重 → 表中为 `—`。

```bash
# 1) 全数据集 × 池内全部模型 → MSE/MAE（含归档 CastMind 行）
bash scripts/run.sh scripts/eval_baselines_table.py \
  --datasets EPF_NP EPF_PJM EPF_BE EPF_FR EPF_DE ETTh1 ETTm1 windy_power sunny_power MOPEX \
  --from-archive

# 2) 合并论文参考（scripts/paper_table1_reference.json，十套齐全）→ 对比表
.venv/bin/python scripts/assemble_comparison_table.py \
  --datasets EPF_NP EPF_PJM EPF_BE EPF_FR EPF_DE ETTh1 ETTm1 windy_power sunny_power MOPEX
```

产物：`outputs/comparison/<ds>_baselines.json`、CastMind 行 JSON、[`docs/comparison_table.md`](docs/comparison_table.md)（副本 `outputs/comparison/table1_local.md`）。

**读表纪律：** 版式/方法行可与 Table 1 对照；**数值不宣称复现**（DeepSeek≠GPT-5、自训 DL≠作者 `.pth`、WP/SP/MOPEX 代理）。  
EPF 基线应与论文同量级；若再出现 MSE 几十万，先查 `Price` 列是否又写反。EPF 的 CastMind 归档行在重跑 LLM 前视为**失效**。

## 6. 可选：TSLib 风格 DL 自训

**非本轮默认。** 入口是本仓脚本，不是 [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)。  
细节见 [PAPER_ALIGNMENT.md 分支 E](./PAPER_ALIGNMENT.md#分支-e深度学习近似对齐)。

```bash
.venv/bin/python scripts/train_checkpoints.py --datasets ETTh1 --preset official --epochs 10
```

权重目录：`castmind/DeepLearningCheckpoints/ETTh1_official/`（**非**作者官方 `.pth`）。  
改 `config.yaml` 的 checkpoints 指向后再评测。

## 7. 口径提醒

- 公开权重基线（Chronos / Sundial）适合与论文比绝对数量级；设备≠A100 是次要浮点残差。
- 统计：方法 + 协议可对齐，**非**与 Table 1 格级完整一致。
- 自训 light DL + DeepSeek≠GPT-5：不宣称与论文 Table 1 逐格复现。
- 完整复现口径见 [PAPER_ALIGNMENT.md 分支 H](./PAPER_ALIGNMENT.md#分支-h三类基线如何得到--能否复现作者)。
- 数据缺口（WP/SP/MOPEX）：[`docs/数据缺口_Windy_Sundy_MOPEX.md`](docs/数据缺口_Windy_Sundy_MOPEX.md)。
