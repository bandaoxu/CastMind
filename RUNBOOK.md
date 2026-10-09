# CastMind 本机实验手册

更新：2026-10-09。原始模型与数据来源需按实际条件披露；本地论文记录不随仓库发布。以下命令在仓库根目录执行；运行会消耗本地计算资源，LLM 模式还会调用所配置的 API。

## 1. 环境与材料

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
cp -n .env_template .env
```

在 .env 配置 OPENAI_API_KEY、OPENAI_BASE_URL 和 MODEL，不提交密钥。历史使用 deepseek-chat；实际模型按实验记录披露。requirements.txt 固定 transformers==4.40.1。数据准备入口为 scripts/prepare_data.py，先检查现有数据及来源，再决定是否重新下载/生成；代理数据不能改名充当作者数据。

检查 data/<ds>/{train,val,test}.csv 的目标列、行数、时间边界和哈希。ETTh1 为 8544/1344/2544；EPF 为 10224/1584/3024。系统只用 Train 建案例库。

config.yaml 的五个 DL 路径并不证明权重来自何处。保留训练日志、种子、权重哈希；正确的 Train/Val 自训权重可以继续使用，旧合并训练集或错误 Price 目标训练的权重需重新审核。可选补训：

```bash
bash scripts/run.sh scripts/train_checkpoints.py --datasets ETTh1 --preset official --epochs 10
```

official 是本地预设名，不代表作者训练配方或作者权重。训练后核对产出路径与 config。Chronos/Sundial 目录、依赖及实际加载结果也需检查。

## 2. 冒烟与正式运行

每次必须给出唯一 --run-name；若名字已存在，新实验改编号，续跑才用 --resume。

```bash
ORCHESTRATION_MODE=deterministic bash scripts/run.sh --dataset ETTh1 --run-name deterministic_smoke_1
CASTMIND_MAX_STEPS=2 ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset ETTh1 --run-name llm_smoke_1
ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset ETTh1 --run-name Full_2
```

正式运行前移除环境中 CASTMIND_MAX_STEPS 限制。deterministic 只能证明相应路径可运行，不能冒充三 Agent 完整系统。

```bash
ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset ETTh1 --run-name Full_2 --resume
```

续跑保持原配置和环境；manifest 会核对数据、运行配置、源码及提示词；不匹配会拒绝。案例库缺失或核心文件变化也会拒绝续跑。不要删 CSV 后在原目录重跑。旧迁移 manifest 可能没有完整指纹，不能保证可以续跑。

## 3. 案例库与产物

- outputs/<ds>/runs/<run_name>/：predictions.csv、metrics.json、run_manifest.json，以及实际生成的 emissions.jsonl、chain_of_thought.log、llm_resume_state.json 等。
- outputs/<ds>/case_libraries/<lib_id>/：memory.json、case_base.json、case_neighbor.json、cluster_base.json 及 library_manifest.json 等。

案例库复用要求 manifest 匹配训练文件、目标列、窗长和步长，并校验核心文件哈希；无 manifest 的旧库不会自动认证或复用。案例库指纹尚不覆盖全部模型权重与建库代码，依赖变化须强制新建，不能仅凭“复用成功”证明来源正确。来源不明或依赖改变时，新开实验并强制重建：

```bash
CASTMIND_FORCE_ANALYZE=1 ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset ETTh1 --run-name Full_rebuild_1
```

该开关用于新建，不作为同名续跑的通用选项。保留旧库供溯源。

旧产物迁移先审计：

```bash
.venv/bin/python scripts/migrate_outputs_to_runs.py --dry-run
```

确认报告后再决定是否去掉 --dry-run。迁移不证明旧结果满足当前科研协议。

## 4. 基线评测与表格

先为一个经过有效性核验的系统跑次明确指定预测文件。下例 Full_2 是示例名，应替换为实际通过检查的跑次。

```bash
bash scripts/run.sh scripts/eval_baselines_table.py \
  --datasets ETTh1 --output-dir outputs/comparison/ETTh1_Full_2 \
  --from-archive CastMind_Full_2=outputs/ETTh1/runs/Full_2/predictions.csv
.venv/bin/python scripts/assemble_comparison_table.py \
  --datasets ETTh1 --comparison-dir outputs/comparison/ETTh1_Full_2 \
  --out-md outputs/comparison/ETTh1_Full_2/table.md \
  --out-copy-md outputs/comparison/ETTh1_Full_2/table_local.md \
  --out-csv outputs/comparison/ETTh1_Full_2/table.csv
```

--from-archive 不带值时，当前脚本只找旧扁平 outputs/<ds>/predictions.csv，不自动发现 runs/。多数据集须逐套传对应文件，不要把一个系统 CSV 传给所有数据集。比较目录按实验隔离，避免旧片段混入。

表格脚本仍会读取 paper_table1_reference.json 中的历史论文数值；输出版式或误差差值不能证明可比。定稿前核对论文版本，将原论文成绩和本地同条件比较分开。

## 5. 消融与扩展

```bash
ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset ETTh1 --ablation no_case --run-name no_case_2
ORCHESTRATION_MODE=llm bash scripts/run.sh --dataset ETTh1 --ablation no_reflect --run-name no_reflect_2
```

其他选项为 no_feature、no_knowledge、two_stage、enhanced_reflect。v1/v2 特征检索保留在独立实验分支，本分支不提供该选项；two_stage/enhanced_reflect 为本地实现的推理长度实验，不保证精确等价于论文全部细节。结果优劣由实验决定。

本分支保留原始案例路线；Full 与消融除研究因素外保持一致。提示词、权重或代码改变后，重新建立可比的 Full，不能直接拿迁移后的 Full_1 作对照。

## 6. 覆盖与有效性

当前 ETTh1 test=2544、L=96 时，完整可预测部分为 2448 点（尾窗可为 48）；EPF test=3024、L=168、stride=24 时为 2856 点。以实际 CSV 和时间戳核验，不只检查 metrics 中的 n。

重复最终时间戳当前会导致评测报错，不能静默去重凑分。成功退出、非空指标或迁移 format_checked 都不足以认定主实验有效；按 实验有效性检查要求 检查目标列、案例库来源、完整覆盖和同条件基线。

## 7. 扩展顺序与故障定位

先 ETTh1，再 ETTm1/一个 EPF，最后按需要扩展其余公开集；代理集单独标注。Sundial 未入池检查依赖和模型目录；DL 未入池检查配置路径、加载日志与结构参数；LLM 中断检查该跑次日志和 resume 状态。反复拒收须分析原因，不能以强制接受代替实验验证。

此手册随通用隔离修复更新；迁移只复制旧产物并标记格式检查结果，实验有效性仍需独立核验。
