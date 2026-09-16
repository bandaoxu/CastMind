# AlphaCast 论文事实库

> 用途：组会汇报的**论文侧**唯一依据。本文件只记录能在论文中核对的内容；解释与推断单独标注。  
> **本阶段不生成 PPT。**

## 0. 文献元信息与阅读纪律

### 0.1 文献身份

| 项 | 事实 | 来源 |
|----|------|------|
| 标题 | AlphaCast: A Human Wisdom-LLM Intelligence Co-Reasoning Framework for Interactive Time Series Forecasting | 题名页 |
| 作者 | Xiaohan Zhang, Tian Gao, Mingyue Cheng, Bokai Pan, Ze Guo, Yaguo Liu, Xiaoyu Tao | 题名页 |
| 单位 | State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China | 题名页 |
| arXiv | 2511.08947 | 仓库文档交叉引用；HTML 版 ar5iv |
| 本机 PDF | `/Users/bandaoxu/Desktop/毕设/AlphaCast.pdf`（11 页；**不在** CastMind 仓库根目录） | 文件系统 |
| 官方代码脚注 | https://github.com/SkyeGT/AlphaCast_Official | Abstract 脚注 1 |
| Keywords | Time Series, Interactive Forecasting, Human Wisdom-LLM Intelligence Co-Reasoning | Keywords |

### 0.2 标注约定（本文件强制）

| 标记 | 含义 |
|------|------|
| 无标记或「论文明确」 | 论文原文可核对 |
| 【论文未明确说明】 | 论文没写，禁止用常识补全 |
| 【论文内部不一致】 | 论文不同位置互相冲突 |
| 【基于论文方法的概括】 | 为汇报方便做的结构转写，不是论文原句 |
| 【分析/解释】 | 我对论文的理解，不是论文结论 |
| 【我的问题/讨论】 | 组会可讨论、但论文未下结论的问题 |

数字一律按 PDF/HTML 抄录。Table 1 / Table 2 / Table 3 / Table 4 已与 ar5iv HTML 及 PDF 抽取交叉核对。

---

## 1. 研究背景

### 1.1 时间序列预测传统上如何进行？

论文明确（§1, §2.1, Figure 1）：

- 预测在能源调度、健康预警、气候预估等高风险场景中关键（§1）。
- 领域经历：专家手工分析 → 统计学习/神经网络自动化建模（§2.1）。
- 早期：专家目视识别趋势、季节、异常并做启发式预测（§2.1）。
- 统计模型：ARIMA、Exponential Smoothing (ETS)、Theta 等（§2.1）。
- 机器学习：梯度提升、随机森林等（§2.1）。
- 深度学习：RNN、LSTM、TCN、Transformer 族（§2.1）。
- 时序基础模型：Sundial、Chronos、TimesFM（§2.1）。

论文认为这些路线**大多仍是单次前向的预测器**（§2.1 末句）。

### 1.2 AlphaCast 认为传统预测有什么局限？

论文明确（Abstract, §1）：

- 多数方法仍把预测当作 **static one-time / one-shot mapping**（Abstract, §1）。
- 缺乏人类专家的 **interaction、reasoning、adaptability**（Abstract）。
- 把预测当成纯数据拟合：从过去观测直接映射到未来（§1）。
- 缺少人类预报员常见能力：iterative thinking、contextual reasoning、对先前结论的 reflect 与 revise（§1）。
- 缺少人类使用的多样工具：统计诊断、外部知识与上下文检索、领域模拟器；也缺乏选择与协调这些工具的「human wisdom」（§1）。
- 在外部因素、异构数据、领域知识起关键作用的复杂环境中往往吃力（§1）。
- 缺少 adaptation、interaction、tool-use、reflection，限制灵活性与可信度（§1）。

论文还认为：局限不只来自模型容量，也来自把预测窄化为纯算法任务的概念化方式（§1）。

### 1.3 为什么需要交互式预测？

论文明确：

- 现代真实环境更复杂；有效预测不只要求数值精度，还要求适应模式漂移、感知上下文、支持以人为中心的解释与决策（§1）。
- 主张把预测重定义为 **interactive process**（Abstract）。
- 人类预报员把预测当作多步认知过程（§1，见下一小节）。

【论文未明确说明】「交互式」是否包含人在环路上逐步改数；论文写的是 human wisdom 与 LLM intelligence 的 co-reasoning，以及 agent 流程。

### 1.4 为什么需要外部信息？

论文明确：

- 实践中预测常涉及辅助信息：情境变量 \(C\)（节假日、天气、领域指标）、历史知识 \(K\)、辅助预测器 \(A\)（§3.1）。
- Contextual Repository：节假日、天气、营销活动等外部因素，可补足历史数据理解；在异常事件或突变时尤其有用（§3.3.4）。
- 长序列数据「含三个以上外生变量」，提供更丰富的 conditioning（§4.1）。
- 短序 EPF：电价为内生，并加入两个外生信号（如日历/节假日与系统负荷）（§4.1）。

### 1.5 为什么需要反思？

论文明确（§3.4.2）：

- 传统流程通常是线性单次：常规模型输出直接当作最终预测。
- 引入 reflective mechanism 是为了保证结果可靠，避免盲目依赖单一模型输出。
- Reflector 在 **forecasting level** 与 **CoT level** 做评估与修正。
- Agent ablation（§4.3.2）：去掉 reflection 后，FR、NP、Sunny Power、ETTm 性能下降。

【论文未明确说明】反思的迭代轮数上限、温度、是否使用真实未来值。

---

## 2. AlphaCast 核心思想

### 2.1 论文原句级主张

Abstract / §1：

- 提出 **human wisdom–LLM intelligence co-reasoning framework**，把预测重定义为交互过程。
- 关键想法：human wisdom 与 LLM intelligence **逐步协作**，共同 **prepare、generate、verify** forecasts。
- LLM 不应只是被动预测工具，而应是可协作的主动 agent（§1）。

两阶段（Abstract, §3.2）：

1. **Automated prediction preparation**：构建多源认知地基（feature set、domain knowledge base、contextual repository、case base）。
2. **Generative reasoning and reflective optimization**：整合统计时序特征、先验知识、情境与较小模型的预测策略，触发 meta-reasoning loop 做持续 self-correction 与 strategy refinement。

### 2.2 传统 vs AlphaCast（结构转写）

传统（【基于论文方法的概括】，对应 Figure 1 / §1「one-shot mapping」）：

```text
Historical Data
      ↓
Forecasting Model
      ↓
Future Prediction
```

AlphaCast（【基于论文方法的概括】；论文自己的阶段名是 prepare / generate / verify，角色名是 Investigator / Generator / Reflector）：

```text
Investigation
      ↓
Generation
      ↓
Reflection
      ↓
Iteration
```

论文更贴近的两阶段表述（§3.2，非四步口号）：

```text
Stage 1: Contextual Grounding / Prediction Preparation
  (Feature Set, Knowledge Base, Contextual Repository, Case Library)
      ↓
Stage 2: Reasoning-based Forecasting
  Investigator 选信息 → Generator 生成预测+CoT → Reflector 评估
      ↓（若不合理则反馈 Investigator，迭代至合理）
输出：预测窗口 + 解释
```

> 【基于论文方法的概括】AlphaCast 不是简单地把 LLM 当作又一个时间序列预测模型，而是构建包含信息选择、预测生成、反思与迭代的协同推理流程。  
> 论文没有出现上述整句；对应分散在 Abstract、§3.2、§3.4。

### 2.3 问题形式化（§3.1）

多元时间序列 \(X=\{x_1,\ldots,x_T\}\)，\(x_t\in\mathbb{R}^d\)。  
历史窗口长度记为 \(H\)：\(X_{\text{past}}=\{x_{t-H+1},\ldots,x_t\}\)。  
预测长度记为 \(L\)：\(X_{\text{future}}=\{x_{t+1},\ldots,x_{t+L}\}\)。

标准映射（公式 1）：

\[
f: X_{\text{past}} \longrightarrow \hat{X}_{\text{future}},\quad \hat{X}_{\text{future}}\approx X_{\text{future}}.
\]

AlphaCast 工作流（公式 2）：

\[
(X_{\text{past}}, C, K, A) \xrightarrow{\text{Co-Reasoning}} \hat{X}_{\text{future}}.
\]

其中 \(C\) 为情境，\(K\) 为历史/领域知识，\(A\) 为 auxiliary predictors（论文：提供 baseline forecasts 或作为 pattern retrieval 的参考）。

【论文内部不一致】问题定义里 \(H\)=look-back、\(L\)=horizon；实验协议写短序 input 168 / horizon 24，长序 look-back=96 / prediction=96（§4.1）。同一字母在不同节含义不同。

标准化输入（公式 3，§3.3.1）：

\[
I_{\text{input}}=(I_{\text{tp}}, I_{\text{dp}}, T, X_{\text{exo}}, X_{\text{endo}}).
\]

| 符号 | 论文定义 |
|------|----------|
| \(I_{\text{tp}}\) | task prompt：look-back \(H\)、预测地平 \(L\) 等要求 |
| \(I_{\text{dp}}\) | data profile：数据集属性、变量物理含义 |
| \(T\) | 原始时间戳 |
| \(X_{\text{exo}}\) | 外生，\(\mathbb{R}^{d\times(H+L)}\) |
| \(X_{\text{endo}}\) | 内生回看，\(\mathbb{R}^{H}\) |

---

## 3. 三个 Agent

Figure 2：三个 agent 协同完成时序预测。论文**没有**给出三个 agent 的完整 I/O 接口表；下面只摘原文。

### 3.1 Investigator Agent

**负责什么？**（§3.4.1）

- 收到 task prompt 后做 **semantic parsing** 与 **requirement decomposition**，推断预测所需的特征类型与背景信息。
- 回溯目标序列与外生序列，做 **feature engineering**，从 feature set 中抽出 \(F_{\text{selected}}\)。

公式 (6)：

\[
F_{\text{selected}}=S(F, I_{\text{input}}).
\]

**输入什么？** 论文明确出现：task prompt / \(I_{\text{input}}\)、feature set \(F\)。  
【论文未明确说明】Investigator 是否同时检索 Case Library；§3.4.1 把聚类检索写在 Generator 下。

**输出什么？** 论文明确：\(F_{\text{selected}}\)。  
若预测被 Reflector 判为不合理，Reflector 把 reflective feedback 传回 Investigator，迭代更新（§3.4.2）。  
【论文未明确说明】Investigator 的 JSON schema、是否输出工具调用列表。

**为什么存在？**  
【基于论文方法的概括】把「需要哪些证据」从生成过程中拆出来，先选信息再生成。  
论文原句强调的是解析任务并抽取所需特征（§3.4.1）。

### 3.2 Generator Agent

**负责什么？**（§3.4.1）

- 用 \(X_{\text{endo}}\) 在 Case Library 中按欧氏距离检索最相似簇中心。
- 用该簇内 cases 关联的 local models 做加权预测，得到 auxiliary forecast \(\hat{X}_{\text{auxiliary}}\)（正文符号为 \(X_{\text{auxiliary}}\)）。
- 检索 neighboring series，返回其 look-back 与 prediction 窗口。
- 调用相关工具，汇总四类输入，进入 generative reasoning，得到 raw prediction \(X_{\text{raw}}\)。
- 额外输出 **chain of thought (CoT)**，标明从关键证据到结论的推理，便于归因与迭代。

**输入什么？** 公式 (8) 列出：

\[
(I_{\text{input}}, F_{\text{selected}}, K, E, X_{\text{auxiliary}}, X_{\text{neighbor}}) \xrightarrow{\text{reasoning}} X_{\text{raw}}.
\]

**输出什么？** \(X_{\text{raw}}\) 与 CoT。

用户要求提取的函数写法：

```text
X_raw = Generator(F_selected, K, E, X_auxiliary, X_neighbor)
```

【基于论文方法的概括 / 符号转写】上式便于口播。论文原文是公式 (8) 的箭头形式，且**还包含** \(I_{\text{input}}\)。不要把用户函数式写成论文原公式。

各符号：

| 符号 | 论文中的含义 | 出处 |
|------|----------------|------|
| \(F_{\text{selected}}\) | 从特征集选出的特征，\(S(F,I_{\text{input}})\) | §3.4.1 式 (6) |
| \(K\) | 从 Knowledge Base 取出的 domain knowledge | §3.3.3 |
| \(E\) | Contextual Repository 的情境线索，以文本形式提供 | §3.3.4 |
| \(X_{\text{auxiliary}}\) | 匹配簇内模型的加权辅助预测 | §3.4.1 式 (7) |
| \(X_{\text{neighbor}}\) | 近邻序列；正文同时用该符号指 look-back 与 prediction 窗口 | §3.4.1 |
| \(I_{\text{input}}\) | 标准化任务/数据/时间戳/外生/内生输入 | 式 (3)(8) |
| \(X_{\text{raw}}\) | Generator 的原始预测 | 式 (8) |

式 (7)：

\[
\begin{aligned}
c_m &= \arg\min_{c_j\in C} \operatorname{sim}(X_{\text{endo}}, c_j),\\
X_{\text{auxiliary}} &= \sum_{i\in C_m} w_i M_i(X_{\text{endo}}),\\
X_{\text{neighbor}} &= \arg\min_i \|X_{\text{endo}}-H_i\|_2.
\end{aligned}
\]

【论文未明确说明】

- \(\operatorname{sim}\) 与 \(\|\cdot\|_2\) 是否为同一度量（上一行写 sim，下一行写 L2）。
- 权重 \(w_i\) 的计算公式。
- \(X_{\text{neighbor}}\) 是单个最近邻还是一组；look-back 与 pred 共用同一符号，疑为排版问题。
- 「local models associated with the cases」是否允许一簇多模型加权（公式写成簇内求和，暗示可以）。

### 3.3 Reflector Agent

**负责什么？**（§3.4.2）保证可靠性，避免盲目采用单次模型输出。角色分两部分：

1. **Forecasting level**：按任务要求与各类情境因素分析结果；评估预测是否足够可靠、是否有充分证据。
2. **CoT level**：逐步审计推理链，澄清每步证据来源，识别 hallucination、circular reasoning、skipped steps；防止约束违反、推理漂移与逻辑缺口。

**检查什么？** 上两条。论文未给出可执行的数值阈值。

**为什么不能简单地和真实未来值比较？**

【论文未明确说明】论文**没有**写「不能与真实未来比较」。  
它写的评估依据是：任务要求、情境因素、可靠性、证据是否充分、CoT 是否出幻觉/循环/跳步。

> 【分析/解释】测试阶段未来真值不可用于在线修正，否则泄漏；因此论文把反思设计成证据/推理审计，而不是 MSE 对答案。这是推断，不是原文。

**发现问题后如何处理？**（§3.4.2 末）

- 若合理：输出目标变量的 predict window 与相应解释。
- 若不合理：产生 reflective feedback，传给 **Investigator**，迭代更新直到结果合理。

【论文未明确说明】最大迭代次数、失败时是否回退到 auxiliary/base model。

---

## 4. 四类信息源

论文用语：feature set / feature library（消融表用 Feature Library）、knowledge base、contextual repository、case library / case base。Abstract 与 §4.3.1 用词不完全统一，所指为同一套工具集。

### 4.1 Feature Library / Feature Set（§3.3.2 + Appendix B + §4.5.1）

对内生与外生 look-back 做特征抽取。公式 (4) 为对各外生通道与内生施加 \(f_1,\ldots,f_n\)。  
论文写：**目前抽取 20 个特征**，两大类：statistical features 与 time series features。细节在 Appendix B。

Appendix B 实际点名的特征（不是 20 个逐项清单）：

| 组别 | 论文出现的名称 |
|------|----------------|
| Basic statistical | mean, variance, skewness, kurtosis |
| 频域 | spectral entropy |
| tsfeatures / 自相关 | acf1, acf10, differenced variants |
| 复杂度 | entropy |
| 结构 | lumpiness, flat_spots, crossing_points |
| 季节 | seasonal strength |

Case Study §4.5.1 另用实现名：

- **basic mean, basic std, basic count**（agent 最偏好）
- Autocorrelation 与 seasonal indicators：**xacf1, seasonal strength**（中等、稳定；电价 FR/DE 更明显）
- **entropy-based measures**（贡献很小）

【论文未明确说明】20 个特征的完整枚举表；Appendix B 说 mean/variance，Case Study 写 basic std / basic count，未解释 count 如何计入「20」。

> 不要把未出现的特征名补进论文事实。

### 4.2 Knowledge Base（§3.3.3）

**保存什么？** 多层次信息，含：

1. **Conceptual knowledge**：理论、定义、既有知识体系；通常来自领域专家与文献；支持抽象推理与概念分类。
2. **Empirical patterns**：直接从数据集蒸馏的数据驱动知识，捕捉常见趋势、行为与响应。例子：某些内生变量在特定条件下反复出现的倾向，作为 empirical references。

记 \(K\) 为从 knowledge base 取出的 domain knowledge。

**解决什么问题？** 让 LLM 的视角不只停留在原始数值模式（§3.3.3）。消融：去掉后「contextual understanding」受损，ETTm 掉点突出（§4.3.1）。

**如何被使用？** 作为 Generator 公式 (8) 的输入 \(K\)。

> **禁止的解释：** 不要自行把它说成「预测价格上涨/下跌的原因库」。论文没这样写。

【论文未明确说明】知识条目的存储格式、检索算法、是人工撰写还是自动抽取。

### 4.3 Contextual Repository（§3.3.4）

**保存什么外部信息？** 与当前时间窗口相关的外部因素。论文举例：holidays、weather variations、marketing activities。符号 \(E\)，以**文本**作为 supplementary background。

**如何参与预测？** 作为公式 (8) 的 \(E\)，与历史数据一起帮助复杂外部条件下的预测。例子：节假日消费波动、极端天气对风电的影响。

【论文未明确说明】每个数据集实际写入了哪些具体文本、是否含实时新闻。

### 4.4 Case Library（§3.3.5, §3.4.1）

**如何构建？**

1. 在**训练集**上切出全部 look-back 窗口 \(H_i\) 与对应预测窗口 \(P_i\)，作为样本。
2. 候选模型池分三类：statistical / deep learning / foundation models。
3. 每个样本在**全部**候选模型上评估，记录 **best-performing model**。
4. 每个 \(H_i\) 与其最优模型 \(M_i\) 配成一条 case；全部 cases 构成 case library。
5. 为了后续快速生成 auxiliary predictions，对数据做 **K-means**，得到簇中心 \(C=\{c_1,\ldots,c_j\}\)。\(c_j\) 为簇 \(C_j\) 内样本均值。公式 (5)。

**使用什么历史窗口？** 训练集的 look-back \(H_i\)（及对应 \(P_i\)）。  
实验中短序 look-back=168、长序=96（§4.1）。  
【论文未明确说明】建库时滑动步长、是否重叠。

**候选模型如何参与？** 每个训练窗上全池比武，记录该窗最优模型。它们也是 Table 1 的对照方法（用法不同，见第 5 节）。

**如何选择历史最佳模型？** 按该训练窗上谁最好，记下 \(M_i\)。  
【论文未明确说明】「最好」用 MSE 还是 MAE。

**K-means 如何使用？** 对 look-back 窗口聚类；中心为簇内样本均值。测试时 \(c_m=\arg\min \operatorname{sim}(X_{\text{endo}},c_j)\)。  
【论文未明确说明】聚类数 \(k\)、是否 z-score、随机种子。

**当前任务如何检索相似案例？**

- 用当前 \(X_{\text{endo}}\) 找最近簇中心。
- 簇内模型加权得到 \(X_{\text{auxiliary}}\)。
- 另按 L2 检索 neighbor 的 look-back/pred。
- 二者作为 **structured references and weak supervisory guidance**，不是最终答案。
- 再交给 Generator 与 \(F_{\text{selected}},K,E\) 一起推理出 \(X_{\text{raw}}\)。

> **论文事实（必须强调）：** AlphaCast **不是**「找到一个历史最佳模型，然后直接用该模型完成当前预测」。  
> 当前预测是：簇内加权辅助预测 + 近邻轨迹 + 特征/知识/情境 → Generator 推理 → Reflector 审核（可能迭代）。

---

## 5. Auxiliary Prediction 与 Baseline

### 5.1 论文里的两套用法

**Baseline（实验比较）**

- §4.1 / Table 1：十四（正文计数）个 SOTA 模型，用于与 AlphaCast **比 MSE/MAE**。
- 跑在 NVIDIA A100 上（§4.1 Implementation Details）。
- 短序 168→24，长序 96→96。

**Auxiliary Prediction（为 Generator 提供预测证据）**

- §3.1：auxiliary predictors \(A\)「offer baseline forecasts or serve as references for pattern retrieval」。
- §3.4.1：\(X_{\text{auxiliary}}\) 是匹配簇内模型对当前 \(X_{\text{endo}}\) 的加权预测；与 \(X_{\text{neighbor}}\) 一起提供弱监督参考。

| | Baseline | Auxiliary Prediction |
|--|----------|---------------------|
| 论文角色 | Table 1 对照行 | Generator 的输入证据之一 |
| 是否经过 LLM | 否（各方法单独预测） | 否（它本身是小模型输出）；随后被 Generator 使用 |
| 是否等于 AlphaCast 最终输出 | 否 | 否 |

【论文未明确说明】Table 1 某行（如 DLinear）与某次 auxiliary 所用权重是否为同一份 checkpoint。按行文两者共用同一候选池，但是否同一权重文件未写。

---

## 6. 实验协议

### 6.1 数据集

**短期（EPF，1 Hour）** 附录 A.1 + Table 5：

| 论文列名 | 内生 | 外生（#Num=2） | 划分 (Train, Val, Test) |
|----------|------|----------------|-------------------------|
| BE | Belgium electricity price | Generation, System Load | (10224, 1584, 3024) |
| DE | German electricity price | Wind power, Amprion zonal load | 同上 |
| FR | France electricity price | Generation, System Load | 同上 |
| NP | Nord Pool electricity price | Grid Load, Wind Power | 同上 |
| PJM | PJM / COMED zonal price | System Load, COMED load | 同上 |

Table 5 PJM 外生原文：`System Load, SyZonal COMED load`（拼写即如此）。

**长期：**

| 论文列名 | 附录名称 | 采样 | 内生 | 外生 | 划分 |
|----------|----------|------|------|------|------|
| ETTh | ETTh1 | 1 Hour | Oil Temperature | 6 个 Power Load Feature | (8544, 1344, 2544) |
| ETTm | ETTm1 | 15 Minutes | Oil Temperature | 6 个 Power Load Feature | (16896, 2496, 4896) |
| Windy Power | Windy Power（讯飞竞赛风电场实发功率） | 15 Minutes | Power Generation | 6 类气象 | (16896, 2496, 4896) |
| Sunny Power | 附录写作 **Sundy Power**（另一座风电场，气象字段相同） | 15 Minutes | Power Generation | 同上 | Table 5 行名 `Sunny power`，划分同上 |
| MOPEX | 水文日序列：流量 + MAP, CPE, Tmax, Tmin | 1 Day | streamflow discharge | 4 类气象 | **Table 5 无 MOPEX 行** |

Windy/Sundy 气象六项（A.1）：Direct Radiation, Wind Direction at 80m, Wind Speed at 80m, Temperature at 2m, Relative Humidity at 2m, Precipitation。

【论文内部不一致】主表/§4 用 Sunny Power；附录 A.1 用 Sundy Power。

【论文未明确说明】

- MOPEX 的 (Train, Val, Test) 长度；Table 6 标题为 “Overview of Datasets”，PDF 抽取无数据行。
- MOPEX 流域/测站编号。
- Windy/Sundy 的具体场站与竞赛届次。

### 6.2 窗长、指标、硬件、LLM

| 项 | 论文规定 | 来源 |
|----|----------|------|
| 短序 | input 168，horizon 24（N-BEATSx 标准协议） | §4.1 |
| 长序 | look-back 96，prediction 96 | §4.1 |
| 指标 | MSE、MAE，越小越好 | Table 1 题注 |
| LLM | **GPT-5** | §4.1 Implementation Details：「We use GPT-5 for the artificial intelligence model.」 |
| 统计实现 | conventional methods | §4.1 |
| DL | PyTorch + Adam | §4.1 |
| 硬件 | 全部 baseline 在 NVIDIA **A100** 上跑 | §4.1 |

【论文未明确说明】GPT-5 的 API 参数、温度、max tokens、是否对三个 agent 用同一模型；DL 的 epoch、学习率、种子、官方 `.pth` 路径。

### 6.3 Baselines（只记论文用过的）

§4.1 写「fourteen」个 SOTA，分类如下。

**Statistical Models（§4.1 列举）：**

- Prophet
- Seasonal Naive (SNaive)
- ARIMA
- CES
- CrostonClassic
- Optimizers（附录 A.2 对应 DynamicOptimizedTheta）
- HistoricAverage

**Deep Learning Models（§4.1 列举）：**

- DLinear
- PatchTST
- TimesNet
- iTransformer
- Autoformer

**Foundation Models（§4.1 列举）：**

- Sundial
- Chronos

**Table 1 额外出现、§4.1 十四列表未点名：**

- **TimeXer**
- 列名 **Transformer**（§4.1 正文写 iTransformer；附录 A.2 有 iTransformer）

【论文内部不一致】

- 正文称 fourteen，Table 1 在 AlphaCast 之外还有 TimeXer + Transformer 等行，行数多于十四。
- 附录 A.2 还介绍了 **Holter-Winter** 与 **TimesFM**，二者**未**出现在 Table 1。
- 不要把 Holter-Winter / TimesFM 算进 Table 1 主比较集。

---

## 7. Table 1 主结果（完整抄录）

来源：Table 1。单位：各数据集原始尺度上的 MSE / MAE。

### 7.1 短序

| Method | BE MSE | BE MAE | DE MSE | DE MAE | FR MSE | FR MAE | NP MSE | NP MAE | PJM MSE | PJM MAE |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|--------:|
| AlphaCast | 536.454 | 9.612 | 193.829 | 9.925 | 721.023 | 7.133 | 27.113 | 3.145 | 25.151 | 3.596 |
| Sundial | 651.237 | 10.997 | 264.618 | 10.707 | 942.081 | 8.536 | 28.667 | 3.362 | 30.704 | 3.987 |
| Chronos | 625.634 | 9.623 | 223.153 | 9.999 | 803.076 | 7.536 | 22.180 | 3.176 | 25.695 | 3.630 |
| DLinear | 658.530 | 12.833 | 239.928 | 12.833 | 811.453 | 10.449 | 32.215 | 3.842 | 42.154 | 4.645 |
| PatchTST | 627.149 | 11.435 | 208.888 | 10.042 | 797.263 | 9.126 | 24.634 | 3.266 | 31.874 | 4.166 |
| TimesNet | 636.660 | 11.300 | 209.366 | 10.197 | 929.100 | 11.099 | 32.116 | 3.805 | 34.890 | 4.342 |
| TimeXer | 702.862 | 10.700 | 252.001 | 10.259 | 834.693 | 9.377 | 27.306 | 3.436 | 25.876 | 3.651 |
| Transformer | 606.528 | 11.242 | 229.955 | 10.227 | 940.227 | 11.093 | 27.088 | 3.434 | 35.131 | 4.373 |
| Autoformer | 890.843 | 16.619 | 331.306 | 13.329 | 934.364 | 13.362 | 46.097 | 4.856 | 77.570 | 6.426 |
| Prophet | 992.900 | 16.942 | 320.631 | 13.254 | 1035.704 | 14.422 | 57.794 | 4.999 | 52.261 | 5.533 |
| SNaive | 857.119 | 13.704 | 415.723 | 13.362 | 915.798 | 11.155 | 44.893 | 4.102 | 41.387 | 4.648 |
| ARIMA | 869.444 | 17.038 | 372.910 | 11.816 | 963.997 | 13.224 | 55.507 | 4.445 | 33.021 | 4.004 |
| CES | 808.608 | 14.414 | 315.576 | 11.451 | 1084.713 | 12.144 | 40.776 | 3.993 | 38.327 | 4.467 |
| CrostonClassic | 927.984 | 17.597 | 294.443 | 12.983 | 1020.099 | 15.158 | 47.462 | 4.804 | 99.401 | 7.819 |
| Optimizers | 750.780 | 12.839 | 424.497 | 12.532 | 855.155 | 11.204 | 68.113 | 4.782 | 36.066 | 4.304 |
| HistoricAverage | 745.450 | 16.741 | 381.713 | 14.568 | 955.659 | 14.634 | 54.432 | 5.225 | 108.607 | 8.101 |

### 7.2 长序

| Method | ETTh MSE | ETTh MAE | ETTm MSE | ETTm MAE | Windy MSE | Windy MAE | Sunny MSE | Sunny MAE | MOPEX MSE | MOPEX MAE |
|--------|---------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|
| AlphaCast | 7.641 | 2.017 | 2.414 | 1.057 | 1548.825 | 24.540 | 13.294 | 1.843 | 4.771 | 1.353 |
| Sundial | 9.437 | 2.314 | 3.211 | 1.258 | 2167.943 | 31.218 | 100.304 | 6.687 | 5.149 | 1.422 |
| Chronos | 9.397 | 2.250 | 2.749 | 1.180 | 2268.519 | 29.617 | 79.245 | 4.566 | 5.283 | 1.307 |
| DLinear | 8.506 | 2.239 | 2.631 | 1.138 | 1932.200 | 29.325 | 19.058 | 2.735 | 5.256 | 1.333 |
| PatchTST | 8.396 | 2.169 | 2.622 | 1.114 | 2269.641 | 32.462 | 19.435 | 2.782 | 5.358 | 1.439 |
| TimesNet | 7.940 | 2.079 | 2.439 | 1.066 | 1930.876 | 29.398 | 18.426 | 2.781 | 4.955 | 1.350 |
| TimeXer | 8.765 | 2.204 | 2.557 | 1.113 | 2169.818 | 31.914 | 20.114 | 2.915 | 4.825 | 1.277 |
| Transformer | 8.307 | 2.136 | 3.136 | 1.277 | 2063.985 | 29.893 | 17.956 | 2.673 | 4.883 | 1.337 |
| Autoformer | 11.598 | 2.637 | 4.224 | 1.571 | 2567.613 | 37.002 | 39.129 | 4.339 | 6.168 | 1.958 |
| Prophet | 46.697 | 4.592 | 21.183 | 3.113 | 8071.244 | 56.059 | 64.807 | 6.442 | 12.395 | 2.503 |
| SNaive | 10.753 | 2.469 | 2.746 | 1.186 | 1948.056 | 27.876 | 79.611 | 4.558 | 7.180 | 1.619 |
| ARIMA | 10.879 | 2.429 | 2.709 | 1.168 | 2166.644 | 28.767 | 80.478 | 4.598 | 9.190 | 1.852 |
| CES | 24.218 | 3.646 | 4.124 | 1.483 | 9364.980 | 42.040 | 78.713 | 4.555 | 6.399 | 1.470 |
| CrostonClassic | 9.719 | 2.331 | 3.287 | 1.365 | 2206.911 | 32.036 | 84.788 | 8.250 | 5.322 | 1.449 |
| Optimizers | 11.772 | 2.541 | 3.533 | 1.261 | 2277.535 | 29.878 | 78.985 | 4.558 | 8.064 | 1.777 |
| HistoricAverage | 10.309 | 2.571 | 3.298 | 1.306 | 2671.839 | 36.529 | 59.759 | 6.130 | 6.142 | 1.709 |

### 7.3 论文对主结果的文字结论（§4.2）

- Overall：多数数据集上 AlphaCast 最好，相对统计、深度、基础模型有显著优势。
- 短序：BE、FR、DE 的 MSE 改进尤其显著；这三套波动大、更难。
- 长序：ETTh、ETTm 上表现突出；Windy Power、Sunny Power 高噪声/强非线性上也明显更低误差。

**必须保留的反例（Table 1 数字，不是文字）：**

- NP **MSE**：Chronos 22.180 < AlphaCast 27.113。
- NP **MAE**：Chronos 3.176 vs AlphaCast 3.145（AlphaCast 略好）。
- MOPEX **MAE**：TimeXer 1.277、Chronos 1.307，均低于 AlphaCast 1.353。

因此「consistently … across most datasets」≠「每一格都最优」。

---

## 8. Toolset Ablation（Table 2）——已与 PDF 核对

来源：Table 2 + §4.3.1。数字与用户给定表一致。

| Dataset | Metric | Full Model | w/o Feature Library | w/o Knowledge Base | w/o Case Library |
| --- | ---: | ---: | ---: | ---: | ---: |
| BE | MSE | 536.454 | 624.267 | 641.524 | 607.572 |
| BE | MAE | 9.612 | 11.361 | 10.819 | 10.846 |
| DE | MSE | 193.829 | 221.970 | 211.768 | 253.103 |
| DE | MAE | 9.925 | 10.082 | 9.848 | 11.643 |
| Windy Power | MSE | 1548.825 | 2442.103 | 2579.324 | 1670.633 |
| Windy Power | MAE | 24.540 | 31.452 | 32.800 | 27.831 |
| ETTm | MSE | 2.414 | 3.595 | 4.486 | 3.900 |
| ETTm | MAE | 1.057 | 1.302 | 1.568 | 1.476 |

论文解释（§4.3.1）：

- 三组件互补、都不可少。
- 去 Feature Library：各数据集误差明显上升；时序表征对稳定特征空间关键。
- 去 Knowledge Base：严重退化，**尤其 ETTm**；领域外部知识帮助捕捉统计相关之外的真实依赖。
- 去 Case Library：适应性变差；通过类比历史场景做 instance-based reasoning。

注意：DE 的 MAE 上 w/o Knowledge Base = 9.848，略低于 Full Model 9.925。论文仍把 knowledge base 说成显著贡献；该格是表格中的局部反例，正文未单独讨论。

【论文未明确说明】消融时 LLM 是否仍为 GPT-5、随机种子、是否同一测试窗。

---

## 9. Reflection Ablation（§4.3.2, Figure 3）

比较：Full Model vs **LLM reflection ablation**（保留底层模型，去掉 reflection）。

**论文写明下降的数据集：** FR、NP、Sunny Power、ETTm。

论文给出的原因：

- 无法做适当修正，初始预测误差得不到纠正；
- 鲁棒性下降，更易受复杂动态环境干扰；
- 失去对长程依赖与时间上下文的建模，只依赖当前时间步，点与点之间连不上。

因此论文认为 reflection 能更准确捕捉时间依赖、保持上下文一致、增强鲁棒性。

【论文未明确说明】这四个数据集去掉 reflection 后的具体 MSE/MAE（只有 Figure 3，抽取不到可靠数字）。

---

## 10. Two-stage 实验（Table 4, §4.4.2）

| Dataset | Full Model MSE | Full Model MAE | Two-stage MSE | Two-stage MAE |
|---------|---------------:|---------------:|--------------:|--------------:|
| BE | 536.454 | 9.612 | 616.026 | 10.881 |
| PJM | 25.151 | 3.596 | 31.872 | 4.102 |
| Windy Power | 1548.825 | 24.540 | 2164.368 | 29.829 |

与用户给定的三个 Full / Two-stage MSE **完全一致**。

**设定：** Full Model 一次不间断生成整段预测；Two-stage 先输出前半段，暂停，再生成后半段。

### 【论文结论】（§4.4.2）

- 单步 Full Model 更好。
- 分段会打断时间连续性与上下文；时序预测依赖长程依赖；内部预测状态被打断后，生成后半段时可能丢失前半段上下文。
- 更易误差累积：第一段的误差成为第二段的错误基础。
- 因此 **direct, continuous reasoning** 对保持预测完整性是关键。

### 相关但不同的实验：Enhanced Reflection（§4.4.3, Figure 5, Table 未给数字）

- 变体：更长 reflection chain，并用时间戳对齐检索的历史片段做二次修正。
- 结果：单次连续推理更好；在 **PJM** 与 **Sunny** 上 enhanced reflection 误差显著更高。
- 原因：打断连续性；时间戳对齐可能召回外部条件不同的片段，错误锚定季节；迭代反思会放大微小基线误差。
- 论文建议：若使用 reflection，应做轻量、残差域、由相似度与外生信号引导的调整，而不是单纯加长 reflection chain 或只靠时间戳对齐。

### 为什么「更多/更长的推理不一定更好？」

**【论文结论】** 来自两处：  
（1）两段式生成（Table 4）变差；  
（2）加长 reflection + 时间戳二次修正（Figure 5）变差。  

**【我的理解】** 推理长度≠有效证据整合；把地平切开或把反思做成二次锚定，可能破坏时序状态并累积误差。论文没有把这一点推广到「任何更长 CoT 都更差」。

---

## 11. 其他消融（§4.3.3, Figure 4）

对 exogenous variables、timestamps、attribute information（数据集描述与变量细节）做消融。论文：Full Model 始终优于三种去掉其一的模型。

- 去 timestamps：Windy Power、ETTm 上 MSE/MAE 急剧上升。
- 去 exogenous：MSE/MAE 显著上升。
- 去 attribute：尤其影响 Windy Power。

【论文未明确说明】Figure 4 的具体数值。

---

## 12. Backbone 比较（Table 3, §4.4.1）

| Dataset | GPT-5 MSE | GPT-5 MAE | DeepSeek-V3 MSE | DeepSeek-V3 MAE | Gemini-2.5-pro MSE | Gemini-2.5-pro MAE |
|---------|----------:|----------:|----------------:|----------------:|-------------------:|-------------------:|
| DE | 193.829 | 9.925 | 213.856 | 10.295 | 227.809 | 10.691 |
| PJM | 25.151 | 3.596 | 35.136 | 4.293 | 36.435 | 4.420 |
| Windy Power | 1548.825 | 24.540 | 1841.589 | 29.802 | 1930.523 | 30.214 |

论文结论：GPT-5 在三个数据集上 MSE/MAE 都最低；Windy Power 上明显低于 DeepSeek-V3 与 Gemini-2.5-pro。认为 GPT-5 的架构为时序分析提供了更强基础。

注意：GPT-5 三行数字与 Table 1 的 AlphaCast 在 DE / PJM / Windy Power 上**相同**。  
【分析/解释】Table 1 的 AlphaCast 主结果应理解为 **AlphaCast 框架 + GPT-5 backbone**，不是「GPT-5 单独当时序模型」的分数。

**不要直接得出：「AlphaCast 一定优于 GPT-5 本身。」** 论文没有把 GPT-5 当作独立 one-shot 时序模型来对打。

> 【我的问题/讨论】主表相对 Chronos/Sundial/DL 的提升，有多少来自 **框架（选信息 + 案例辅助 + 反思）**，有多少来自 **更强的 LLM backbone（GPT-5）**？Table 3 说明换弱 backbone 会掉点，但没有「去框架、只留 GPT-5 回归」的对照。组会应把这个问题留作开放讨论，而不是论文结论。

---

## 13. Case Study

### 13.1 Feature Usage（§4.5.1, Figure 6）

分析 agent 对不同时间线索的优先级。每个 cell 表示某特征参与预测的相对频率。

论文点名的特征：

- Mean → **basic mean**
- Standard Deviation → **basic std**
- Count → **basic count**
- Autocorrelation → **xacf1**
- Seasonal Indicators → **seasonal strength**
- Entropy → **entropy-based measures**

Investigator/agent 选择情况（论文原意）：

- 所有数据集上都明显偏好 basic mean / std / count，作为稳定的水平与尺度指示。
- 自相关与季节指标中等但稳定，电价（FR、DE）上短期依赖更重要。
- 熵类特征贡献最小；论文认为在结构化、周期性环境里复杂度特征用处有限。
- 总体：主要依赖简单可靠的统计量，并在能增强预测连续性时选择性地加入时间依赖。论文称这与人类专家直觉一致。

【论文未明确说明】Figure 6 的具体频率百分比。

### 13.2 Prediction Case（§4.5.2, Figure 7）

对比 ground truth、base model、proposed agent。数据集：DE、ETTm、Windy Power、MOPEX。

- AlphaCast 轨迹更平滑、更贴近真值。
- DE、ETTm：能抓住周期峰谷。
- Windy、MOPEX：适应局部波动同时抑制噪声。
- base model 常滞后或振幅失真。

【论文内部不一致】Figure 7 图内标签抽取为 `CastAgent`，题注为 AlphaCast。

【论文未明确说明】此处 base model 具体是哪一个 Table 1 方法。

---

## 14. 结论与未来工作（§5）

- 把时序预测重定义为交互式认知过程。
- 两阶段：automated preparation、generative reasoning、reflective assessment（正文把三个动作放进「two stages」）。
- 实验表明相对现有 baseline 提升精度。
- Future work：增强 memory integration、更广领域泛化、扩大 autonomous tool use。

---

## 15. 论文未说明 / 内部不一致（汇报时不要假装知道）

### 15.1 【论文未明确说明】

- K-means 的 \(k\)、距离是否标准化、\(w_i\) 公式。
- 建 case 时的误差指标与滑动步长。
- 20 个特征的完整名单。
- Knowledge Base / Contextual Repository 的具体条目与检索实现。
- GPT-5 采样参数；三个 agent 是否同一模型。
- Reflector 能否看见未来真值；迭代上限。
- MOPEX 划分长度与测站 ID。
- DL 训练配方与官方权重。
- Figure 3/4/5 的精确数值。
- Chronos / Sundial 的具体 checkpoint 名（正文只给模型名）。

### 15.2 【论文内部不一致】

- Sunny Power vs Sundy Power。
- Table 1 有 TimeXer，§4.1「fourteen」列举未含 TimeXer。
- Table 1 列名 Transformer vs 正文 iTransformer。
- 附录 A.2 含 Holter-Winter、TimesFM，主表没有。
- 问题定义 \(H/L\) 与实验 168/24、96/96 字母冲突。
- 式 (7) 中 \(X_{\text{neighbor}}\) 同时指 look-back 与 pred。
- Figure 7 标签 CastAgent vs AlphaCast。
- Table 5 无 MOPEX，但 Table 1 有 MOPEX 列。
