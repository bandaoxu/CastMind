# AlphaCast 组会 PPT 内容大纲（16 页）

> 目标：论文精读 · 方法解析 · 实验分析 · 复现与思考  
> 时长：20–30 分钟  
> **本文件只设计内容结构，不生成 PowerPoint。**  
> 数字纪律：论文数字只来自 `analysis/paper_facts.md`；复现数字只来自 `analysis/reproduction_facts.md`。二者禁止混在同一根柱子上。

## 全局版式（制作时遵守）

- 每页一个核心观点。
- 方法页：图 > 字；实验页：图 > 表 > 字；复现页：Paper / My Reproduction 对照。
- 中文为主，专有名词保留英文。
- 不要截论文图；图表按本大纲重画。
- 页脚建议：左「论文 / 本机复现」标签；右页码。

---

## 第 1 页｜封面

**页码：** 1 / 16

**标题：** AlphaCast：人类智慧与 LLM 协同推理的交互式时间序列预测框架

**副标题：** 论文精读 · 方法解析 · 实验分析 · 复现与思考

**核心观点：** 这不是论文摘要，而是把方法讲清楚，再用实验和复现边界收住结论。

**页面文字（尽量少）：**

- 论文：AlphaCast: A Human Wisdom–LLM Intelligence Co-Reasoning Framework for Interactive Time Series Forecasting
- 作者：Zhang, Gao, Cheng 等 · 中国科学技术大学认知智能全国重点实验室
- arXiv: 2511.08947
- 开源实现：CastMind
- 汇报人 / 日期：【制作时填写】

**图示设计：**

- 全页留白，标题居中。
- 底部一条细流程：`One-shot → Interactive Co-Reasoning`（仅作视觉钩子，不展开）。
- 不要放实验数字。

**图表设计：** 无。

**数据来源：** 论文题名页；仓库 README。

**Speaker Notes（约 150 字）：**  
今天组会精读 AlphaCast。它不是再提一个预测网络，而是把预测改成：先准备证据、再生成、再反思。我会按方法、实验、我的复现、边界和开放问题这条线讲。主表数字全部来自论文；复现数字会单独标注，不和论文混在一张图里。有夸大的地方我会主动说「论文没写」或「我还没对齐」。

**Takeaway：** 今天听三件事——交互式预测是什么、实验是否撑得住、我复现到哪一步。

---

## 第 2 页｜研究背景：时间序列预测的问题在哪里？

**页码：** 2 / 16

**标题：** 研究背景：时间序列预测的问题在哪里？

**核心观点：** 传统预测是一次性映射；缺的不只是更大的模型，而是选信息、用工具、能改错的流程。

**页面文字：**

- 场景：能源调度、健康预警、气候预估（论文 §1）。
- 路线已经很多：统计 → 深度学习 → 基础模型。
- 论文认为多数仍是 **static one-shot mapping**。

传统（画面主图）：

```text
历史数据  →  预测模型  →  未来预测
```

右侧三句短标签（不要写成段落）：

- 一次性
- 信息来源有限
- 缺少动态选择与反馈

**图示设计：**

- 左：一条直线流水（灰）。
- 右：三个灰色缺口图标（交互 / 外部信息 / 反思），先点出缺口，第 3 页再展开。

**图表设计：** 无数值图。

**数据来源：** 论文 Abstract、§1、§2.1、Figure 1。结构图为【基于论文方法的概括】。

**Speaker Notes（约 160 字）：**  
统计、深度、Sundial/Chronos 这类基础模型都在变强，但论文的批评很集中：它们大多还是「看过去、一次映射未来」。人类预报员会看特征、查知识、翻类似历史、再用工具试，不行就改。所以 AlphaCast 的起点不是「再拟合准一点」，而是：一次性流程在复杂环境里不够用。这句话是后面所有设计的理由。

**Takeaway：** 问题被重定义为流程太短，而不只是模型不够大。

---

## 第 3 页｜为什么“一次性预测”不够？

**页码：** 3 / 16

**标题：** 为什么“一次性预测”不够？

**核心观点：** 复杂现实预测需要外部信息、领域知识、历史案例、其他模型的预测，以及反馈修正。

**页面文字：**

左栏标题：**Traditional One-shot Forecasting**

- 输入：\(X_{\text{past}}\)
- 学习：\(f: X_{\text{past}} \rightarrow \hat{X}_{\text{future}}\)（论文公式 1）
- 输出：一次前向，结束

右栏标题：**复杂现实预测需要什么**

- 外部信息（节假日、天气、负荷）
- 领域知识
- 历史案例
- 其他模型的预测（辅助，不是最终答案）
- 反馈修正

底部一句：

> 论文把这些写成 \((X_{\text{past}}, C, K, A) \xrightarrow{\text{Co-Reasoning}} \hat{X}_{\text{future}}\)（公式 2）

**图示设计：**

- 左右对照卡片。左栏一条直线；右栏五枚材料芯片汇入「共推理」圆。
- 不要画成「人在环路里逐步改数」——论文写的是 human wisdom 与 LLM 协同，未明确人在线改数。【论文未明确说明】不要画成真人值班台。

**图表设计：** 无。

**数据来源：** 论文 §3.1、§3.3.4。

**Speaker Notes（约 155 字）：**  
一次性预测不是「错」，而是信息不够。电价会碰到节假日和负荷冲击，风电会碰到天气，水文还有气象驱动。论文认为这些不该只当额外特征塞进同一个回归器，而应进入推理过程：选哪些信息、参考哪类历史、要不要改辅助预测。右栏五项就是后面 Feature、Knowledge、Context、Case、Auxiliary 的伏笔。

**Takeaway：** 交互式预测 = 把情境、知识、案例和修正写进流程，而不是再加一层网络。

---

## 第 4 页｜AlphaCast 的核心思想

**页码：** 4 / 16

**标题：** AlphaCast 的核心思想

**核心观点：** One-shot Forecasting → Interactive Co-Reasoning。LLM 不是又一个时序模型，而是协同推理的主体。

**页面文字：**

中心大字：

> **One-shot Forecasting → Interactive Co-Reasoning**

核心循环（主视觉）：

```text
Investigation
      ↓
Generation
      ↓
Reflection
      ↓
Iteration
```

底部两句（小字）：

- 论文原阶段名：prepare → generate → verify（Abstract / §1）。
- 上图四步是【基于论文方法的概括】，便于口播，不是论文原标题。

**图示设计：**

- 中央竖向循环，最后 Iteration 用虚线回到 Investigation。
- 左侧小标签「人智启发的证据」；右侧小标签「LLM 推理」。
- 不要出现 GPT-5 分数。

**图表设计：** 无。

**数据来源：** 论文 Abstract、§1、§3.2。四步口号见 `paper_facts.md` §2.2。

**Speaker Notes（约 165 字）：**  
请记住这页的转向：不是把 GPT-5 当成 Chronos 的替代品去直接吐曲线。论文的主张是协同推理——先调查该看什么，再生成预测和思维链，再检查是否站得住，不行就迭代。后面架构、三个 Agent、案例库，都是在落实这个循环。如果有人问「和 Time-LLM 有何不同」，先答：这里强调的是流程，不是把序列重编程进语言模型。

**Takeaway：** AlphaCast 卖的是交互式共推理流程，不是「更大的 LLM 回归器」。

---

## 第 5 页｜AlphaCast 整体架构

**页码：** 5 / 16

**标题：** AlphaCast 整体架构

**核心观点：** 多源材料进入 Investigator，Generator 给出原始预测与 CoT，Reflector 决定接受或退回重查。

**页面文字：** 尽量不写段落，只留架构图标签。

**图示设计（必须按此画，作为本页唯一主视觉）：**

```text
Time Series
Features
Knowledge
Context
Cases
Auxiliary Predictions
        ↓
   Investigator
        ↓
    Generator
        ↓
   Raw Forecast  (+ CoT)
        ↓
    Reflector
        ↓
      合理？
   ↙          ↘
  否            是
  ↓             ↓
重新调查     Final Forecast
```

补充标注（小字，贴在对应节点）：

- Investigator：选 \(F_{\text{selected}}\)（论文 §3.4.1）
- Generator：公式 (8) 含 \(I_{\text{input}}, F_{\text{selected}}, K, E, X_{\text{auxiliary}}, X_{\text{neighbor}}\)
- Reflector：预测层 + CoT 层（§3.4.2）
- 「否」箭头回到 Investigator（论文：feedback 传给 investigator）

**图表设计：** 无数值。可用极浅色区分「准备材料」与「推理闭环」两块。

**数据来源：** 论文 Figure 2、§3.2、§3.4。本图为结构转写，不是论文原图描摹。

**Speaker Notes（约 170 字）：**  
这页只讲走线图。上面一排不是五个模型，是五类材料。Investigator 决定看哪些；Generator 综合后给出原始预测和思维链；Reflector 不拿未来真值对答案，而是问：够不够可靠、证据够不够、推理有没有跳步。不合理就退回调查，不是把错误曲线直接交差。案例库和辅助预测在这里只是输入，下一页才拆开。

**Takeaway：** 最终预测在闭环之后，不在某一个基线模型的输出上。

---

## 第 6 页｜四类信息源：LLM 到底在“看什么”？

**页码：** 6 / 16

**标题：** 四类信息源：LLM 到底在“看什么”？

**核心观点：** LLM 看到的是特征、知识、情境、案例，外加 Auxiliary Predictions；这些是证据，不是最终答案。

**页面文字：**

四个模块（2×2）：

| 模块 | 一句话 | 不要说成 |
|------|--------|----------|
| Feature Library | 窗口长什么样：mean / std / count、自相关、季节强度、熵等 | 「20 个特征的完整名单」——论文未逐项列表 |
| Knowledge Base | 概念知识 + 从数据蒸馏的经验模式 | 「涨跌原因库」 |
| Contextual Repository | 节假日、天气、营销等文本情境 \(E\) | 实时新闻流（论文未写） |
| Case Library | 相似历史窗 + 当时优胜策略 | 直接投票出最终模型 |

下方单独一条（与四类并列强调）：

> **Auxiliary Predictions**：簇内模型加权得到的参考曲线 \(X_{\text{auxiliary}}\)，是 Generator 的证据，不是 Table 1 对照行。

底部对照（一行）：

- **Baseline** = 实验比较（Table 1）
- **Auxiliary Prediction** = 给 Generator 的预测证据

**图示设计：**

- 四张卡片 + 底部一条 Auxiliary 横条，箭头指向「Generator 的工作台」。
- Feature 卡片可点名：Mean, Std, Count, Autocorrelation, Seasonal Strength, Entropy（Case Study §4.5.1）。

**图表设计：** 无。

**数据来源：** 论文 §3.3、§3.4.1、§4.5.1、Appendix B；Auxiliary vs Baseline 见 `paper_facts.md` §5。

**Speaker Notes（约 170 字）：**  
很多人会把 Knowledge Base 理解成「解释电价为什么涨」。论文不是这样写的：它是概念知识和数据里的经验模式。Contextual Repository 是窗口相关的外部文本。Case Library 提供相似历史。Auxiliary 是参考预测，和 Table 1 的 Baseline 用法不同：一个进推理，一个拿来比分。四类去掉任何一个，后面消融会显示误差上升，所以不是装饰模块。

**Takeaway：** LLM 看的是证据包；Auxiliary 是参考，不是最终预测。

---

## 第 7 页｜三个 Agent 如何协同？

**页码：** 7 / 16

**标题：** 三个 Agent 如何协同？

**核心观点：** Investigator 负责看什么，Generator 负责怎么预测，Reflector 负责是否合理；合起来才是 Investigation → Generation → Reflection → Iteration。

**页面文字（三栏，每栏不超过四行）：**

**Investigator｜看什么？**

- 解析任务，分解需要哪些特征与背景
- 抽取 \(F_{\text{selected}}=S(F,I_{\text{input}})\)
- 论文把案例检索写在 Generator；本机开源把检索打进 Investigator packet（复现差异留到第 14 页）

**Generator｜怎么预测？**

- 检索相似簇，得到 \(X_{\text{auxiliary}}\) 与近邻
- \(X_{\text{raw}} \leftarrow\) 推理\((I_{\text{input}}, F_{\text{selected}}, K, E, X_{\text{auxiliary}}, X_{\text{neighbor}})\)
- 同时输出 CoT

**Reflector｜合理吗？**

- 预测层：是否可靠、证据是否充分
- CoT 层：幻觉、循环论证、跳步
- 否 → 反馈 Investigator；是 → 输出最终预测

底部横条：

> Investigation → Generation → Reflection → Iteration

**图示设计：**

- 三等分竖栏 + 底部闭环箭头。
- Reflector「否」用虚线回到第一栏。

**图表设计：** 无。

**数据来源：** 论文 §3.4.1–3.4.2。公式 (8) 含 \(I_{\text{input}}\)，不要写成论文原式 `X_raw = Generator(...)`。

**Speaker Notes（约 165 字）：**  
三个 Agent 不是三个独立预报员投票。调查决定材料，生成给出曲线和理由，反思检查理由能不能撑住曲线。论文没有写可以用未来真值来打分，所以反思是证据审计。若被问「为什么不直接和明天的电价比」，回答：那是测试泄漏；论文设计的是过程检查。本机复现页再说明：默认 llm 下三 Agent 都是 DeepSeek；这里先按论文讲闭环。

**Takeaway：** 协同的关键是闭环，而不是三个模型平均。

---

## 第 8 页｜Case Library：让模型参考“过去怎么解决类似问题”

**页码：** 8 / 16

**标题：** Case Library：让模型参考“过去怎么解决类似问题”

**核心观点：** Case Library 是历史经验与弱监督参考，**不是**直接决定当前最终模型。

**页面文字：** 以流程图为主，底部一句加粗。

**图示设计（主视觉，按此画）：**

```text
历史数据
      ↓
滑动窗口（look-back + pred）
      ↓
候选模型预测（统计 / 深度 / 基础模型）
      ↓
历史案例评估（记下该窗最优模型）
      ↓
K-means 聚类
      ↓
Case Library
      ↓
当前任务：相似案例检索
      ↓
辅助预测 + 近邻轨迹
      ↓
Generator（可改、可几乎不改）
```

必须强调（底部横幅，不要缩小成脚注）：

> **Case Library 是历史经验，而不是直接决定最终模型。**

可加一句小字：测试时是簇内加权 \(X_{\text{auxiliary}}=\sum w_i M_i(X_{\text{endo}})\)，再进入推理；【论文未明确说明】\(w_i\) 与 \(k\)。

**图表设计：** 无数值。可用「错误示范」小戳：`argmax 历史冠军 → 直接当最终预测` 打叉。

**数据来源：** 论文 §3.3.5、§3.4.1 公式 (5)(7)。

**Speaker Notes（约 175 字）：**  
这页请讲慢。训练时每个窗口让候选池比武，记下谁赢，再把相似窗口聚成类。测试时找到最像的类，用类里的模型做加权参考，再加上近邻轨迹。然后交给 Generator。所以 AlphaCast 不是「历史上 ARIMA 常赢，今天就永远用 ARIMA」。辅助预测可以被改，也可以几乎原样采用，但最终分数是系统行，不是某一个基线行。消融去掉 Case Library 后误差上升，说明这套经验有用，但仍是材料。

**Takeaway：** 案例库提供「过去类似问题怎么解」，不代替当前推理。

---

## 第 9 页｜实验设置：AlphaCast 与谁比较？

**页码：** 9 / 16

**标题：** 实验设置：AlphaCast 与谁比较？

**核心观点：** 先对齐协议，再谈谁更好；主结果 backbone 是 GPT-5。

**页面文字：**

**短期 EPF（168 → 24）：** BE / DE / FR / NP / PJM

**长期（96 → 96）：** ETTh / ETTm / Windy Power / Sunny Power / MOPEX

**Baseline 三类（只列论文用过的）：**

- Statistical：Prophet, SNaive, ARIMA, CES, CrostonClassic, Optimizers, HistoricAverage
- Deep Learning：DLinear, PatchTST, TimesNet, iTransformer, Autoformer（Table 1 另有 TimeXer、列名 Transformer）
- Foundation：Sundial, Chronos

**协议条：**

- 指标：MSE / MAE（越小越好）
- Backbone：**GPT-5**
- 硬件：baseline 在 NVIDIA A100（论文 §4.1）

**图示设计：**

- 左：短/长两列数据集芯片。
- 右：三层金字塔（统计 / 深度 / 基础模型），顶上单独一枚「AlphaCast = 框架 + GPT-5」。
- 不要把 TimesFM、Holt-Winters 画进主比较集（附录有、主表无）。

**图表设计：** 无成绩柱状图（留给第 10 页）。

**数据来源：** 论文 §4.1、附录 A.1、Table 5。Sunny / Sundy 命名不一致，口播用主表 Sunny Power。

**Speaker Notes（约 150 字）：**  
短序是五个电价市场，长序是变压器油温、风电、光伏和水文。比的是同一协议下的方法，不是两个 GitHub 仓库对打。AlphaCast 这一行是框架加上 GPT-5，不是「GPT-5 单独当时序模型」的分数。A100 和官方权重是作者实验条件；我后面复现对齐不了这两项，会单独说。

**Takeaway：** 比较对象是协议下的方法行；AlphaCast 行 = 框架 + GPT-5。

---

## 第 10 页｜实验结果：AlphaCast 真的有效吗？

**页码：** 10 / 16

**标题：** 实验结果：AlphaCast 真的有效吗？

**核心观点：** 多数数据集上 AlphaCast 最优或接近最优，但不是每一格都赢；不要把「most datasets」说成「全部第一」。

**页面文字（极少）：**

- 论文结论：多数数据集最优（§4.2）。
- 必须同时露出反例：NP 的 MSE 上 Chronos 更好。

**图示设计：** 无论文截图。自绘两块。

**图表设计：**

**图 A｜长序 ETTh，MSE（越小越好）** 分组柱或并排柱：

| 方法 | MSE |
|------|----:|
| AlphaCast | **7.641** |
| TimesNet | 7.940 |
| Transformer | 8.307 |
| PatchTST | 8.396 |
| DLinear | 8.506 |
| Chronos | 9.397 |
| Sundial | 9.437 |

配一句：长序 ETTh 上相对两个基础模型大约从 9.4 降到 7.641。

**图 B｜短序高波动三市场，AlphaCast vs 次优量级（MSE）**

| 数据集 | AlphaCast | 同表参考 |
|--------|----------:|----------|
| BE | 536.454 | Sundial 651.237；Chronos 625.634 |
| DE | 193.829 | PatchTST 208.888 |
| FR | 721.023 | PatchTST 797.263 |

**小表或角标（必须出现，避免夸大）：**

- NP MSE：Chronos **22.180** < AlphaCast 27.113
- MOPEX MAE：TimeXer 1.277、Chronos 1.307 < AlphaCast 1.353

**数据来源：** 论文 Table 1（`paper_facts.md` §7）。**禁止**把本机 7.651 画进这页。

**Speaker Notes（约 175 字）：**  
先看 ETTh：AlphaCast 7.641，Sundial 和 Chronos 大约 9.4。短序 BE、DE、FR 波动大，论文强调这里 MSE 改进更明显。但请看角标：北欧电价 NP 的 MSE，Chronos 比 AlphaCast 低；MOPEX 的 MAE 也不是 AlphaCast 最低。所以严谨说法是「多数数据集最好」，不是「全面碾压」。这页全部是论文数字，我的复现还没出场。

**Takeaway：** 主结果支持框架有效，但存在明确反例，汇报时要自己点出来。

---

## 第 11 页｜消融实验：去掉信息源会怎样？

**页码：** 11 / 16

**标题：** 消融实验：去掉信息源会怎样？

**核心观点：** Feature / Knowledge / Case 三类信息源具有互补作用；不是「只换了一个更大的语言模型」。

**页面文字：**

结论句：

> 三类信息源具有互补作用。

点名：去 Knowledge Base 后 **ETTm** 掉点突出；去 Feature 后全面变差；去 Case 后适应性变差（论文 §4.3.1）。

诚实脚注：DE 的 MAE 上 w/o Knowledge Base = 9.848，略低于 Full 9.925，正文未单独讨论。

**图示设计：** 四组「Full vs 三个去掉」的柱簇，数据集用 BE / DE / Windy / ETTm。

**图表设计（MSE，论文 Table 2）：**

| Dataset | Full | w/o Feature | w/o Knowledge | w/o Case |
|---------|-----:|------------:|--------------:|---------:|
| BE | 536.454 | 624.267 | 641.524 | 607.572 |
| DE | 193.829 | 221.970 | 211.768 | 253.103 |
| Windy Power | 1548.825 | 2442.103 | 2579.324 | 1670.633 |
| ETTm | 2.414 | 3.595 | 4.486 | 3.900 |

柱状图：每数据集四根柱，Full 用强调色，其余灰色。Windy 与 ETTm 量级差两个数量级，**不要画在同一 y 轴**；可分两个 panel，或对每个数据集做「相对 Full 的升幅」。

相对升幅（制作柱图可用，口播不必全读）：

- ETTm MSE：Feature +49%；Knowledge +86%；Case +62%

**数据来源：** 论文 Table 2。数字已与 PDF 核对。

**Speaker Notes（约 160 字）：**  
如果有人怀疑「不就是 GPT-5 强吗」，请看这页：同一个框架下拿掉特征库、知识库或案例库，误差都会升。Windy 去掉知识库从 1549 升到 2579，ETTm 去掉知识库从 2.414 升到 4.486。这说明准备阶段的证据模块对精度有实质贡献。DE 的 MAE 有一个小反例，我不回避，但 MSE 和其他数据集仍支持互补结论。

**Takeaway：** 消融支持「证据模块有用」，不能把主表优势全部归因于 GPT-5，但也不能反向说与 GPT-5 无关。

---

## 第 12 页｜Reflection 真的有用吗？

**页码：** 12 / 16

**标题：** Reflection 真的有用吗？

**核心观点：** 去掉 Reflection 后，论文在部分数据集上观察到下降；因此 Reflection 有用，但「更多 Reflection」是否更好，留给下一页。

**页面文字：**

左：**With Reflection** = Full Model  
右：**Without Reflection** = LLM reflection ablation（保留底层，去掉反思机制）

论文写明下降的数据集（无可用精确 MSE，不要编数字）：

- FR
- NP
- Sunny Power
- ETTm

原因（三枚短标签）：初始误差得不到纠正；鲁棒性下降；难以建立时间点之间的联系。

底部引出（箭头指向第 13 页）：

> 但更多 Reflection 是否一定更好？

**图示设计：**

- 左右对照。因 Figure 3 抽不出可靠数值，**不要伪造柱高**。
- 可用四个数据集芯片，从「Full」到「w/o Reflection」画向下箭头，旁注「论文：性能下降；具体数值见 Figure 3，本页不抄不可核的刻度」。

**图表设计：** 定性示意图，禁止捏造 MSE。

**数据来源：** 论文 §4.3.2、Figure 3。`paper_facts.md`：未给出具体 MSE/MAE。

**Speaker Notes（约 155 字）：**  
Reflection 的消融没有公开精确表格，只有图。我只敢说：在 FR、NP、Sunny、ETTm 上，去掉反思会变差。这支持「要有反思」，还不能推出「反思越长越好」。下一页的两段式生成，以及论文里加长反思链的实验，会给出反直觉的结果。所以第 12 页和第 13 页要连着听：有用，但不等于越多越好。

**Takeaway：** Reflection 在部分数据上有用；下一问是「更长的推理还是否有用」。

---

## 第 13 页｜一个反直觉结论：更多推理 ≠ 更好预测

**页码：** 13 / 16

**标题：** 一个反直觉结论：更多推理 ≠ 更好预测

**核心观点：** Reasoning is useful, but more reasoning is not always better.

**页面文字：**

设定（一行）：Full Model = 一次生成整段；Two-stage = 先生成前半段，再生成后半段。

三组数据（MSE，论文 Table 4）：

- BE：536.454 → **616.026**
- PJM：25.151 → **31.872**
- Windy Power：1548.825 → **2164.368**

英文结论：

> **Reasoning is useful, but more reasoning is not always better.**

中文解释：

> 过度延长推理过程可能破坏时间连续性和上下文一致性，并造成误差累积。

【论文结论】来自 Table 4 / §4.4.2。  
【论文另证】加长 reflection chain + 时间戳二次修正，在 PJM、Sunny 上更差（§4.4.3，无表内数字）。  
不要把两处实验画成同一根柱。

**图示设计：**

- 三组「Full（实心）vs Two-stage（浅色）」水平条形，箭头表示变差。
- 背景不要写「LLM 越想越差」这种过度概括。

**图表设计：**

| Dataset | Full MSE | Two-stage MSE |
|---------|---------:|--------------:|
| BE | 536.454 | 616.026 |
| PJM | 25.151 | 31.872 |
| Windy Power | 1548.825 | 2164.368 |

**数据来源：** 论文 Table 4、§4.4.2。解释句对应论文「disruption of temporal continuity and context / error accumulation」。

**Speaker Notes（约 170 字）：**  
这是我觉得最值得组会讨论的实验。反思本身有用，但把预测地平切开、分两段生成，三个数据集误差都升。论文的解释是：时序依赖被打断，前半段的错会变成后半段的错基础。加长反思链、用时间戳去对齐历史再改一次，同样会变差。所以我的读法是：有效推理比更长推理重要。这是论文结论，不是我发明的；我只是把它收成一句口播。

**Takeaway：** 要推理，但不要为了「看起来想得更久」而切开时间或拉长反思链。

---

## 第 14 页｜我的复现：从论文到代码

**页码：** 14 / 16

**标题：** 我的复现：从论文到代码

**核心观点：** 协议和机制可以对齐；骨干模型和权重不能对齐。本页数字是**本机复现**，不是论文 Table 1。

**页面文字 / 图示设计：** 三列对照，禁止把左右数字画在同一坐标轴。

```text
Paper                          My Reproduction
GPT-5                          DeepSeek  deepseek-chat
K-means                        K-means（写死，k 默认 6）
Case Library                    Case Library（训练窗比武 + 簇加权）
Reflector（LLM 审计）           Reflector = DeepSeek LLM（默认可切 rules）
Chronos                         Chronos bolt-base
Sundial                         Sundial 本地 .venv + transformers 4.40.1
Investigator（LLM）             Investigator = DeepSeek LLM + tools
```

**ETTh1 本机 LLM 锚点（CastMind + DeepSeek，禁止当论文主表）：**

```text
K-means  k = 6
Case Library includes Sundial（本跑次获胜 3 windows）
MSE = 7.651
MAE = 2.137
点数 = 2448
chosen_model = LLM
```

必须用标签：**本机 LLM / DeepSeek `deepseek-chat`**，并写「不是论文 AlphaCast 行 7.641 / 2.017」。  
不要把 7.651 画在论文柱状图旁边暗示「已经对齐」或「超过论文」。

**图表设计：** 对照表，不要柱状图混论文/本机。

**数据来源：**

- 对照项：`reproduction_facts.md` §1–3。
- MSE 7.650819 / MAE 2.136725 / 2448 点：`outputs/comparison/ETTh1_castmind_CastMind_llm_deepseek.json`（与 `outputs/ETTh1/predictions.csv` 一致，2026-09-09）。
- k=6：`outputs/ETTh1/cluster_base.json` 共 6 簇；日志 `K-means (k=6)`。
- Sundial 3 windows：同跑次 `outputs/ETTh1/cases_stats.json`（训练窗获胜计数，总和 102）。
- 本页**不使用** deterministic / HoltWinters 的 9.266。

**Speaker Notes（约 175 字）：**  
从这页开始换成「我做了什么」。论文用 GPT-5，我用 DeepSeek；三 Agent（Investigator / Generator / Reflector）默认都是 DeepSeek LLM，Reflector 仅在开环境变量时退回规则模式。Chronos 用 bolt-base，Sundial 钉在 transformers 4.40.1。ETTh1 走完 LLM 闭环后，MSE 7.651、MAE 2.137，2448 个点。数字接近论文 7.641，但骨干、自训权重都不同，不能说复现了主表。案例库 K-means 为 6 类，Sundial 在训练比武里赢了 3 个窗，说明它进了池，不是全局冠军。

**Takeaway：** 本机展示 LLM 系统行 7.651 / 2.137；这是流程级验证，不是 Table 1 复现。

---

## 第 15 页｜复现边界：我验证了什么？还缺什么？

**页码：** 15 / 16

**标题：** 复现边界：我验证了什么？还缺什么？

**核心观点：** 代码/流程级复现验证 ≠ 严格数值复现。

**页面文字：**

**左｜已验证**

- ✓ Agent 流程（调查 → 生成 → 规则反思）
- ✓ Feature / Knowledge（briefing）/ Case 机制
- ✓ K-means Case Library（k=6）
- ✓ Sundial / Chronos 公开权重可推理
- ✓ ETTh1 实际结果（本机 LLM：MSE 7.651 / MAE 2.137，2448 点）

**右｜尚未完全一致**

- × GPT-5（本机 DeepSeek）
- × A100 官方环境与作者 DL `.pth`
- × 部分数据集版本（Windy / Sunny / MOPEX 无作者一致 CSV）
- × 论文级 LLM Reflector
- × 最后测试窗曾出现 48 点（2448 点协议不完整）——若组会要极严谨再口述

**底部横幅：**

> **代码/流程级复现验证 ≠ 严格数值复现**

**图示设计：** 左右两栏，底下一句。对勾绿色、叉号灰色，不要用红叉显得「失败」，这是边界披露。

**图表设计：** 无。不要把 7.651 和 7.641 并排暗示「已经对齐」。

**数据来源：** `reproduction_facts.md` §0、§5、§6、§8。

**Speaker Notes（约 165 字）：**  
这页是防止被问「那你复现成功了吗」时答错。我验证的是流程：三个 Agent、案例库、特征、公开基础模型、ETTh1 能出数。我没有 GPT-5，没有作者权重，风电和 MOPEX 也拿不到同一份数据。所以正确名称是代码和流程级验证。DeepSeek 的 7.651 接近 7.641 看起来很诱人，但骨干、权重、最后一窗长度都不同，我不会说已经复现主表。

**Takeaway：** 能讲机制、能讲本机数字；不能讲「复现了 Table 1」。

---

## 第 16 页｜我的理解：AlphaCast 真正值得关注什么？

**页码：** 16 / 16

**标题：** 我的理解：AlphaCast 真正值得关注什么？

**核心观点：** 值得关注的是证据化流程与「有效推理」，不是「再堆一个大模型」。

**页面文字：**

**① LLM 不是单纯预测器**

> 信息选择 + 证据整合 + 推理 + 反思

**② 多源信息成为预测的一部分**

> Feature + Knowledge + Context + Case + Auxiliary Prediction

**③ 有效推理比更多推理更重要**

开放问题（三问，不要给假答案）：

1. Human Wisdom 如何真正进入预测闭环？（论文未明确「人在线改数」）
2. 性能提升来自 AlphaCast 框架，还是 GPT-5？（Table 3 换 backbone 会掉点，但没有「去框架、只留 GPT-5」对照）【我的问题/讨论】
3. 如何降低 LLM 预测成本与随机性？

**图示设计：**

- 三张大卡片 + 底部三问。
- 结束页不要再堆 Table 1。

**图表设计：** 无。

**数据来源：** 综合 `paper_facts.md` §2、§12、§16；思考题标为个人讨论。

**Speaker Notes（约 170 字）：**  
收尾只留三个判断。第一，LLM 在这里是调度证据的推理者。第二，特征、知识、情境、案例和辅助预测都被写进预测，而不只是预处理。第三，论文自己的两段式实验说明，推理变长不一定变准。我复现只走到流程级，所以最后三问我答不完全：人智如何真正进闭环、主表红利有多少来自 GPT-5、成本如何降。欢迎就这三问讨论。谢谢。

**Takeaway：** 关注流程、证据与有效推理；把框架贡献和 GPT-5 贡献留作开放问题。

---

## 时间分配（20–30 分钟）

| 页 | 建议 | 累计 |
|----|------|------|
| 1 封面 | 0.5 分钟 | 0.5 |
| 2–3 背景与动机 | 4 分钟 | 4.5 |
| 4–5 思想与架构 | 4 分钟 | 8.5 |
| 6–8 信息源 / Agent / Case | 6 分钟 | 14.5 |
| 9–10 设置与主结果 | 4 分钟 | 18.5 |
| 11–13 消融与反直觉 | 5 分钟 | 23.5 |
| 14–15 复现与边界 | 3 分钟 | 26.5 |
| 16 思考 + Q&A | 剩余 | 30 |

若只有 20 分钟：第 10 页只留 ETTh + 一个反例；第 11 页只留 ETTm/Windy；第 14 页只读对照表和本机 LLM 7.651。

---

## 最终检查

| 项 | 状态 |
|----|------|
| 是否 16 页？ | 是，第 1–16 页 |
| 是否中文为主？ | 是；专有名词保留英文 |
| 是否包含复现？ | 第 14–15 页 |
| 是否包含消融？ | 第 11 页 |
| 是否包含 Reflection？ | 第 12 页 |
| 是否包含 Case Library？ | 第 8 页（第 6 页也点名） |
| 是否包含个人评价？ | 第 16 页；第 13 页口播可带个人读法但结论标为论文 |
| 是否把论文数据和复现数据区分？ | 第 10–13 页仅论文；第 14–15 页仅本机并打标签 |
| 是否有 Speaker Notes？ | 每页均有，约 150–175 字 |
| 是否有 Takeaway？ | 每页均有 |
| 是否生成 PPT 文件？ | **否** |

**制作下一阶段注意：**

1. 第 12 页不要发明 Reflection 的 MSE。
2. 第 14 页只用本机 LLM：MSE 7.651 / MAE 2.137；Sundial 获胜窗以 `cases_stats.json` 的 **3** 为准，不用 deterministic 的 7/10 窗旧文档。
3. 第 14 页禁止展示 HoltWinters 9.266；7.651 必须标明 DeepSeek，不得口播成复现 7.641。
4. 不要截论文 Table 1 当第 10 页。
