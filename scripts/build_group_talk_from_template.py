#!/usr/bin/env python3
"""Build a 9-slide paper-complete group talk PPT from the visual template.

Keeps template colors/fonts/layouts; expands pages by duplicating template slides.
Mainline: paper only (no repro / next-plan slides).
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "AlphaCast论文阅读与复现进展汇报.pptx"
OUT = ROOT / "docs" / "组会分享_AlphaCast.pptx"
FIGS = ROOT / "docs" / "talk_figures"

# Final order after duplication (indices into the 9-slide deck before reorder helpers)
# Built as: start from template [0..6], append dup of 1, append dup of 2 -> [0..8]
# Then reorder to paper narrative.

SLIDE_TEXTS: list[list[str]] = [
    # 1 Cover (4)
    [
        "AlphaCast：人智与大模型协同的交互式时序预测",
        "组会论文精读 · 把方法讲清楚",
        "汇报人：____________",
        "日期：2026年9月",
    ],
    # 2 Motivation (8)
    [
        "01 研究背景与动机",
        "图：从一次静态映射，到类人动态迭代预测",
        "01 传统时序预测的局限",
        "统计、深度学习与基础模型虽强，但多数仍是“输入历史→一次输出”。缺少对情境、相似历史与结论合理性的显式反思，复杂动态场景下可信度不足。",
        "02 人类专家如何做预报",
        "专家通常会：看形态与特征 → 查知识与经验 → 参考相似历史 → 校验后再改。预报是多步闭环，而不是黑盒交卷。",
        "03 缺口总结",
        "缺的不是“再一个更强的拟合器”，而是可准备证据、可推理、可改错的交互式流程——这正是 AlphaCast 要解决的问题。",
    ],
    # 3 Claim (8) — same layout as motivation
    [
        "02 核心主张与问题设定",
        "主线：预测 = 交互式共推理（准备 → 生成 → 验证）",
        "01 核心主张",
        "AlphaCast = 人的预报智慧 + LLM 的推理能力。大模型不是被动回归器，而是带着多源证据协作完成预测，并输出可检查的推理过程。",
        "02 经典问题（白话）",
        "看过去一段时间（look-back），预测未来一段时间（horizon）。学习一次映射 f：历史 → 未来。",
        "03 AlphaCast 如何扩展",
        "仍预测未来，但输入更丰富：情境信息、领域知识、以及来自案例/基线的辅助参考。结果经多步共推理得到，而不是单次前向。",
    ],
    # 4 Foundation (10)
    [
        "03 认知底座：四类证据",
        "01 准备阶段：先回答四个问题",
        "特征集：从回看窗口提取统计与时序特征。回答：这条序列现在长什么样？",
        "知识库：领域常识 + 数据中提炼的经验。回答：这类问题通常该怎么理解？",
        "上下文库：节假日、气象等外部信息。回答：当前窗口有什么特殊背景？",
        "案例库：相似历史窗与优胜策略。回答：以前类似情况谁预测得好？（下一页细讲）",
        "01 为何需要底座",
        "没有结构化证据，LLM 容易“空想曲线”。底座把数字、先验、外部事件与历史经验变成可引用材料。",
        "02 与最终预测的关系",
        "底座不直接等于答案；它为调研、生成、反思提供输入。下一页专门拆开案例库四步——最容易误解的模块。",
    ],
    # 5 Case library (12) — plan layout
    [
        "04 案例库：四步讲透",
        "关键提醒：辅助参考 ≠ 最终答案",
        "步骤 1 · 训练比武\n训练集每个 look-back 窗，让统计 / 深度 / 基础模型候选池都预测，记下该窗误差最小的模型。",
        "步骤 2 · 聚类\n把形态相似的历史窗归为一类，形成可快速检索的结构（不必与每条案例逐一比对）。",
        "步骤 3 · 测试检索\n当前窗找最像的类 → 得到类内加权辅助预测，以及近邻历史轨迹作为对照。",
        "步骤 4 · 交给生成器\n与特征、知识、情境一并输入 LLM。辅助预测是弱监督参考，最终仍经推理与反思。",
        "类比理解",
        "像先查“历史上相似天气谁预报得准”，再自己综合判断；而不是只信某一个 App，也不是简单投票出最终结果。",
        "常见误解",
        "误解：案例库 = ensemble 投票。\n正解：检索式参考 + 弱监督，最终决策在生成与反思阶段。",
        "为何重要",
        "分布偏移或异常形态时，相似历史能提供可核对的锚定，提高系统适应性与可解释性。",
    ],
    # 6 Agents (10) — framework layout
    [
        "05 三智能体推理流水线",
        "01 从证据到可校验的预测",
        "调研智能体（Investigator）：解析任务需求，从底座中筛选本窗真正需要的特征与信息，避免无关噪声进入推理。",
        "生成智能体（Generator）：整合特征、知识、情境与案例参考，经 CoT 思维链生成初步预测曲线，并留下推理依据。",
        "反思智能体（Reflector）：检查预测是否合理、证据是否充分；并审计 CoT 是否幻觉、循环或跳步。",
        "不合格则修正或重试，形成闭环，降低“一次交卷”的风险。",
        "01 调研：准备输入质量",
        "先问“需要什么证据”，再调用工具与底座，保证后续生成有材料可依。",
        "02 生成 + 反思：答题与质检",
        "生成负责提出答案与理由；反思负责质检。两者配合，才是 AlphaCast 相对 one-shot 模型的关键差异。",
    ],
    # 7 Protocol (8) — experiment layout
    [
        "06 实验怎么比：先对齐协议",
        "短序协议：输入 168，预测 24（电价等，常用 N-BEATSx 设定）。",
        "长序协议：look-back 96，预测 96（ETT、电力等）。不同协议下的数字不要混着比。",
        "01 比较对象（三类基线）\n统计模型；深度学习模型（论文称 TSLib 设定，PyTorch + Adam）；时序基础模型（如 Sundial、Chronos）。",
        "02 评价指标\nMSE（均方误差）与 MAE（平均绝对误差）。数值越小，预测越准。",
        "03 数据范围\n短序与长序共多条公开基准，覆盖高波动场景。完整数字见论文 Table 1。",
        "比较纪律",
        "先对齐窗口与指标，再比较方法优劣。下一页给出 ETTh 记忆点，并用消融说明各组件为何必要。",
    ],
    # 8 Results — remove left placeholder picture in cleanup
    [
        "07 主结果与消融",
        "目的：在标准协议下验证共推理框架有效，且各组件缺一不可。",
        "主结果记忆点（长序 ETTh · MSE）",
        "AlphaCast 约 7.641；Sundial / Chronos 约 9.4；DLinear 约 8.5。Table 1 上多数数据集最优或接近最优。",
        "消融结论（Table 2）",
        "去掉特征库、知识库或案例库，性能都会变差；反思机制对稳定性有关键贡献。",
        "主张被什么支撑",
        "提升来自“证据底座 + 生成反思闭环”，而不是“只换了一个更大的语言模型”。",
    ],
    # 9 Closing (3)
    [
        "小结 · Q&A",
        "感谢聆听",
        "贡献：① 预测重定义为共推理 ② 准备—生成—反思框架 ③ 短长序结果与消融支撑。局限：依赖 LLM 成本与系统复杂度。欢迎提问。",
    ],
]

NOTES: list[str] = [
    "大家好。今天组会精读 AlphaCast。"
    "目标是把论文方法讲清楚：它为何提出、系统如何工作、实验如何支撑。"
    "请抓住主线——预测不应只是一次映射，而应是准备证据、推理生成、再反思修正。",
    "传统方法多为一次静态输出；专家预报是多步闭环。"
    "AlphaCast 要补的是情境、案例与反思，而不是再堆一个黑盒拟合器。",
    "主张一句话：人智启发的交互式共推理。"
    "经典问题仍是看过去、预测未来；AlphaCast 额外纳入情境、知识与案例参考，用多步推理得到结果。",
    "四类证据各有分工：特征看形态，知识给先验，情境给外部事件，案例给相似历史。"
    "底座是材料，不是最终答案。下面专门讲案例库。",
    "案例库四步：训练比武、聚类、测试检索、交给生成器。"
    "请强调：辅助预测是参考，不是 ensemble 投票出的最终答案。可用天气 App 类比。",
    "三智能体：调研选证据，生成出预测和思维链，反思做质检与修正。"
    "这是相对 one-shot 模型最关键的流程差异。",
    "实验先对齐协议：短序 168 到 24，长序 96 到 96；指标 MSE、MAE；基线含统计、深度与基础模型。",
    "现场记住两句话：ETTh 上 AlphaCast 约 7.641、基础模型约 9.4；"
    "消融去掉特征/知识/案例都会变差，说明不是只换大模型。",
    "总结三句贡献，承认 LLM 成本与复杂度局限。欢迎提问。",
]


def _set_shape_text(shape, new_text: str) -> None:
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    lines = new_text.split("\n")
    while len(paras) < len(lines):
        tf.add_paragraph()
        paras = list(tf.paragraphs)
    for i, para in enumerate(paras):
        line = lines[i] if i < len(lines) else ""
        if para.runs:
            para.runs[0].text = line
            for run in para.runs[1:]:
                run.text = ""
        else:
            run = para.add_run()
            run.text = line
    for i in range(len(lines), len(paras)):
        if paras[i].runs:
            paras[i].runs[0].text = ""
            for run in paras[i].runs[1:]:
                run.text = ""


def _duplicate_slide(prs: Presentation, index: int) -> None:
    """Append a deep copy of slides[index], including image relationships."""
    source = list(prs.slides)[index]
    blank_layout = source.slide_layout
    dest = prs.slides.add_slide(blank_layout)

    for shape in list(dest.shapes):
        sp = shape.element
        sp.getparent().remove(sp)

    # Copy image parts first; map old rId -> new rId so blips keep working.
    rid_map: dict[str, str] = {}
    for rel in source.part.rels.values():
        if "image" in rel.reltype:
            new_rid = dest.part.relate_to(rel.target_part, rel.reltype)
            rid_map[rel.rId] = new_rid

    for shape in source.shapes:
        new_el = copy.deepcopy(shape.element)
        # Rewrite blip rIds if this shape references an image
        for blip in new_el.xpath(
            ".//*[local-name()='blip']"
        ):
            embed = blip.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
            )
            if embed in rid_map:
                blip.set(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed",
                    rid_map[embed],
                )
        dest.shapes._spTree.insert_element_before(new_el, "p:extLst")

    try:
        src_notes = source.notes_slide.notes_text_frame.text
        dest.notes_slide.notes_text_frame.text = src_notes
    except Exception:
        pass


def _reorder_slides(prs: Presentation, order: list[int]) -> None:
    """Reorder slides by permuting sldIdLst according to current indices in `order`."""
    sld_id_lst = prs.slides._sldIdLst
    entries = list(sld_id_lst)
    if len(entries) != len(order):
        raise RuntimeError(f"reorder size mismatch: {len(entries)} vs {len(order)}")
    new_entries = [entries[i] for i in order]
    for el in list(sld_id_lst):
        sld_id_lst.remove(el)
    for el in new_entries:
        sld_id_lst.append(el)


def _text_shapes(slide):
    return [sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]


def _remove_shape(shape) -> None:
    el = shape.element
    el.getparent().remove(el)


def _is_broken_picture(slide, shape) -> bool:
    if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
        return False
    try:
        blips = shape._element.xpath(".//*[local-name()='blip']")
        if not blips:
            return True
        rid = blips[0].get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
        )
        slide.part.related_part(rid)
        return False
    except Exception:
        return True


def _cleanup_mismatched_media(prs: Presentation) -> None:
    """Remove leftover / broken template visuals that clash with remapped paper content."""
    slides = list(prs.slides)

    # Slide 3 claim (index 2): drop broken or leftover left decoration.
    for sh in list(slides[2].shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
            _remove_shape(sh)

    # Slide 6 agents (index 5): drop broken / old right decoration.
    for sh in list(slides[5].shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and (
            _is_broken_picture(slides[5], sh) or (sh.width and sh.width > 2_500_000)
        ):
            _remove_shape(sh)

    # Slide 7 protocol (index 6): ablation table does not belong here.
    for sh in list(slides[6].shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.TABLE:
            _remove_shape(sh)

    # Slide 8 results (index 7): left Lorem-ipsum process graphic from old repro layout.
    for sh in list(slides[7].shapes):
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and sh.width and sh.width > 2_500_000:
            _remove_shape(sh)


def _add_picture(slide, path: Path, left, top, width, height) -> None:
    if not path.is_file():
        raise SystemExit(f"[error] missing figure: {path}")
    slide.shapes.add_picture(str(path), Emu(left), Emu(top), width=Emu(width), height=Emu(height))


def _embed_paper_figures(prs: Presentation) -> None:
    """Fill empty media slots with paper figures / protocol graphic."""
    slides = list(prs.slides)
    fig1 = FIGS / "fig1_motivation.png"
    fig2 = FIGS / "fig2_architecture.png"
    protocol = FIGS / "protocol_alignment.png"
    # Left panel is portrait: ETTh bar card fits better than full Table 1
    results = FIGS / "etth_mse_bars.png"
    if not results.is_file():
        results = FIGS / "table1_main.png"

    # Slide 3 left panel (claim) — Figure 1
    _add_picture(slides[2], fig1, 762000, 1651000, 4318000, 2743200)
    # Slide 6 right panel (agents) — Figure 2
    _add_picture(slides[5], fig2, 6350000, 1778000, 5207000, 2730500)
    # Slide 7 right panel (protocol) — alignment graphic (between title and bottom tip)
    _add_picture(slides[6], protocol, 6350000, 2000000, 5080000, 3000000)
    # Slide 8 left panel (results) — Table 1
    _add_picture(slides[7], results, 762000, 1524000, 3556000, 3000000)


def build() -> Path:
    if not TEMPLATE.is_file():
        raise SystemExit(f"[error] missing template: {TEMPLATE}")
    if not FIGS.is_dir():
        raise SystemExit(f"[error] missing figures dir: {FIGS}")

    shutil.copy2(TEMPLATE, OUT)
    prs = Presentation(str(OUT))

    # Template: 0 cover, 1 motiv, 2 framework, 3 exp, 4 repro, 5 plan, 6 qa
    _duplicate_slide(prs, 1)
    _duplicate_slide(prs, 2)

    # cover, motiv, claim, foundation, case, agents, protocol, results, qa
    _reorder_slides(prs, [0, 1, 7, 2, 5, 8, 3, 4, 6])

    slides = list(prs.slides)
    if len(slides) != len(SLIDE_TEXTS):
        raise SystemExit(f"[error] expected {len(SLIDE_TEXTS)} slides, got {len(slides)}")

    for si, slide in enumerate(slides):
        texts = _text_shapes(slide)
        wanted = SLIDE_TEXTS[si]
        if len(texts) != len(wanted):
            raise SystemExit(
                f"[error] slide {si + 1}: text shapes {len(texts)} != replacements {len(wanted)}; "
                f"titles={[t.text_frame.text[:20] for t in texts[:3]]}"
            )
        for shape, new_text in zip(texts, wanted):
            _set_shape_text(shape, new_text)
        slide.notes_slide.notes_text_frame.text = NOTES[si]

    _cleanup_mismatched_media(prs)
    _embed_paper_figures(prs)

    prs.save(str(OUT))
    print(f"[ok] wrote {OUT} ({len(slides)} slides)")
    return OUT


if __name__ == "__main__":
    build()
