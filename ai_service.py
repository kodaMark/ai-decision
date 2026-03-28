"""
Multi-model AI service for AI深度决策.

Models used:
- GLM-5 (智谱 AI): SOP conversation guidance
- Claude claude-opus-4-6 (Anthropic): Step 2, Step 4, Step 5
- GPT-5.4 via Qiniu relay (openai/gpt-5.4): Step 3, fallback to Claude Sonnet
- 讯飞 STT: speech-to-text (placeholder)
- Edge TTS (Microsoft, free): text-to-speech
"""

import asyncio
import json
import os
import re
from typing import Generator

import anthropic
from openai import OpenAI

# ---------------------------------------------------------------------------
# Client initialisation (lazy, uses env vars)
# ---------------------------------------------------------------------------


def _get_anthropic_client(use_backup: bool = False) -> anthropic.Anthropic:
    """Return Anthropic client. use_backup=True switches to backup key from DB/env."""
    try:
        from models import APIConfig
        if use_backup:
            api_key = APIConfig.get("backup_api_key") or os.environ.get("BACKUP_API_KEY", "")
            base_url = APIConfig.get("backup_base_url") or os.environ.get("BACKUP_BASE_URL", "")
        else:
            api_key = APIConfig.get("primary_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = APIConfig.get("primary_base_url") or os.environ.get("ANTHROPIC_BASE_URL", "")
    except Exception:
        # Outside app context: fall back to env vars
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def _anthropic_call_with_fallback(fn, *args, **kwargs):
    """Call fn(client, *args, **kwargs) with primary key; retry with backup on failure."""
    try:
        return fn(_get_anthropic_client(use_backup=False), *args, **kwargs)
    except Exception as e:
        try:
            from models import APIConfig
            has_backup = bool(APIConfig.get("backup_api_key") or os.environ.get("BACKUP_API_KEY"))
        except Exception:
            has_backup = bool(os.environ.get("BACKUP_API_KEY"))
        if has_backup:
            return fn(_get_anthropic_client(use_backup=True), *args, **kwargs)
        raise


def _get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def _get_doubao_client() -> OpenAI:
    """豆包 API 兼容 OpenAI 格式."""
    return OpenAI(
        api_key=os.environ["DOUBAO_API_KEY"],
        base_url="https://ark.volces.com/api/v3",
    )


# ---------------------------------------------------------------------------
# 豆包 Lite: conversation guidance (streaming)
# ---------------------------------------------------------------------------


def chat_with_glm_stream(messages: list) -> Generator[str, None, None]:
    """
    Stream SOP conversation response via GLM-5 (Zhipu AI).
    messages: list of {"role": ..., "content": ...} including system message.
    Yields text chunks.
    """
    model = os.environ.get("SOP_MODEL", "openai/gpt-5.4-mini")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        stream = client.chat.completions.create(
            model=model,
            max_tokens=300,
            messages=messages,
            stream=True,
        )
        buffer = ""
        done = False
        for chunk in stream:
            if done:
                break
            delta = chunk.choices[0].delta.content
            if delta:
                buffer += delta
                if "？" in buffer:
                    cut = buffer.index("？") + 1
                    yield buffer[:cut]
                    done = True
                else:
                    yield delta
    except Exception as e:
        yield f"\n（抱歉，网络出现了小问题，请重试。错误：{e}）"


def chat_with_glm(messages: list) -> str:
    """Non-streaming call, returns full response string."""
    full = "".join(chat_with_glm_stream(messages))
    if "？" in full:
        full = full[: full.index("？") + 1]
    return full


# ---------------------------------------------------------------------------
# Step 2 — Claude: Structured analysis
# ---------------------------------------------------------------------------

STEP2_SYSTEM = """你是一位顶级战略分析师。基于用户提供的决策信息，进行严格的结构化分析。

请以 JSON 格式输出，包含以下字段：
{
  "fact_table": [
    {"category": "已知事实", "items": ["事实1", "事实2", ...]},
    {"category": "假设前提", "items": ["假设1", ...]},
    {"category": "情绪因素", "items": ["..."]},
  ],
  "information_gaps": ["缺失信息1", "缺失信息2", ...],
  "core_contradiction": "用一两句话描述这个决定的核心矛盾或张力",
  "key_variables": ["影响结果的关键变量1", "变量2", ...],
  "decision_type": "类型描述（如：不可逆的人生选择 / 可迭代的商业决策 / 关系类决定）",
  "complexity_score": 7,
  "complexity_reason": "复杂度评分说明（1-10分）"
}

只输出 JSON，不要有其他文字。"""


def run_step2_claude(collected_info: str) -> dict:
    """
    Claude structured analysis of the collected decision info.
    Returns parsed dict.
    """
    client = _get_anthropic_client()
    prompt = f"以下是用户的决策信息：\n\n{collected_info}\n\n请进行结构化分析。"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=STEP2_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text
            break

    return _parse_json_safe(raw_text, default_key="raw", label="Step2")


# ---------------------------------------------------------------------------
# Step 3 — GPT-4.5: Three alternative pathways
# ---------------------------------------------------------------------------

STEP3_SYSTEM = """你是一位战略顾问，专门帮助人们发现决策的多元路径，并进行量化推演。

基于用户的决策信息和前期分析，设计3条截然不同的行动路径。每条路径必须做量化估算和双场景推演，不能只给 pros/cons。

请以 JSON 格式输出：
{
  "pathways": [
    {
      "id": 1,
      "name": "路径名称（2-4字，有冲击力）",
      "tagline": "一句话描述这条路的精髓",
      "description": "详细描述这条路径的具体做法（2-3句话）",
      "impact_12m": {
        "direction": "正向/负向/中性",
        "estimate": "12个月内的核心影响估算（尽量量化，如时间、金钱、机会成本、关系变化等）",
        "key_assumption": "这个估算成立的关键前提"
      },
      "scenarios_2y": {
        "base": "2年后基础场景：最可能出现的状态",
        "downside": "2年后下行场景：如果关键假设失败，会是什么局面"
      },
      "reversibility": "高/中/低",
      "reversibility_note": "说明为什么可逆或不可逆，一旦走上这条路哪些事情无法撤回",
      "key_risk": "这条路最大的单一风险",
      "time_to_first_signal": "多久能看到第一个明确的结果信号"
    },
    ... (共3条路径)
  ],
  "reversibility_comparison": "三条路径可逆性对比总结，哪条路一旦走上就很难回头",
  "critical_question": "做这个决定前最需要回答的一个关键问题"
}

只输出 JSON，不要有其他文字。"""


def run_step3_gpt(collected_info: str, step2_result: dict) -> dict:
    """
    GPT-5.4 via Qiniu relay generates three alternative pathways.
    Falls back to Claude Sonnet if OpenAI call fails.
    Returns parsed dict.
    """
    step2_text = json.dumps(step2_result, ensure_ascii=False, indent=2)
    prompt = (
        f"用户决策信息：\n\n{collected_info}\n\n"
        f"结构化分析结果：\n\n{step2_text}\n\n"
        "请设计3条截然不同的行动路径。"
    )

    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        response = client.chat.completions.create(
            model="openai/gpt-5.4",
            max_tokens=2500,
            messages=[
                {"role": "system", "content": STEP3_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw_text = response.choices[0].message.content or ""
    except Exception:
        # Fallback to Claude Sonnet
        client = _get_anthropic_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            system=STEP3_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text = block.text
                break

    return _parse_json_safe(raw_text, default_key="raw", label="Step3")


# ---------------------------------------------------------------------------
# Step 4 — Claude: Weighted scoring matrix
# ---------------------------------------------------------------------------

STEP4_SYSTEM = """你是一位量化决策专家。基于收集的信息和三条路径，建立加权评分矩阵。

评分维度（4个，权重之和=100%）：
1. 目标匹配度：与用户期望结果的吻合程度
2. 风险可控性：最坏情况的可承受程度
3. 资源可行性：现有资源条件下的执行可能
4. 时间适配性：与时间节点的匹配程度

请以 JSON 格式输出：
{
  "dimensions": [
    {"id": "goal_match", "name": "目标匹配度", "weight": 30, "description": "..."},
    {"id": "risk_control", "name": "风险可控性", "weight": 25, "description": "..."},
    {"id": "resource_fit", "name": "资源可行性", "weight": 25, "description": "..."},
    {"id": "time_fit", "name": "时间适配性", "weight": 20, "description": "..."}
  ],
  "scores": [
    {
      "pathway_id": 1,
      "pathway_name": "路径名称",
      "dimension_scores": {
        "goal_match": {"score": 8, "reason": "评分理由"},
        "risk_control": {"score": 6, "reason": "..."},
        "resource_fit": {"score": 7, "reason": "..."},
        "time_fit": {"score": 9, "reason": "..."}
      },
      "weighted_total": 7.45,
      "overall_comment": "对这条路径的整体评价"
    },
    ... (3条路径各自评分)
  ],
  "ranking": [路径id按分数排序, 如 [1, 3, 2]],
  "score_note": "评分说明，提醒用户分数只是参考，不是唯一标准"
}

只输出 JSON，不要有其他文字。"""


def run_step4_claude(collected_info: str, step2_result: dict, step3_result: dict) -> dict:
    """
    Claude builds a weighted scoring matrix for the three pathways.
    Returns parsed dict.
    """
    client = _get_anthropic_client()
    step2_text = json.dumps(step2_result, ensure_ascii=False, indent=2)
    step3_text = json.dumps(step3_result, ensure_ascii=False, indent=2)
    prompt = (
        f"用户决策信息：\n\n{collected_info}\n\n"
        f"结构化分析：\n\n{step2_text}\n\n"
        f"三条路径：\n\n{step3_text}\n\n"
        "请建立加权评分矩阵。"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=STEP4_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text
            break

    return _parse_json_safe(raw_text, default_key="raw", label="Step4")


# ---------------------------------------------------------------------------
# Step 5 — Claude: Final recommendation
# ---------------------------------------------------------------------------

STEP5_SYSTEM = """你是一位顶级人生顾问，也是严谨的战略思维者。

基于完整的分析，给出你的最终建议。你的建议要有立场，要敢于推荐，但也要诚实面对不确定性。

请以 JSON 格式输出：
{
  "recommended_pathway_id": 1,
  "recommended_pathway_name": "路径名称",
  "confidence": "高/中/低",
  "confidence_reason": "推荐信心说明",
  "core_recommendation": "用2-3句有力的话说明你的核心建议",
  "reasoning_summary": "综合以上分析，为什么这条路是最佳选择？（详细版，3-5句话）",
  "red_lines": [
    {"line": "绝对不能越过的底线1", "reason": "为什么"},
    {"line": "红线2", "reason": "..."}
  ],
  "green_lights": [
    {"signal": "出现这个信号说明在正确轨道上", "action": "此时应该做什么"},
    {"signal": "绿灯信号2", "action": "..."}
  ],
  "action_timeline": [
    {"phase": "未来7天", "actions": ["行动1", "行动2"]},
    {"phase": "未来30天", "actions": ["行动1", "行动2"]},
    {"phase": "未来90天", "actions": ["行动1"]}
  ],
  "decision_reframe": "换一个角度看这个决定——一句洞见，帮助用户超越当下的焦虑",
  "final_words": "给用户的一段温暖而有力的结语（2-3句话）"
}

只输出 JSON，不要有其他文字。"""


def run_step5_claude(
    collected_info: str,
    step2_result: dict,
    step3_result: dict,
    step4_result: dict,
) -> dict:
    """
    Claude generates the final recommendation.
    Returns parsed dict.
    """
    client = _get_anthropic_client()
    step2_text = json.dumps(step2_result, ensure_ascii=False, indent=2)
    step3_text = json.dumps(step3_result, ensure_ascii=False, indent=2)
    step4_text = json.dumps(step4_result, ensure_ascii=False, indent=2)
    prompt = (
        f"用户决策信息：\n\n{collected_info}\n\n"
        f"结构化分析：\n\n{step2_text}\n\n"
        f"三条路径：\n\n{step3_text}\n\n"
        f"评分矩阵：\n\n{step4_text}\n\n"
        "请给出最终建议。"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=STEP5_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text
            break

    return _parse_json_safe(raw_text, default_key="raw", label="Step5")


# ---------------------------------------------------------------------------
# Report HTML generation
# ---------------------------------------------------------------------------


def generate_report_html(
    collected_info: str,
    step2: dict,
    step3: dict,
    step4: dict,
    step5: dict,
) -> str:
    """
    Combine all step results into a beautiful HTML report string.
    This HTML is injected into report.html via Jinja.
    """
    # Extract key data with safe fallbacks
    core_contradiction = step2.get("core_contradiction", "")
    decision_type = step2.get("decision_type", "")
    complexity_score = step2.get("complexity_score", "")
    info_gaps = step2.get("information_gaps", [])
    fact_table = step2.get("fact_table", [])
    key_variables = step2.get("key_variables", [])

    pathways = step3.get("pathways", [])
    synthesis = step3.get("synthesis", "")

    dimensions = step4.get("dimensions", [])
    scores = step4.get("scores", [])
    ranking = step4.get("ranking", [])

    rec_pathway_name = step5.get("recommended_pathway_name", "")
    confidence = step5.get("confidence", "")
    core_rec = step5.get("core_recommendation", "")
    reasoning = step5.get("reasoning_summary", "")
    red_lines = step5.get("red_lines", [])
    green_lights = step5.get("green_lights", [])
    timeline = step5.get("action_timeline", [])
    reframe = step5.get("decision_reframe", "")
    final_words = step5.get("final_words", "")

    html_parts = []

    # --- Section 1: Background ---
    html_parts.append('<section class="report-section" id="s1">')
    html_parts.append('<div class="section-label">01 · 决策背景</div>')
    html_parts.append(f'<div class="core-contradiction">{_esc(core_contradiction)}</div>')
    html_parts.append('<div class="meta-row">')
    html_parts.append(f'<span class="badge badge-blue">{_esc(decision_type)}</span>')
    html_parts.append(
        f'<span class="badge badge-orange">复杂度 {complexity_score}/10</span>'
    )
    html_parts.append("</div>")

    if fact_table:
        html_parts.append('<div class="fact-table">')
        for cat in fact_table:
            html_parts.append(f'<div class="fact-category">')
            html_parts.append(f'<div class="fact-cat-name">{_esc(cat.get("category",""))}</div>')
            html_parts.append("<ul>")
            for item in cat.get("items", []):
                html_parts.append(f"<li>{_esc(item)}</li>")
            html_parts.append("</ul></div>")
        html_parts.append("</div>")

    if info_gaps:
        html_parts.append('<div class="info-gaps">')
        html_parts.append('<div class="sub-title">信息缺口</div>')
        for gap in info_gaps:
            html_parts.append(f'<div class="gap-item">⚠️ {_esc(gap)}</div>')
        html_parts.append("</div>")

    html_parts.append("</section>")

    # --- Section 2: Three Pathways ---
    html_parts.append('<section class="report-section" id="s2">')
    html_parts.append('<div class="section-label">02 · 三条路径</div>')
    reversibility_colors = {"高": "green", "中": "orange", "低": "red"}
    for pw in pathways:
        pid = pw.get("id", "")
        is_recommended = (pid == step5.get("recommended_pathway_id"))
        rec_class = "pathway-recommended" if is_recommended else ""
        html_parts.append(f'<div class="pathway-card {rec_class}">')
        if is_recommended:
            html_parts.append('<div class="recommended-tag">推荐路径</div>')
        html_parts.append(
            f'<div class="pathway-header">'
            f'<span class="pathway-num">路径 {pid}</span>'
            f'<span class="pathway-name">{_esc(pw.get("name",""))}</span>'
            f"</div>"
        )
        html_parts.append(f'<div class="pathway-tagline">{_esc(pw.get("tagline",""))}</div>')
        html_parts.append(f'<div class="pathway-desc">{_esc(pw.get("description",""))}</div>')

        # 12个月影响估算
        impact = pw.get("impact_12m", {})
        if impact:
            direction = impact.get("direction", "")
            direction_class = {"正向": "positive", "负向": "negative"}.get(direction, "neutral")
            html_parts.append(
                f'<div class="impact-box impact-{direction_class}">'
                f'<div class="impact-title">📊 12个月影响（{_esc(direction)}）</div>'
                f'<div class="impact-estimate">{_esc(impact.get("estimate",""))}</div>'
                f'<div class="impact-assumption">前提：{_esc(impact.get("key_assumption",""))}</div>'
                f'</div>'
            )

        # 2年双场景
        scenarios = pw.get("scenarios_2y", {})
        if scenarios:
            html_parts.append('<div class="scenarios">')
            html_parts.append('<div class="sub-title">2年后推演</div>')
            html_parts.append(
                f'<div class="scenario base-scenario">'
                f'<span class="scenario-label">基础场景</span>'
                f'<span>{_esc(scenarios.get("base",""))}</span>'
                f'</div>'
            )
            html_parts.append(
                f'<div class="scenario down-scenario">'
                f'<span class="scenario-label">下行场景</span>'
                f'<span>{_esc(scenarios.get("downside",""))}</span>'
                f'</div>'
            )
            html_parts.append('</div>')

        rev = pw.get("reversibility", "中")
        rev_color = reversibility_colors.get(rev, "orange")
        html_parts.append(f'<div class="pathway-meta">')
        html_parts.append(
            f'<span class="rev-badge rev-{rev_color}">可逆性：{_esc(rev)}</span>'
        )
        html_parts.append(
            f'<span class="time-badge">首个信号：{_esc(pw.get("time_to_first_signal",""))}</span>'
        )
        html_parts.append("</div>")
        if pw.get("reversibility_note"):
            html_parts.append(f'<div class="rev-note">{_esc(pw.get("reversibility_note",""))}</div>')

        html_parts.append(
            f'<div class="key-risk">核心风险：{_esc(pw.get("key_risk",""))}</div>'
        )
        html_parts.append("</div>")  # pathway-card

    if synthesis:
        html_parts.append(f'<div class="synthesis">{_esc(synthesis)}</div>')

    rev_comparison = step3.get("reversibility_comparison", "")
    critical_q = step3.get("critical_question", "")
    if rev_comparison:
        html_parts.append(f'<div class="rev-comparison"><strong>可逆性对比：</strong>{_esc(rev_comparison)}</div>')
    if critical_q:
        html_parts.append(f'<div class="critical-question">❓ 关键问题：{_esc(critical_q)}</div>')

    html_parts.append("</section>")

    # --- Section 3: Scoring Matrix ---
    html_parts.append('<section class="report-section" id="s3">')
    html_parts.append('<div class="section-label">03 · 量化评分</div>')

    if dimensions and scores:
        html_parts.append('<div class="score-table-wrap">')
        html_parts.append('<table class="score-table">')
        html_parts.append("<thead><tr><th>维度</th>")
        for sc in scores:
            rec_mark = " ★" if sc.get("pathway_id") == step5.get("recommended_pathway_id") else ""
            html_parts.append(f'<th>{_esc(sc.get("pathway_name",""))}{rec_mark}</th>')
        html_parts.append("</tr></thead><tbody>")

        for dim in dimensions:
            dim_id = dim.get("id", "")
            html_parts.append(
                f'<tr><td class="dim-name">{_esc(dim.get("name",""))}'
                f'<span class="dim-weight">({dim.get("weight",0)}%)</span></td>'
            )
            for sc in scores:
                ds = sc.get("dimension_scores", {}).get(dim_id, {})
                score_val = ds.get("score", "-")
                reason = ds.get("reason", "")
                html_parts.append(
                    f'<td class="score-cell" title="{_esc(reason)}">'
                    f'<span class="score-num">{score_val}</span></td>'
                )
            html_parts.append("</tr>")

        # Total row
        html_parts.append('<tr class="total-row"><td>加权总分</td>')
        for sc in scores:
            total = sc.get("weighted_total", "-")
            html_parts.append(f'<td><strong>{total}</strong></td>')
        html_parts.append("</tr>")
        html_parts.append("</tbody></table></div>")

        # Score note
        note = step4.get("score_note", "")
        if note:
            html_parts.append(f'<div class="score-note">{_esc(note)}</div>')

    html_parts.append("</section>")

    # --- Section 4: Final Recommendation ---
    html_parts.append('<section class="report-section" id="s4">')
    html_parts.append('<div class="section-label">04 · 最终建议</div>')

    html_parts.append(
        f'<div class="rec-header">'
        f'<div class="rec-title">建议选择：<span>{_esc(rec_pathway_name)}</span></div>'
        f'<span class="confidence-badge conf-{confidence}">{_esc(confidence)}信心</span>'
        f"</div>"
    )
    html_parts.append(f'<div class="core-rec">{_esc(core_rec)}</div>')
    html_parts.append(f'<div class="reasoning">{_esc(reasoning)}</div>')

    if reframe:
        html_parts.append(f'<div class="reframe">💡 {_esc(reframe)}</div>')

    # Red lines
    if red_lines:
        html_parts.append('<div class="red-lines">')
        html_parts.append('<div class="sub-title red-title">🚫 红线（绝对不能越过）</div>')
        for rl in red_lines:
            html_parts.append(
                f'<div class="red-line-item">'
                f'<strong>{_esc(rl.get("line",""))}</strong>'
                f'<span>{_esc(rl.get("reason",""))}</span>'
                f"</div>"
            )
        html_parts.append("</div>")

    # Green lights
    if green_lights:
        html_parts.append('<div class="green-lights">')
        html_parts.append('<div class="sub-title green-title">✅ 绿灯信号（说明走对了）</div>')
        for gl in green_lights:
            html_parts.append(
                f'<div class="green-light-item">'
                f'<strong>{_esc(gl.get("signal",""))}</strong>'
                f'<span>→ {_esc(gl.get("action",""))}</span>'
                f"</div>"
            )
        html_parts.append("</div>")

    # Timeline
    if timeline:
        html_parts.append('<div class="timeline">')
        html_parts.append('<div class="sub-title">行动时间轴</div>')
        for phase in timeline:
            html_parts.append(
                f'<div class="timeline-phase">'
                f'<div class="phase-label">{_esc(phase.get("phase",""))}</div>'
                f"<ul>"
            )
            for action in phase.get("actions", []):
                html_parts.append(f"<li>{_esc(action)}</li>")
            html_parts.append("</ul></div>")
        html_parts.append("</div>")

    if final_words:
        html_parts.append(f'<div class="final-words">{_esc(final_words)}</div>')

    html_parts.append("</section>")

    return "\n".join(html_parts)


def generate_full_report(session_id: int, app_context) -> None:
    """
    Orchestrate steps 2-5 and save results to DecisionReport.
    Runs synchronously (call from background thread with app context pushed).
    """
    import traceback
    from models import ConversationMessage, DecisionReport, DecisionSession, db
    from conversation import ConversationSOP

    print(f"[分析] Session {session_id} 开始", flush=True)

    with app_context:
        session = DecisionSession.query.get(session_id)
        if not session:
            print(f"[分析] Session {session_id} 不存在", flush=True)
            return

        session.status = "analyzing"
        db.session.commit()

        # Get or create report record
        report = session.report or DecisionReport(session_id=session_id)
        db.session.add(report)

        try:
            # Reconstruct collected info from messages
            messages = (
                ConversationMessage.query.filter_by(session_id=session_id)
                .order_by(ConversationMessage.created_at)
                .all()
            )
            sop = ConversationSOP.from_messages(messages)
            collected_info = sop.extract_collected_info()
            report.step1_raw = collected_info
            db.session.commit()
            print(f"[分析] Step1 完成（信息收集）", flush=True)

            # Step 2: Claude structured analysis
            print(f"[分析] Step2 开始（结构化分析）...", flush=True)
            step2 = run_step2_claude(collected_info)
            report.step2_raw = json.dumps(step2, ensure_ascii=False)
            db.session.commit()
            print(f"[分析] Step2 完成", flush=True)

            # Step 3: Claude three pathways
            print(f"[分析] Step3 开始（三条路径）...", flush=True)
            step3 = run_step3_gpt(collected_info, step2)
            report.step3_raw = json.dumps(step3, ensure_ascii=False)
            db.session.commit()
            print(f"[分析] Step3 完成", flush=True)

            # Step 4: Claude scoring
            print(f"[分析] Step4 开始（量化评分）...", flush=True)
            step4 = run_step4_claude(collected_info, step2, step3)
            report.step4_raw = json.dumps(step4, ensure_ascii=False)
            db.session.commit()
            print(f"[分析] Step4 完成", flush=True)

            # Step 5: Claude recommendation
            print(f"[分析] Step5 开始（最终建议）...", flush=True)
            step5 = run_step5_claude(collected_info, step2, step3, step4)
            report.step5_raw = json.dumps(step5, ensure_ascii=False)
            db.session.commit()
            print(f"[分析] Step5 完成", flush=True)

            # Generate HTML
            report.final_report_html = generate_report_html(
                collected_info, step2, step3, step4, step5
            )
            session.status = "done"
            db.session.commit()
            print(f"[分析] Session {session_id} 全部完成！", flush=True)

        except Exception as e:
            print(f"[分析] 出错: {e}", flush=True)
            traceback.print_exc()
            report.error_message = str(e)
            session.status = "error"
            db.session.commit()
            raise


# ---------------------------------------------------------------------------
# STT / TTS placeholders (讯飞)
# ---------------------------------------------------------------------------


def xfyun_stt(audio_bytes: bytes, audio_format: str = "wav") -> str:
    """
    TODO: integrate real 讯飞 STT API.
    Currently returns a placeholder error message.

    Real implementation would:
    1. Load XFYUN_APP_ID, XFYUN_API_KEY, XFYUN_API_SECRET from env
    2. Build WebSocket URL with auth signature
    3. Send audio data and receive transcription
    """
    app_id = os.environ.get("XFYUN_APP_ID", "")
    api_key = os.environ.get("XFYUN_API_KEY", "")

    if not app_id or not api_key:
        raise NotImplementedError(
            "讯飞 STT 未配置。请在 .env 中设置 XFYUN_APP_ID, XFYUN_API_KEY, XFYUN_API_SECRET。"
        )

    # TODO: integrate real Xfyun STT WebSocket API here
    # Reference: https://www.xfyun.cn/doc/asr/voicedictation/API.html
    raise NotImplementedError("讯飞 STT 集成待实现")


async def _edge_tts_async(text: str, voice: str) -> bytes:
    """Async Edge TTS call, returns MP3 bytes."""
    import edge_tts
    import io
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def edge_tts_synthesize(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> bytes:
    """
    Free TTS via Microsoft Edge TTS.
    voice options: zh-CN-XiaoxiaoNeural (女) / zh-CN-YunxiNeural (男)
    Returns MP3 bytes.
    """
    return asyncio.run(_edge_tts_async(text, voice))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text) -> str:
    """HTML-escape a string value."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _parse_json_safe(raw_text: str, default_key: str = "raw", label: str = "") -> dict:
    """
    Attempt to parse JSON from raw_text.
    If it fails, wrap the raw text in a dict under default_key.
    Strips markdown code fences if present.
    """
    # Strip ```json ... ``` fences
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {default_key: raw_text, "_parse_error": f"{label}: JSON parse failed"}
