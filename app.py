"""
AI深度决策 — Flask application entry point.

Run with:
    python app.py
"""

import json
import os
import queue
import threading
import uuid
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
    url_for,
)

load_dotenv()

from models import APIConfig, ConversationMessage, DecisionReport, DecisionSession, StepFeedback, SupportTicket, db
from conversation import ConversationSOP
from ai_service import chat_with_glm_stream, generate_full_report, xfyun_stt

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

db_url = os.environ.get("DATABASE_URL", "sqlite:///decisions.db")
# Railway PostgreSQL compat: replace postgres:// with postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_or_create_user_session_id() -> str:
    """Return a persistent browser-session UUID for the current visitor."""
    if "uid" not in session:
        session["uid"] = str(uuid.uuid4())
    return session["uid"]


def _get_decision_session(session_id: int) -> DecisionSession:
    """Fetch DecisionSession or raise 404."""
    ds = DecisionSession.query.get(session_id)
    if not ds:
        from flask import abort
        abort(404)
    return ds


def _history_to_glm(messages: list) -> list:
    """Convert ConversationMessage ORM list to GLM-compatible dicts."""
    return [{"role": m.role, "content": m.content} for m in messages]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/session/start", methods=["POST"])
def session_start():
    """Create a new DecisionSession and return its ID."""
    uid = _get_or_create_user_session_id()
    ds = DecisionSession(user_session_id=uid)
    db.session.add(ds)
    db.session.commit()
    return jsonify({"session_id": ds.id})


@app.route("/session/<int:session_id>")
def session_view(session_id: int):
    ds = _get_decision_session(session_id)
    # Load existing messages
    messages = (
        ConversationMessage.query.filter_by(session_id=session_id)
        .order_by(ConversationMessage.created_at)
        .all()
    )
    return render_template(
        "session.html",
        ds=ds,
        messages=[m.to_dict() for m in messages],
    )


@app.route("/session/<int:session_id>/message", methods=["POST"])
def session_message(session_id: int):
    """
    Receive a user message, persist it, get GLM response (SSE streaming).
    Expects JSON: {"content": "user text"}
    Returns Server-Sent Events stream.
    """
    ds = _get_decision_session(session_id)

    if ds.status not in ("collecting",):
        return jsonify({"error": "Session is not in collecting state"}), 400

    data = request.get_json(silent=True) or {}
    user_text = (data.get("content") or "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    # Load existing messages to reconstruct SOP state
    existing_messages = (
        ConversationMessage.query.filter_by(session_id=session_id)
        .order_by(ConversationMessage.created_at)
        .all()
    )

    sop = ConversationSOP.from_messages(existing_messages)
    current_q = sop.get_current_question()
    current_step = current_q["step"] if current_q else None

    # Count follow-ups already done on this step (user messages without a step recorded)
    followup_count = sum(
        1 for m in existing_messages
        if m.role == "user" and m.step is None
    )

    # Persist user message — step is None until GLM confirms answer is sufficient
    user_msg = ConversationMessage(
        session_id=session_id,
        role="user",
        content=user_text,
        step=None,
    )
    db.session.add(user_msg)
    db.session.commit()
    user_msg_id = user_msg.id

    # Build GLM message list
    history = _history_to_glm(existing_messages)
    history.append({"role": "user", "content": user_text})
    glm_messages = sop.build_glm_messages(history, followup_count=followup_count)

    def generate():
        yield "data: " + json.dumps({"type": "start"}) + "\n\n"

        result_holder = {}

        def fetch_full():
            try:
                result_holder["text"] = chat_with_glm(glm_messages)
            except Exception as e:
                result_holder["error"] = str(e)

        t = threading.Thread(target=fetch_full, daemon=True)
        t.start()
        # keepalive pings while waiting
        while t.is_alive():
            t.join(timeout=10)
            if t.is_alive():
                yield ": ping\n\n"

        if "error" in result_holder:
            yield "data: " + json.dumps({"type": "error", "message": result_holder["error"]}) + "\n\n"
            return

        assistant_text = result_holder.get("text", "")
        advanced = "[NEXT]" in assistant_text
        clean_text = assistant_text.replace("[NEXT]", "").lstrip()

        # 模拟流式输出（每次10个字符）
        chunk_size = 10
        for i in range(0, len(clean_text), chunk_size):
            chunk = clean_text[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "chunk", "content": chunk}) + "\n\n"

        with app.app_context():
            # If GLM approved the answer, mark user message with the step
            if advanced and current_step:
                um = ConversationMessage.query.get(user_msg_id)
                if um:
                    um.step = current_step

            asst_msg = ConversationMessage(
                session_id=session_id,
                role="assistant",
                content=clean_text,
                step=None,
            )
            db.session.add(asst_msg)
            db.session.commit()

            # Recompute completion after possible step advancement
            updated_messages = (
                ConversationMessage.query.filter_by(session_id=session_id)
                .order_by(ConversationMessage.created_at)
                .all()
            )
            collection_complete = ConversationSOP.from_messages(updated_messages).is_collection_complete()

        yield "data: " + json.dumps({"type": "done", "collection_complete": collection_complete}) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/session/<int:session_id>/analyze", methods=["POST"])
def session_analyze(session_id: int):
    """Trigger background analysis job."""
    ds = _get_decision_session(session_id)

    if ds.status == "analyzing":
        return jsonify({"status": "already_analyzing"}), 200
    if ds.status == "done":
        return jsonify({"status": "already_done", "redirect": url_for("session_report", session_id=session_id)}), 200

    ds.status = "analyzing"
    db.session.commit()

    # Run in background thread
    ctx = app.app_context()

    def run():
        generate_full_report(session_id, ctx)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return jsonify({"status": "started"})


@app.route("/session/<int:session_id>/status")
def session_status(session_id: int):
    """Poll analysis status."""
    ds = _get_decision_session(session_id)
    resp = {"status": ds.status}
    if ds.status == "done":
        resp["redirect"] = url_for("session_report", session_id=session_id)
    if ds.status == "error" and ds.report:
        resp["error"] = ds.report.error_message
    return jsonify(resp)


@app.route("/session/<int:session_id>/report")
def session_report(session_id: int):
    """Display the final report."""
    ds = _get_decision_session(session_id)
    if ds.status != "done" or not ds.report:
        return redirect(url_for("session_view", session_id=session_id))
    return render_template("report.html", ds=ds, report=ds.report)


@app.route("/debug/quick-test")
def debug_quick_test():
    """
    开发用：跳过对话，用预设答案直接测试分析流程。
    访问此路由 → 自动创建会话 + 填入9问答案 + 触发分析 → 跳转状态页
    """
    uid = "test-" + str(uuid.uuid4())
    ds = DecisionSession(user_session_id=uid)
    db.session.add(ds)
    db.session.commit()

    # 预设测试答案（真实场景：要不要辞职去创业）
    test_answers = [
        (1, "我在考虑要不要辞掉现在的工作，去全职做自己的AI工具产品"),
        (2, "需要在3个月内决定，因为公司新项目要开始了，届时很难抽身"),
        (3, "主要涉及我自己、女朋友（她收入稳定但希望我稳定），以及现在的老板"),
        (4, "最担心的是创业失败后没有收入，同时影响和女朋友的关系"),
        (5, "最好的结果是产品半年内找到付费用户，一年后月收入超过现在工资"),
        (6, "最坏能接受的是：花6个月时间验证失败，然后重新找工作，损失半年收入"),
        (7, "目前已有一个MVP产品原型，有3个种子用户愿意试用，存款够支撑9个月生活"),
        (8, "不确定目标用户的付费意愿，也不确定能否在6个月内做到产品市场契合"),
        (9, "考虑过：继续上班同时兼职做，或者找联合创始人分担风险"),
    ]

    for step, content in test_answers:
        msg = ConversationMessage(
            session_id=ds.id,
            role="user",
            content=content,
            step=step,
        )
        db.session.add(msg)
    db.session.commit()

    # 触发分析
    ds.status = "analyzing"
    db.session.commit()

    new_session_id = ds.id  # 在 app context 内取出纯 int，避免后台线程懒加载
    ctx = app.app_context()

    def run():
        generate_full_report(new_session_id, ctx)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    return redirect(url_for("debug_status", session_id=ds.id))


@app.route("/debug/raw/<int:session_id>")
def debug_raw(session_id: int):
    """Dump raw step4/step5 JSON from DB for diagnosis."""
    from models import DecisionReport
    report = DecisionReport.query.filter_by(session_id=session_id).first()
    if not report:
        return {"error": "report not found"}, 404
    return {
        "step4_raw": report.step4_raw,
        "step5_raw": report.step5_raw,
    }


@app.route("/debug/status/<int:session_id>")
def debug_status(session_id: int):
    """简单状态页，轮询分析进度并显示完整错误信息。"""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>分析中...</title>
<style>
body {{ font-family: sans-serif; max-width: 600px; margin: 60px auto; padding: 0 20px; }}
.status {{ font-size: 18px; margin: 20px 0; }}
.error {{ background: #fee; border: 1px solid #f99; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-break: break-all; }}
.done {{ color: green; }}
.analyzing {{ color: #888; }}
</style>
</head>
<body>
<h2>AI深度决策 · 分析状态</h2>
<div class="status analyzing" id="status">正在分析中，请稍候（约30秒）...</div>
<div id="error-box"></div>
<script>
const sid = {session_id};
function poll() {{
    fetch('/session/' + sid + '/status')
        .then(r => r.json())
        .then(data => {{
            const el = document.getElementById('status');
            if (data.status === 'done') {{
                el.className = 'status done';
                el.textContent = '✅ 分析完成！正在跳转报告...';
                setTimeout(() => window.location = data.redirect, 800);
            }} else if (data.status === 'error') {{
                el.className = 'status';
                el.textContent = '❌ 分析出错';
                document.getElementById('error-box').innerHTML =
                    '<div class="error"><strong>错误详情：</strong>\\n' + (data.error || '未知错误') + '</div>';
            }} else {{
                setTimeout(poll, 2000);
            }}
        }})
        .catch(() => setTimeout(poll, 3000));
}}
poll();
</script>
</body></html>"""


@app.route("/api/stt", methods=["POST"])
def api_stt():
    """
    Speech-to-text endpoint.
    Expects multipart/form-data with 'audio' file field.
    Returns JSON: {"text": "transcribed text"}
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    audio_format = audio_file.filename.rsplit(".", 1)[-1].lower() if audio_file.filename else "wav"

    try:
        text = xfyun_stt(audio_bytes, audio_format)
        return jsonify({"text": text})
    except NotImplementedError as e:
        return jsonify({"error": str(e), "placeholder": True}), 501
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Undo (return to previous question)
# ---------------------------------------------------------------------------


@app.route("/session/<int:session_id>/undo", methods=["POST"])
def session_undo(session_id: int):
    """Delete messages from the last answered step onwards, return previous answer."""
    ds = _get_decision_session(session_id)
    if ds.status != "collecting":
        return jsonify({"error": "Cannot undo after collection is complete"}), 400

    # Find the last user message with a step recorded
    last_answered = (
        ConversationMessage.query
        .filter_by(session_id=session_id, role="user")
        .filter(ConversationMessage.step.isnot(None))
        .order_by(ConversationMessage.id.desc())
        .first()
    )

    if not last_answered:
        return jsonify({"error": "Nothing to undo"}), 400

    previous_content = last_answered.content
    previous_step = last_answered.step

    # Delete this message and everything after it
    ConversationMessage.query.filter(
        ConversationMessage.session_id == session_id,
        ConversationMessage.id >= last_answered.id
    ).delete()
    db.session.commit()

    return jsonify({"previous_content": previous_content, "step": previous_step})


# ---------------------------------------------------------------------------
# Admin panel
# ---------------------------------------------------------------------------


def _admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        admin_pw = os.environ.get("ADMIN_PASSWORD", "admin123")
        if password == admin_pw:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        error = "密码错误"
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


@app.route("/admin", methods=["GET", "POST"])
@_admin_required
def admin_dashboard():
    message = None
    if request.method == "POST":
        for key in ["primary_api_key", "primary_base_url", "backup_api_key", "backup_base_url"]:
            value = request.form.get(key, "").strip()
            if value:
                APIConfig.set(key, value)
        message = "保存成功"

    config = {
        "primary_api_key": APIConfig.get("primary_api_key", ""),
        "primary_base_url": APIConfig.get("primary_base_url", ""),
        "backup_api_key": APIConfig.get("backup_api_key", ""),
        "backup_base_url": APIConfig.get("backup_base_url", ""),
    }
    return render_template("admin.html", config=config, message=message)


# ---------------------------------------------------------------------------
# Step feedback
# ---------------------------------------------------------------------------


@app.route("/session/<int:session_id>/feedback", methods=["POST"])
def session_feedback(session_id: int):
    _get_decision_session(session_id)  # 404 if not found
    data = request.get_json(silent=True) or {}
    step = data.get("step")
    rating = data.get("rating")
    comment = (data.get("comment") or "").strip()

    if step not in (2, 3, 4, 5) or rating not in (1, 2, 3, 4, 5):
        return jsonify({"error": "Invalid step or rating"}), 400

    # Upsert: one feedback per session per step
    existing = StepFeedback.query.filter_by(session_id=session_id, step=step).first()
    if existing:
        existing.rating = rating
        existing.comment = comment
    else:
        db.session.add(StepFeedback(
            session_id=session_id, step=step, rating=rating, comment=comment
        ))
    db.session.commit()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Customer support
# ---------------------------------------------------------------------------


@app.route("/support", methods=["GET", "POST"])
def support():
    uid = _get_or_create_user_session_id()
    ticket = None
    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        contact = (request.form.get("contact") or "").strip()
        if question:
            t = SupportTicket(user_session_id=uid, question=question, contact=contact)
            db.session.add(t)
            db.session.commit()
            ticket = t
    # Show latest ticket for this user
    latest = (SupportTicket.query
              .filter_by(user_session_id=uid)
              .order_by(SupportTicket.created_at.desc())
              .first())
    return render_template("support.html", submitted_ticket=ticket, latest=latest)


# ---------------------------------------------------------------------------
# Admin: support + feedback
# ---------------------------------------------------------------------------


@app.route("/admin/support", methods=["GET", "POST"])
@_admin_required
def admin_support():
    if request.method == "POST":
        ticket_id = request.form.get("ticket_id", type=int)
        reply_text = (request.form.get("reply") or "").strip()
        if ticket_id and reply_text:
            t = SupportTicket.query.get_or_404(ticket_id)
            t.reply = reply_text
            t.status = "replied"
            from datetime import datetime
            t.replied_at = datetime.utcnow()
            db.session.commit()

    tickets = SupportTicket.query.order_by(SupportTicket.created_at.desc()).all()
    return render_template("admin_support.html", tickets=tickets)


@app.route("/admin/feedback")
@_admin_required
def admin_feedback():
    feedbacks = (StepFeedback.query
                 .order_by(StepFeedback.created_at.desc())
                 .all())
    # Compute average per step
    from collections import defaultdict
    step_stats = defaultdict(lambda: {"count": 0, "total": 0, "comments": []})
    for f in feedbacks:
        step_stats[f.step]["count"] += 1
        step_stats[f.step]["total"] += f.rating
        if f.comment:
            step_stats[f.step]["comments"].append(f.comment)
    averages = {
        step: round(v["total"] / v["count"], 1) if v["count"] else 0
        for step, v in step_stats.items()
    }
    return render_template("admin_feedback.html", feedbacks=feedbacks,
                           step_stats=step_stats, averages=averages)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
