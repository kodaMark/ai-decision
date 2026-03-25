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

from models import ConversationMessage, DecisionReport, DecisionSession, db
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
        full_response = []
        yield "data: " + json.dumps({"type": "start"}) + "\n\n"

        chunk_queue: queue.Queue = queue.Queue()

        def fetch_stream():
            try:
                for chunk in chat_with_glm_stream(glm_messages):
                    chunk_queue.put(("chunk", chunk))
            except Exception as e:
                chunk_queue.put(("error", str(e)))
            finally:
                chunk_queue.put(("done", None))

        threading.Thread(target=fetch_stream, daemon=True).start()

        error_occurred = False
        while True:
            try:
                item_type, item_value = chunk_queue.get(timeout=10)
            except queue.Empty:
                yield ": ping\n\n"  # keepalive，防止 Railway 代理30秒空闲断连
                continue

            if item_type == "chunk":
                full_response.append(item_value)
                display_chunk = item_value.replace("[NEXT]", "").lstrip()
                if display_chunk:
                    yield "data: " + json.dumps({"type": "chunk", "content": display_chunk}) + "\n\n"
            elif item_type == "error":
                yield "data: " + json.dumps({"type": "error", "message": item_value}) + "\n\n"
                error_occurred = True
                break
            elif item_type == "done":
                break

        if error_occurred:
            return

        assistant_text = "".join(full_response)
        advanced = "[NEXT]" in assistant_text
        clean_text = assistant_text.replace("[NEXT]", "").lstrip()

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)
