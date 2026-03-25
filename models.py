from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class DecisionSession(db.Model):
    __tablename__ = "decision_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_session_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    status = db.Column(
        db.String(20), nullable=False, default="collecting"
    )  # collecting / analyzing / done / error
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = db.relationship(
        "ConversationMessage", back_populates="session", cascade="all, delete-orphan"
    )
    report = db.relationship(
        "DecisionReport", back_populates="session", uselist=False, cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_session_id": self.user_session_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class ConversationMessage(db.Model):
    __tablename__ = "conversation_messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("decision_sessions.id"), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # user / assistant
    content = db.Column(db.Text, nullable=False)
    step = db.Column(db.Integer, nullable=True)  # which SOP question step (1-9)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship("DecisionSession", back_populates="messages")

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "step": self.step,
            "created_at": self.created_at.isoformat(),
        }


class DecisionReport(db.Model):
    __tablename__ = "decision_reports"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("decision_sessions.id"), nullable=False)
    step1_raw = db.Column(db.Text, nullable=True)   # collected info summary
    step2_raw = db.Column(db.Text, nullable=True)   # Claude structured analysis JSON
    step3_raw = db.Column(db.Text, nullable=True)   # GPT three pathways JSON
    step4_raw = db.Column(db.Text, nullable=True)   # Claude scoring matrix JSON
    step5_raw = db.Column(db.Text, nullable=True)   # Claude final recommendation JSON
    final_report_html = db.Column(db.Text, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship("DecisionSession", back_populates="report")
