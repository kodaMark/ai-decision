from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class APIConfig(db.Model):
    __tablename__ = "api_configs"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @staticmethod
    def get(key, default=None):
        row = APIConfig.query.filter_by(key=key).first()
        return row.value if row else default

    @staticmethod
    def set(key, value):
        row = APIConfig.query.filter_by(key=key).first()
        if row:
            row.value = value
        else:
            row = APIConfig(key=key, value=value)
            db.session.add(row)
        db.session.commit()


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


class SupportTicket(db.Model):
    __tablename__ = "support_tickets"

    id = db.Column(db.Integer, primary_key=True)
    user_session_id = db.Column(db.String(64), nullable=True)
    contact = db.Column(db.String(128), nullable=True)   # 微信/邮箱
    question = db.Column(db.Text, nullable=False)
    reply = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default="open")    # open / replied
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    replied_at = db.Column(db.DateTime, nullable=True)


class StepFeedback(db.Model):
    __tablename__ = "step_feedbacks"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("decision_sessions.id"), nullable=False)
    step = db.Column(db.Integer, nullable=False)         # 2, 3, 4, 5
    rating = db.Column(db.Integer, nullable=False)       # 1-5
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


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
