"""
SOP Conversation State Machine for AI深度决策.

Tracks which of the 9 structured questions have been answered,
and generates context-aware prompts for GLM to guide the user.
"""

from dataclasses import dataclass, field
from typing import Optional


# The 9 SOP questions in order
SOP_QUESTIONS = [
    {
        "step": 1,
        "key": "decision",
        "question": "你在考虑什么决定？用一句话告诉我",
        "glm_prompt": (
            "用温暖、好奇的语气问用户：他们正在考虑什么决定。"
            "让他们用一句话描述这个决定。保持轻松，不要像问卷。"
        ),
    },
    {
        "step": 2,
        "key": "timeline",
        "question": "这个决定的时间节点是什么？",
        "glm_prompt": (
            "自然地过渡，询问这个决定的时间节点——什么时候需要做出决定，"
            "或者决定会影响多长时间？表现出真诚的好奇心。"
        ),
    },
    {
        "step": 3,
        "key": "stakeholders",
        "question": "这件事涉及哪些人？",
        "glm_prompt": (
            "温和地问：这件事涉及到哪些人？家人、同事、朋友？"
            "表达理解，这些关系可能让决定更复杂。"
        ),
    },
    {
        "step": 4,
        "key": "biggest_worry",
        "question": "你最担心的是什么？",
        "glm_prompt": (
            "带着同理心问：在这个决定里，你最担心、最害怕的是什么？"
            "让用户感到被理解和支持，而不是被审问。"
        ),
    },
    {
        "step": 5,
        "key": "best_outcome",
        "question": "你希望的最好结果是什么？",
        "glm_prompt": (
            "鼓励地问：如果一切顺利，你希望最好的结果是什么样的？"
            "帮助用户想象理想的未来场景。"
        ),
    },
    {
        "step": 6,
        "key": "worst_outcome",
        "question": "你能承受的最坏结果是什么？",
        "glm_prompt": (
            "温柔地问：如果事情没有按计划进行，你能接受的最坏情况是什么？"
            "强调这是帮助评估风险，不是预测坏事会发生。"
        ),
    },
    {
        "step": 7,
        "key": "known_info",
        "question": "你现在掌握哪些信息？",
        "glm_prompt": (
            "询问用户目前已经知道的信息和事实——关于这个决定，"
            "他们手头有哪些具体的信息、数据或情况？"
        ),
    },
    {
        "step": 8,
        "key": "uncertainties",
        "question": "还有什么你不确定的？",
        "glm_prompt": (
            "问用户：关于这个决定，还有哪些信息是他们不确定的，"
            "或者希望能知道但目前还不清楚的？"
        ),
    },
    {
        "step": 9,
        "key": "alternatives",
        "question": "有没有其他选项你考虑过？",
        "glm_prompt": (
            "最后问：除了现在考虑的这个方向，有没有其他选项或者路径"
            "你也思考过？哪怕是暂时放弃的想法也值得聊聊。"
        ),
    },
]

SYSTEM_PROMPT_GLM = """你是一位温暖、智慧的决策引导师，名字叫"小决"。

你的任务是通过自然的对话，帮助用户系统地整理他们面临的重要决定。
你不是在做问卷调查，而是在进行一次真诚的深度对话。

风格要求：
- 温暖、有共情心，像朋友一样交谈
- 简洁，不要冗长的铺垫
- 【强制规则】每次回复只能问一个问题，绝对禁止在同一条回复里出现多个问号
- 用"你"而不是"您"，保持亲切感
- 回复控制在80字以内

【关键规则 - 答案质量判断】
收到用户回答后，判断答案是否足够具体：

✅ 答案足够具体 → 在回复开头加上 [NEXT]，然后自然过渡到下一个问题
   例：[NEXT] 明白了，时间节点很清晰。那这件事涉及到哪些人呢？

❌ 答案太模糊 → 不加 [NEXT]，追问一个具体的问题
   例：能再说具体一点吗？比如大概什么时间段需要做出决定？

判断标准：
- "不知道"、"可能吧"、"感觉"、"担心失败"等 → 模糊，追问
- 包含具体数字、时间、人名、事件的 → 具体，加 [NEXT]
- 每个问题最多追问2次，第3次必须加 [NEXT] 继续推进

重要：你只负责收集信息阶段。当所有问题收集完毕后，
系统会自动进入深度分析阶段，你无需主动告知用户分析步骤。
"""


@dataclass
class ConversationSOP:
    """State machine tracking which SOP questions have been answered."""

    answers: dict = field(default_factory=dict)
    current_step: int = 1  # 1-indexed, matches SOP_QUESTIONS step

    def get_current_question(self) -> Optional[dict]:
        """Return the current unanswered question spec."""
        for q in SOP_QUESTIONS:
            if q["key"] not in self.answers:
                return q
        return None  # all answered

    def record_answer(self, step: int, answer: str) -> None:
        """Record an answer for a given step (1-9)."""
        for q in SOP_QUESTIONS:
            if q["step"] == step:
                self.answers[q["key"]] = answer
                self.current_step = step + 1
                return

    def is_collection_complete(self) -> bool:
        """Return True when all 9 answers have been collected."""
        return all(q["key"] in self.answers for q in SOP_QUESTIONS)

    def extract_collected_info(self) -> str:
        """Format all answers into a structured summary for Claude/GPT."""
        lines = ["## 用户决策信息收集结果\n"]
        labels = {
            "decision": "核心决定",
            "timeline": "时间节点",
            "stakeholders": "涉及人员",
            "biggest_worry": "最大担忧",
            "best_outcome": "理想结果",
            "worst_outcome": "可接受的最坏结果",
            "known_info": "已掌握的信息",
            "uncertainties": "不确定的信息",
            "alternatives": "已考虑的其他选项",
        }
        for q in SOP_QUESTIONS:
            key = q["key"]
            label = labels.get(key, key)
            answer = self.answers.get(key, "（未回答）")
            lines.append(f"**{label}**: {answer}")
        return "\n".join(lines)

    def build_glm_messages(self, conversation_history: list, followup_count: int = 0) -> list:
        """
        Build the message list for GLM API call.
        conversation_history: list of {"role": "user"/"assistant", "content": str}
        followup_count: how many times we've already followed up on the current question
        """
        current_q = self.get_current_question()
        if current_q is None:
            extra_instruction = (
                "用户已经回答了所有问题。请用温暖的语气告诉用户："
                "信息收集完毕，系统将为他们进行深度推演分析，大约需要30秒，请稍候。"
                "语气要充满期待感，让用户感到即将收到有价值的洞见。"
            )
        else:
            force_next = followup_count >= 2 or current_q["step"] == 9
            extra_instruction = (
                f"当前需要引导用户回答的问题方向：{current_q['glm_prompt']}\n"
                f"这是第{current_q['step']}个问题（共9个）。\n"
                f"当前已对此问题追问了 {followup_count} 次。\n"
            )
            if force_next:
                extra_instruction += (
                    "已达到最大追问次数，无论用户回答是否具体，"
                    "都必须在回复开头加 [NEXT] 并推进到下一个问题。\n"
                )
            if not conversation_history:
                extra_instruction += "这是第一条消息，先热情地打招呼，然后问第一个问题。"
            else:
                extra_instruction += "根据答案质量决定是否加 [NEXT]，参考系统提示中的判断标准。\n【本条强制】你的回复里只能出现一个问号，违反此规则视为错误输出。"

        system = SYSTEM_PROMPT_GLM + "\n\n[当前引导指令]\n" + extra_instruction

        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history)
        return messages

    @classmethod
    def from_messages(cls, messages: list) -> "ConversationSOP":
        """
        Reconstruct SOP state from stored ConversationMessage records.
        messages: list of ConversationMessage ORM objects (ordered by created_at).
        """
        sop = cls()
        for msg in messages:
            if msg.role == "user" and msg.step is not None:
                sop.record_answer(msg.step, msg.content)
        return sop
