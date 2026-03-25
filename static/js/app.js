/* ============================================================
   AI深度决策 — Frontend Application
   Voice recording + Text input + SSE streaming + Analysis polling
   ============================================================ */

(function () {
  "use strict";

  /* ---------- State ---------- */
  const state = {
    sessionId: null,
    inputMode: "text",       // "text" | "voice"
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    voiceText: "",
    isSending: false,
    isAnalyzing: false,
    collectionComplete: false,
    currentStep: 1,
    totalSteps: 9,
    answeredSteps: [],
  };

  /* ---------- DOM refs (session page) ---------- */
  const $chatArea        = document.getElementById("chatArea");
  const $textToggle      = document.getElementById("textToggle");
  const $voiceToggle     = document.getElementById("voiceToggle");
  const $textInputPanel  = document.getElementById("textInputPanel");
  const $voiceInputPanel = document.getElementById("voiceInputPanel");
  const $textarea        = document.getElementById("msgTextarea");
  const $sendBtn         = document.getElementById("sendBtn");
  const $recordBtn       = document.getElementById("recordBtn");
  const $voiceStatus     = document.getElementById("voiceStatus");
  const $voicePreview    = document.getElementById("voicePreview");
  const $sendVoiceBtn    = document.getElementById("sendVoiceBtn");
  const $analyzingOverlay= document.getElementById("analyzingOverlay");
  const $progressBar     = document.getElementById("progressBar");
  const $stepDots        = document.querySelectorAll(".step-dot");
  const $statusText      = document.getElementById("statusText");

  /* ---------- Init ---------- */
  function init() {
    const dataEl = document.getElementById("appData");
    if (!dataEl) return; // not session page

    state.sessionId = parseInt(dataEl.dataset.sessionId, 10);

    // Parse initial answered steps from server-rendered messages
    const initialMessages = JSON.parse(dataEl.dataset.messages || "[]");
    initialMessages.forEach((m) => {
      if (m.role === "user" && m.step != null) {
        if (!state.answeredSteps.includes(m.step)) {
          state.answeredSteps.push(m.step);
        }
      }
    });
    updateStepProgress();

    // If we have no messages yet, get the first GLM message
    if (initialMessages.length === 0) {
      fetchFirstMessage();
    }

    bindEvents();
    scrollToBottom();
  }

  function bindEvents() {
    if ($textToggle)     $textToggle.addEventListener("click",  () => setInputMode("text"));
    if ($voiceToggle)    $voiceToggle.addEventListener("click", () => setInputMode("voice"));
    if ($sendBtn)        $sendBtn.addEventListener("click",     sendTextMessage);
    if ($textarea)       $textarea.addEventListener("keydown",  onTextareaKeydown);
    if ($textarea)       $textarea.addEventListener("input",    autoResizeTextarea);
    if ($recordBtn)      $recordBtn.addEventListener("click",   toggleRecording);
    if ($sendVoiceBtn)   $sendVoiceBtn.addEventListener("click", sendVoiceMessage);
  }

  /* ---------- Input mode toggle ---------- */
  function setInputMode(mode) {
    state.inputMode = mode;
    if (mode === "text") {
      $textToggle?.classList.add("active");
      $voiceToggle?.classList.remove("active");
      $textInputPanel?.classList.remove("hidden");
      $voiceInputPanel?.classList.add("hidden");
    } else {
      $voiceToggle?.classList.add("active");
      $textToggle?.classList.remove("active");
      $voiceInputPanel?.classList.remove("hidden");
      $textInputPanel?.classList.add("hidden");
    }
  }

  /* ---------- Auto-resize textarea ---------- */
  function autoResizeTextarea() {
    if (!$textarea) return;
    $textarea.style.height = "auto";
    $textarea.style.height = Math.min($textarea.scrollHeight, 120) + "px";
  }

  function onTextareaKeydown(e) {
    // Send on Enter (not Shift+Enter)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendTextMessage();
    }
  }

  /* ---------- First message (cold start) ---------- */
  function fetchFirstMessage() {
    // Show hardcoded opening to ensure consistent single-question format
    appendMessage("assistant", "你好！我是小决，专门帮你理清重要决定。\n\n你在考虑什么决定？用一句话告诉我就好。");
  }

  /* ---------- Send text message ---------- */
  function sendTextMessage() {
    if (!$textarea) return;
    const text = $textarea.value.trim();
    if (!text || state.isSending) return;
    $textarea.value = "";
    autoResizeTextarea();
    sendMessage(text, true);
  }

  /* ---------- Core send function ---------- */
  function sendMessage(text, showUserBubble = true) {
    if (state.isSending || !state.sessionId) return;
    state.isSending = true;
    setSendEnabled(false);

    if (showUserBubble) {
      appendMessage("user", text);
    }

    // Show typing indicator
    const typingId = showTyping();

    const evtSource = new EventSource(
      `/session/${state.sessionId}/message?` +
        new URLSearchParams({ _t: Date.now() })
    );

    // We need to POST, not GET — use fetch + ReadableStream for SSE
    evtSource.close(); // close the dummy one

    fetchSSE(`/session/${state.sessionId}/message`, { content: text })
      .then((reader) => {
        removeTyping(typingId);
        const bubbleId = startAssistantBubble();
        let fullText = "";

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) {
              onMessageComplete();
              return;
            }
            const lines = value.split("\n");
            lines.forEach((line) => {
              if (!line.startsWith("data: ")) return;
              const raw = line.slice(6).trim();
              if (!raw) return;
              try {
                const data = JSON.parse(raw);
                if (data.type === "chunk") {
                  fullText += data.content;
                  updateAssistantBubble(bubbleId, fullText);
                  scrollToBottom();
                } else if (data.type === "done") {
                  if (data.collection_complete) {
                    state.collectionComplete = true;
                    setTimeout(triggerAnalysis, 1500);
                  }
                } else if (data.type === "error") {
                  appendMessage("assistant", "抱歉，出现了一点问题，请重试。");
                }
              } catch (_) {}
            });
            read();
          });
        }
        read();
      })
      .catch((err) => {
        removeTyping(typingId);
        appendMessage("assistant", "网络连接出现问题，请检查后重试。");
        console.error(err);
        onMessageComplete();
      });
  }

  function onMessageComplete() {
    state.isSending = false;
    setSendEnabled(true);
    scrollToBottom();
  }

  /* ---------- Fetch SSE via POST ---------- */
  function fetchSSE(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      // Wrap reader to decode bytes → string
      return {
        read() {
          return reader.read().then(({ done, value }) => ({
            done,
            value: done ? "" : decoder.decode(value, { stream: true }),
          }));
        },
      };
    });
  }

  /* ---------- Chat UI helpers ---------- */
  function appendMessage(role, content) {
    if (!$chatArea) return;
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    const avatarText = role === "assistant" ? "✦" : "你";
    div.innerHTML = `
      <div class="msg-avatar">${avatarText}</div>
      <div class="msg-bubble">${escHtml(content)}</div>
    `;
    $chatArea.appendChild(div);
    scrollToBottom();
    return div;
  }

  let _bubbleCounter = 0;

  function startAssistantBubble() {
    const id = "bubble-" + (++_bubbleCounter);
    const div = document.createElement("div");
    div.className = "msg assistant";
    div.id = id;
    div.innerHTML = `
      <div class="msg-avatar">✦</div>
      <div class="msg-bubble"></div>
    `;
    $chatArea?.appendChild(div);
    return id;
  }

  function updateAssistantBubble(id, text) {
    const el = document.getElementById(id);
    if (!el) return;
    const bubble = el.querySelector(".msg-bubble");
    if (bubble) bubble.textContent = text;
  }

  function showTyping() {
    const id = "typing-" + Date.now();
    const div = document.createElement("div");
    div.className = "msg assistant";
    div.id = id;
    div.innerHTML = `
      <div class="msg-avatar">✦</div>
      <div class="msg-bubble">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    $chatArea?.appendChild(div);
    scrollToBottom();
    return id;
  }

  function removeTyping(id) {
    document.getElementById(id)?.remove();
  }

  function scrollToBottom() {
    if ($chatArea) {
      $chatArea.scrollTop = $chatArea.scrollHeight;
    }
  }

  function setSendEnabled(enabled) {
    if ($sendBtn) $sendBtn.disabled = !enabled;
    if ($textarea) $textarea.disabled = !enabled;
  }

  /* ---------- Step progress ---------- */
  function updateStepProgress(step) {
    if (step && !state.answeredSteps.includes(step)) {
      state.answeredSteps.push(step);
    }
    if (!$stepDots || !$stepDots.length) return;
    $stepDots.forEach((dot, i) => {
      const stepNum = i + 1;
      dot.classList.remove("answered", "current");
      if (state.answeredSteps.includes(stepNum)) {
        dot.classList.add("answered");
      } else if (
        stepNum === Math.max(...state.answeredSteps, 0) + 1 &&
        stepNum <= state.totalSteps
      ) {
        dot.classList.add("current");
      }
    });
    if ($statusText) {
      const answered = state.answeredSteps.length;
      $statusText.textContent =
        answered < state.totalSteps
          ? `收集中 ${answered}/${state.totalSteps}`
          : "信息收集完毕，分析中…";
    }
  }

  /* ---------- Analysis trigger + polling ---------- */
  function triggerAnalysis() {
    if (state.isAnalyzing) return;
    state.isAnalyzing = true;
    showAnalyzingOverlay();

    fetch(`/session/${state.sessionId}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.redirect) {
          window.location.href = data.redirect;
        } else {
          pollStatus();
        }
      })
      .catch(() => pollStatus());
  }

  function pollStatus() {
    let attempts = 0;
    const maxAttempts = 300; // 5 min max

    const interval = setInterval(() => {
      attempts++;
      if (attempts > maxAttempts) {
        clearInterval(interval);
        showToast("分析超时，请刷新页面重试");
        return;
      }

      // Animate progress bar
      const pct = Math.min(5 + (attempts / maxAttempts) * 90, 95);
      if ($progressBar) $progressBar.style.width = pct + "%";

      // Animate analyzing steps
      animateAnalyzingSteps(attempts);

      fetch(`/session/${state.sessionId}/status`)
        .then((r) => r.json())
        .then((data) => {
          if (data.status === "done") {
            clearInterval(interval);
            if ($progressBar) $progressBar.style.width = "100%";
            setTimeout(() => {
              window.location.href = data.redirect;
            }, 600);
          } else if (data.status === "error") {
            clearInterval(interval);
            hideAnalyzingOverlay();
            showToast("分析出错：" + (data.error || "未知错误"));
          }
        })
        .catch(() => {});
    }, 1000);
  }

  const ANALYZING_STEPS = [
    { label: "整理决策信息…", threshold: 0 },
    { label: "Claude 结构化分析…", threshold: 8 },
    { label: "GPT-4.5 推演三条路径…", threshold: 25 },
    { label: "Claude 量化评分矩阵…", threshold: 50 },
    { label: "生成最终建议报告…", threshold: 75 },
  ];

  function animateAnalyzingSteps(attempts) {
    const els = document.querySelectorAll(".analyzing-step");
    if (!els.length) return;
    els.forEach((el, i) => {
      const step = ANALYZING_STEPS[i];
      if (!step) return;
      if (attempts >= step.threshold + (ANALYZING_STEPS[i + 1]?.threshold || 999)) {
        el.classList.remove("active");
        el.classList.add("done");
        el.querySelector(".step-icon") && (el.querySelector(".step-icon").textContent = "✓");
      } else if (attempts >= step.threshold) {
        el.classList.add("active");
        el.classList.remove("done");
      }
    });
  }

  function showAnalyzingOverlay() {
    if ($analyzingOverlay) $analyzingOverlay.classList.remove("hidden");
  }

  function hideAnalyzingOverlay() {
    if ($analyzingOverlay) $analyzingOverlay.classList.add("hidden");
  }

  /* ---------- Voice recording ---------- */
  function toggleRecording() {
    if (state.isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }

  function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      showToast("你的浏览器不支持录音功能");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        state.audioChunks = [];
        state.mediaRecorder = new MediaRecorder(stream);
        state.mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) state.audioChunks.push(e.data);
        };
        state.mediaRecorder.onstop = onRecordingStop;
        state.mediaRecorder.start();
        state.isRecording = true;
        if ($recordBtn) {
          $recordBtn.classList.add("recording");
          $recordBtn.textContent = "⏹";
        }
        if ($voiceStatus) $voiceStatus.textContent = "录音中… 点击停止";
      })
      .catch((err) => {
        showToast("无法访问麦克风：" + err.message);
      });
  }

  function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
      state.mediaRecorder.stop();
      state.mediaRecorder.stream.getTracks().forEach((t) => t.stop());
      state.isRecording = false;
      if ($recordBtn) {
        $recordBtn.classList.remove("recording");
        $recordBtn.textContent = "🎙";
      }
      if ($voiceStatus) $voiceStatus.textContent = "处理中…";
    }
  }

  function onRecordingStop() {
    const blob = new Blob(state.audioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    fetch("/api/stt", { method: "POST", body: formData })
      .then((r) => r.json())
      .then((data) => {
        if (data.error && data.placeholder) {
          // STT not configured — show mock text for demo
          showToast("STT 未配置，请使用文字输入");
          if ($voiceStatus) $voiceStatus.textContent = "请切换到文字输入";
          return;
        }
        if (data.text) {
          state.voiceText = data.text;
          if ($voicePreview) {
            $voicePreview.textContent = data.text;
            $voicePreview.classList.add("has-text");
          }
          if ($voiceStatus) $voiceStatus.textContent = "识别完成，确认后发送";
          if ($sendVoiceBtn) $sendVoiceBtn.style.display = "inline-flex";
        }
      })
      .catch(() => {
        if ($voiceStatus) $voiceStatus.textContent = "识别失败，请重试";
        showToast("语音识别失败，请重试或切换文字输入");
      });
  }

  function sendVoiceMessage() {
    if (!state.voiceText.trim()) return;
    const text = state.voiceText;
    state.voiceText = "";
    if ($voicePreview) {
      $voicePreview.textContent = "";
      $voicePreview.classList.remove("has-text");
    }
    if ($sendVoiceBtn) $sendVoiceBtn.style.display = "none";
    if ($voiceStatus) $voiceStatus.textContent = "按住麦克风开始说话";
    setInputMode("text");
    sendMessage(text, true);
  }

  /* ---------- Report page ---------- */
  function initReport() {
    const shareBtn = document.getElementById("shareBtn");
    const pdfBtn   = document.getElementById("pdfBtn");

    if (shareBtn) {
      shareBtn.addEventListener("click", () => {
        const url = window.location.href;
        if (navigator.share) {
          navigator.share({ title: "AI深度决策报告", url });
        } else {
          navigator.clipboard.writeText(url).then(() => showToast("链接已复制到剪贴板"));
        }
      });
    }

    if (pdfBtn) {
      pdfBtn.addEventListener("click", () => {
        // TODO: integrate server-side PDF generation or use window.print()
        showToast("PDF 导出功能即将上线");
      });
    }
  }

  /* ---------- Toast ---------- */
  function showToast(msg, duration = 2500) {
    let toast = document.getElementById("globalToast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "globalToast";
      toast.className = "toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), duration);
  }

  /* ---------- Utility ---------- */
  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* ---------- Landing page: start session ---------- */
  function initLanding() {
    const startBtn = document.getElementById("startBtn");
    if (!startBtn) return;
    startBtn.addEventListener("click", () => {
      startBtn.disabled = true;
      startBtn.textContent = "正在准备…";
      fetch("/session/start", { method: "POST" })
        .then((r) => r.json())
        .then((data) => {
          if (data.session_id) {
            window.location.href = `/session/${data.session_id}`;
          } else {
            startBtn.disabled = false;
            startBtn.textContent = "开始深度决策";
            showToast("启动失败，请重试");
          }
        })
        .catch(() => {
          startBtn.disabled = false;
          startBtn.textContent = "开始深度决策";
          showToast("网络错误，请重试");
        });
    });
  }

  /* ---------- Entry ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    initLanding();
    init();
    initReport();
  });
})();
