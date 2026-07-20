// MTG Judge Demo — frontend logic
// Vanilla JS, no frameworks. WebRTC mic capture + fetch API.

const API_URL_INPUT = document.getElementById("api-url");
const micBtn = document.getElementById("mic-btn");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer-text");
const confidenceBadge = document.getElementById("confidence-badge");
const queryTypeEl = document.getElementById("query-type");
const citationsEl = document.getElementById("citations");
const chunksEl = document.getElementById("chunks");
const errorEl = document.getElementById("error");
const textInput = document.getElementById("text-question");
const textSubmit = document.getElementById("text-submit");
const connectionStatus = document.getElementById("connection-status");

function getApiUrl() {
  const val = API_URL_INPUT.value.trim();
  return val || window.location.origin;
}

// Check backend connection
async function checkHealth() {
  try {
    const resp = await fetch(`${getApiUrl()}/api/health`);
    if (resp.ok) {
      connectionStatus.className = "status-dot connected";
      connectionStatus.title = "Connected";
    } else {
      throw new Error("Bad status");
    }
  } catch {
    connectionStatus.className = "status-dot disconnected";
    connectionStatus.title = "Disconnected — enter backend URL or start the server";
  }
}

checkHealth();
API_URL_INPUT.addEventListener("change", checkHealth);

// --- Text Q&A ---
textSubmit.addEventListener("click", () => askQuestion(textInput.value));
textInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") askQuestion(textInput.value);
});

async function askQuestion(question) {
  if (!question.trim()) return;
  errorEl.classList.add("hidden");
  answerSection.classList.add("hidden");
  transcriptEl.textContent = question;
  textInput.value = "";
  statusEl.textContent = "⚖️ Judging...";

  try {
    const resp = await fetch(`${getApiUrl()}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (resp.status === 429) {
      const data = await resp.json();
      showError(`${data.detail}`);
      return;
    }

    if (!resp.ok) throw new Error(`API error: ${resp.status}`);

    const result = await resp.json();
    displayAnswer(result);
    statusEl.textContent = "Done";

    // Try to speak the answer (may not be available)
    try {
      speakAnswer(result.answer);
    } catch {
      // TTS not available — that's fine
    }
  } catch (e) {
    showError(`Couldn't reach the judge: ${e.message}`);
  }
}

function displayAnswer(result) {
  answerText.textContent = result.answer;
  confidenceBadge.textContent = result.confidence;
  confidenceBadge.className = `confidence-badge ${result.confidence}`;
  queryTypeEl.textContent = result.query_type;

  citationsEl.innerHTML = (result.citations || [])
    .map((c) => `<span class="citation">${c}</span>`)
    .join("");

  chunksEl.innerHTML = (result.retrieved_chunks || [])
    .map(
      (c) =>
        `<div class="chunk"><span class="chunk-source">${c.source} ${c.section}</span>${c.text}</div>`
    )
    .join("");

  answerSection.classList.remove("hidden");
}

async function speakAnswer(text) {
  const resp = await fetch(`${getApiUrl()}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: text,
  });
  if (!resp.ok) return;
  const blob = await resp.blob();
  const audio = new Audio(URL.createObjectURL(blob));
  audio.play();
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
  statusEl.textContent = "Click and speak, or type below";
}

// --- Voice (WebRTC mic capture) ---
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

micBtn.addEventListener("pointerdown", startRecording);
micBtn.addEventListener("pointerup", stopRecording);
micBtn.addEventListener("pointerleave", stopRecording);

async function startRecording() {
  if (isRecording) return;
  isRecording = true;
  micBtn.classList.add("recording");
  statusEl.textContent = "🎤 Listening...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1 },
    });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);

    mediaRecorder.start();
  } catch (e) {
    showError(`Microphone access denied: ${e.message}`);
    isRecording = false;
    micBtn.classList.remove("recording");
  }
}

async function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  micBtn.classList.remove("recording");

  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach((t) => t.stop());

  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    statusEl.textContent = "⚖️ Transcribing...";

    try {
      // 1. Transcribe
      const formData = new FormData();
      formData.append("audio", blob, "question.webm");
      const sttResp = await fetch(`${getApiUrl()}/api/transcribe`, {
        method: "POST",
        body: formData,
      });

      if (sttResp.status === 429) {
        const data = await sttResp.json();
        showError(data.detail);
        return;
      }

      if (sttResp.status === 501) {
        // STT not available — use browser's Web Speech API as fallback
        showError("Voice input not available. Type your question instead.");
        return;
      }

      const { text } = await sttResp.json();
      transcriptEl.textContent = text;

      // 2. Ask
      await askQuestion(text);
    } catch (e) {
      showError(`Voice processing failed: ${e.message}`);
    }
  };
}
