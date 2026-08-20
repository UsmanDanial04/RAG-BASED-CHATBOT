// Gemini Echo — RAG Chatbot Logic

// ---------------------------------------------------------------------------
// Global State
// ---------------------------------------------------------------------------
let chatHistory = []; // { role: 'user'|'model', text: string }
let isRecording = false;
let currentSpeakingText = null;
let currentPlayButton = null;

// Speech APIs
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
const synthesis = window.speechSynthesis;
let voices = [];

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const chatMessages        = document.getElementById('chat-messages');
const chatInput           = document.getElementById('chat-input');
const sendBtn             = document.getElementById('send-btn');
const micBtn              = document.getElementById('mic-btn');
const waveformContainer   = document.getElementById('waveform-container');
const toggleSettingsBtn   = document.getElementById('toggle-settings-btn');
const closeSettingsBtn    = document.getElementById('close-settings-btn');
const settingsPanel       = document.getElementById('settings-panel');
const apiKeyInput         = document.getElementById('api-key-input');
const toggleKeyVisibility = document.getElementById('toggle-key-visibility');
const modelSelect         = document.getElementById('model-select');
const systemInstruction   = document.getElementById('system-instruction');
const voiceSelect         = document.getElementById('voice-select');
const autoSpeakToggle     = document.getElementById('auto-speak-toggle');
const voiceRate           = document.getElementById('voice-rate');
const voicePitch          = document.getElementById('voice-pitch');
const rateValue           = document.getElementById('rate-value');
const pitchValue          = document.getElementById('pitch-value');
const clearChatBtn        = document.getElementById('clear-chat-btn');
const saveSettingsBtn     = document.getElementById('save-settings-btn');
const statusDot           = document.getElementById('status-dot');
const statusText          = document.getElementById('status-text');

// RAG-specific elements
const ragToggle           = document.getElementById('rag-toggle');
const ragBadge            = document.getElementById('rag-badge');
const ragDocCount         = document.getElementById('rag-doc-count');
const uploadZone          = document.getElementById('upload-zone');
const fileInput           = document.getElementById('file-input');
const uploadProgress      = document.getElementById('upload-progress');
const progressBar         = document.getElementById('progress-bar');
const uploadStatusText    = document.getElementById('upload-status-text');
const docList             = document.getElementById('doc-list');
const docListEmpty        = document.getElementById('doc-list-empty');
const docCountLabel       = document.getElementById('doc-count-label');

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  loadSettings();
  initSpeechRecognition();
  initSpeechSynthesis();
  registerEventListeners();
  registerRagEventListeners();
  loadDocumentsList();   // Fetch existing docs from backend on startup
  chatInput.focus();
});

// ---------------------------------------------------------------------------
// Settings — Load / Save
// ---------------------------------------------------------------------------
function loadSettings() {
  const savedKey = localStorage.getItem('gemini_api_key');
  if (savedKey) apiKeyInput.value = savedKey;

  const savedModel = localStorage.getItem('gemini_model');
  if (savedModel) modelSelect.value = savedModel;

  const savedPrompt = localStorage.getItem('gemini_system_instruction');
  if (savedPrompt) systemInstruction.value = savedPrompt;

  const savedAutoSpeak = localStorage.getItem('gemini_auto_speak');
  if (savedAutoSpeak !== null) autoSpeakToggle.checked = savedAutoSpeak === 'true';

  const savedRate = localStorage.getItem('gemini_voice_rate');
  if (savedRate) {
    voiceRate.value = savedRate;
    rateValue.textContent = `${savedRate}x`;
  }

  const savedPitch = localStorage.getItem('gemini_voice_pitch');
  if (savedPitch) {
    voicePitch.value = savedPitch;
    pitchValue.textContent = savedPitch;
  }

  const savedRag = localStorage.getItem('gemini_use_rag');
  if (savedRag !== null) ragToggle.checked = savedRag === 'true';
}

function saveSettings() {
  localStorage.setItem('gemini_api_key', apiKeyInput.value.trim());
  localStorage.setItem('gemini_model', modelSelect.value);
  localStorage.setItem('gemini_system_instruction', systemInstruction.value.trim());
  localStorage.setItem('gemini_auto_speak', autoSpeakToggle.checked);
  localStorage.setItem('gemini_voice_rate', voiceRate.value);
  localStorage.setItem('gemini_voice_pitch', voicePitch.value);
  localStorage.setItem('gemini_voice_name', voiceSelect.value);
  localStorage.setItem('gemini_use_rag', ragToggle.checked);

  const originalContent = saveSettingsBtn.innerHTML;
  saveSettingsBtn.innerHTML = '<i data-lucide="check"></i> Settings Saved!';
  saveSettingsBtn.style.background = '#10b981';
  lucide.createIcons();

  setTimeout(() => {
    saveSettingsBtn.innerHTML = originalContent;
    saveSettingsBtn.style.background = '';
    lucide.createIcons();
    settingsPanel.classList.add('hidden');
  }, 1200);
}

// ---------------------------------------------------------------------------
// Event Listeners — Chat & UI
// ---------------------------------------------------------------------------
function registerEventListeners() {
  sendBtn.addEventListener('click', () => sendMessage(chatInput.value.trim()));

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(chatInput.value.trim());
    }
  });

  chatInput.addEventListener('input', autoGrowTextarea);
  micBtn.addEventListener('click', toggleRecording);

  toggleSettingsBtn.addEventListener('click', () => {
    settingsPanel.classList.toggle('hidden');
  });

  closeSettingsBtn.addEventListener('click', () => {
    settingsPanel.classList.add('hidden');
  });

  toggleKeyVisibility.addEventListener('click', () => {
    const isPassword = apiKeyInput.type === 'password';
    apiKeyInput.type = isPassword ? 'text' : 'password';
    toggleKeyVisibility.innerHTML = isPassword
      ? '<i data-lucide="eye-off"></i>'
      : '<i data-lucide="eye"></i>';
    lucide.createIcons();
  });

  voiceRate.addEventListener('input', () => {
    rateValue.textContent = `${voiceRate.value}x`;
  });

  voicePitch.addEventListener('input', () => {
    pitchValue.textContent = voicePitch.value;
  });

  clearChatBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear the conversation?')) {
      chatHistory = [];
      chatMessages.innerHTML = `
        <div class="message assistant">
          <div class="message-avatar"><i data-lucide="bot"></i></div>
          <div class="message-content-wrapper">
            <div class="message-content">
              <p>Chat cleared. Ask me anything or upload documents to get started!</p>
            </div>
            <div class="message-meta"><span>System</span></div>
          </div>
        </div>
      `;
      lucide.createIcons();
      stopSpeaking();
    }
  });

  saveSettingsBtn.addEventListener('click', saveSettings);
}

// ---------------------------------------------------------------------------
// RAG — Event Listeners
// ---------------------------------------------------------------------------
function registerRagEventListeners() {
  // Clicking the upload zone opens file dialog
  uploadZone.addEventListener('click', () => fileInput.click());

  // File input change
  fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    files.forEach(uploadFile);
    fileInput.value = ''; // Reset so same file can be re-uploaded
  });

  // Drag-and-drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    files.forEach(uploadFile);
  });
}

// ---------------------------------------------------------------------------
// RAG — Document Upload
// ---------------------------------------------------------------------------
async function uploadFile(file) {
  const allowedExts = ['.pdf', '.txt', '.md', '.docx'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowedExts.includes(ext)) {
    showSystemMessage(`Unsupported file type: "${file.name}". Allowed: PDF, TXT, MD, DOCX.`);
    return;
  }

  // Show progress
  uploadProgress.classList.remove('hidden');
  uploadStatusText.textContent = `Uploading "${file.name}"…`;
  progressBar.style.width = '0%';

  // Animate progress bar (indeterminate-style)
  let pct = 0;
  const interval = setInterval(() => {
    pct = Math.min(pct + Math.random() * 12, 88);
    progressBar.style.width = `${pct}%`;
  }, 200);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/documents/upload', {
      method: 'POST',
      body: formData,
    });

    clearInterval(interval);
    progressBar.style.width = '100%';

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const doc = await res.json();
    uploadStatusText.textContent = `✓ "${doc.filename}" indexed (${doc.chunk_count} chunks)`;

    setTimeout(() => {
      uploadProgress.classList.add('hidden');
    }, 2500);

    // Refresh document list
    await loadDocumentsList();

  } catch (err) {
    clearInterval(interval);
    progressBar.style.width = '0%';
    uploadStatusText.textContent = `✗ ${err.message}`;
    uploadProgress.classList.remove('hidden');
    setTimeout(() => uploadProgress.classList.add('hidden'), 4000);
  }
}

// ---------------------------------------------------------------------------
// RAG — Load Documents List
// ---------------------------------------------------------------------------
async function loadDocumentsList() {
  try {
    const res = await fetch('/api/documents');
    if (!res.ok) return;
    const data = await res.json();
    renderDocumentList(data.documents);
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

function renderDocumentList(documents) {
  const count = documents.length;
  docCountLabel.textContent = count;

  // Update RAG badge in header
  if (count > 0) {
    ragDocCount.textContent = count;
    ragBadge.classList.remove('hidden');
  } else {
    ragBadge.classList.add('hidden');
  }

  // Clear existing items (keep the empty state div)
  const existingItems = docList.querySelectorAll('.doc-item');
  existingItems.forEach(el => el.remove());

  if (count === 0) {
    docListEmpty.classList.remove('hidden');
    return;
  }

  docListEmpty.classList.add('hidden');

  documents.forEach(doc => {
    const item = createDocItem(doc);
    docList.appendChild(item);
  });

  lucide.createIcons();
}

function createDocItem(doc) {
  const ext = doc.filename.split('.').pop().toUpperCase();
  const sizeKb = (doc.size_bytes / 1024).toFixed(1);

  const item = document.createElement('div');
  item.className = 'doc-item';
  item.dataset.docId = doc.doc_id;

  item.innerHTML = `
    <div class="doc-item-icon">
      <i data-lucide="${getFileIcon(ext)}"></i>
    </div>
    <div class="doc-item-info">
      <span class="doc-item-name" title="${doc.filename}">${doc.filename}</span>
      <span class="doc-item-meta">${ext} · ${sizeKb} KB · ${doc.chunk_count} chunks</span>
    </div>
    <button class="doc-delete-btn" title="Remove document" data-doc-id="${doc.doc_id}">
      <i data-lucide="trash-2"></i>
    </button>
  `;

  item.querySelector('.doc-delete-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    deleteDocument(doc.doc_id, doc.filename, item);
  });

  return item;
}

function getFileIcon(ext) {
  const map = { PDF: 'file-text', TXT: 'file', MD: 'file-code', DOCX: 'file-type' };
  return map[ext] || 'file';
}

// ---------------------------------------------------------------------------
// RAG — Delete Document
// ---------------------------------------------------------------------------
async function deleteDocument(docId, filename, itemEl) {
  if (!confirm(`Remove "${filename}" from the knowledge base?`)) return;

  try {
    const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Delete failed');
    }

    // Animate removal
    itemEl.style.opacity = '0';
    itemEl.style.transform = 'translateX(10px)';
    setTimeout(async () => {
      itemEl.remove();
      await loadDocumentsList();
    }, 250);

  } catch (err) {
    showSystemMessage(`Failed to delete "${filename}": ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Textarea auto-grow
// ---------------------------------------------------------------------------
function autoGrowTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = (chatInput.scrollHeight - 4) + 'px';
}

// ---------------------------------------------------------------------------
// Speech Recognition
// ---------------------------------------------------------------------------
function initSpeechRecognition() {
  if (!SpeechRecognition) {
    console.warn('Speech Recognition not supported in this browser.');
    micBtn.title = 'Speech Recognition Not Supported';
    micBtn.disabled = true;
    micBtn.style.opacity = '0.5';
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isRecording = true;
    micBtn.classList.add('recording');
    waveformContainer.classList.add('active');
    statusDot.className = 'status-dot blue';
    statusText.textContent = 'Listening';
    stopSpeaking();
  };

  recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) finalTranscript += transcript;
      else interimTranscript += transcript;
    }
    chatInput.value = finalTranscript || interimTranscript;
    autoGrowTextarea();
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    stopRecordingUI();
    if (event.error === 'not-allowed') {
      showSystemMessage('Microphone permission was denied. Please allow microphone access in your browser settings.');
    } else if (event.error !== 'aborted') {
      showSystemMessage(`Speech recognition error: ${event.error}`);
    }
  };

  recognition.onend = () => {
    stopRecordingUI();
    const text = chatInput.value.trim();
    if (text) sendMessage(text);
  };
}

function toggleRecording() {
  if (!recognition) return;
  if (isRecording) recognition.stop();
  else {
    chatInput.value = '';
    recognition.start();
  }
}

function stopRecordingUI() {
  isRecording = false;
  micBtn.classList.remove('recording');
  waveformContainer.classList.remove('active');
  statusDot.className = 'status-dot green';
  statusText.textContent = 'Ready';
}

// ---------------------------------------------------------------------------
// Speech Synthesis
// ---------------------------------------------------------------------------
function initSpeechSynthesis() {
  if (!synthesis) {
    console.warn('Speech Synthesis not supported in this browser.');
    return;
  }

  const populateVoices = () => {
    voices = synthesis.getVoices();
    voiceSelect.innerHTML = '';
    if (voices.length === 0) {
      const opt = document.createElement('option');
      opt.textContent = 'No system voices available';
      voiceSelect.appendChild(opt);
      return;
    }
    const savedVoiceName = localStorage.getItem('gemini_voice_name');
    voices.forEach((voice) => {
      const option = document.createElement('option');
      option.value = voice.name;
      option.textContent = `${voice.name} (${voice.lang})`;
      if (savedVoiceName && voice.name === savedVoiceName) option.selected = true;
      else if (!savedVoiceName && (voice.name.includes('Google') || voice.lang.startsWith('en-US')) && voice.default) option.selected = true;
      voiceSelect.appendChild(option);
    });
  };

  populateVoices();
  if (synthesis.onvoiceschanged !== undefined) synthesis.onvoiceschanged = populateVoices;
}

function speakText(text, playButton = null) {
  if (!synthesis) return;
  if (synthesis.speaking && currentSpeakingText === text) {
    stopSpeaking();
    return;
  }
  stopSpeaking();

  const cleanText = text
    .replace(/```[\s\S]*?```/g, '[Code block omitted]')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[#*_\-`]/g, '')
    .trim();

  if (cleanText === '') return;

  const utterance = new SpeechSynthesisUtterance(cleanText);
  const voice = voices.find(v => v.name === voiceSelect.value);
  if (voice) utterance.voice = voice;
  utterance.rate = parseFloat(voiceRate.value);
  utterance.pitch = parseFloat(voicePitch.value);

  utterance.onstart = () => {
    currentSpeakingText = text;
    currentPlayButton = playButton;
    statusDot.className = 'status-dot purple';
    statusText.textContent = 'Speaking';
    if (playButton) {
      playButton.innerHTML = '<i data-lucide="volume-x"></i> Stop';
      lucide.createIcons();
    }
  };

  utterance.onend = () => resetSpeakerUI();
  utterance.onerror = (e) => { console.error('Speech synthesis error:', e); resetSpeakerUI(); };

  synthesis.speak(utterance);
}

function stopSpeaking() {
  if (synthesis && synthesis.speaking) synthesis.cancel();
  resetSpeakerUI();
}

function resetSpeakerUI() {
  statusDot.className = 'status-dot green';
  statusText.textContent = 'Ready';
  if (currentPlayButton) {
    currentPlayButton.innerHTML = '<i data-lucide="volume-2"></i> Speak';
    lucide.createIcons();
  }
  currentSpeakingText = null;
  currentPlayButton = null;
}

// ---------------------------------------------------------------------------
// Send Message Flow (RAG-augmented)
// ---------------------------------------------------------------------------
async function sendMessage(text) {
  if (!text || text.trim() === '') return;

  chatInput.value = '';
  autoGrowTextarea();

  if (isRecording && recognition) recognition.stop();
  stopSpeaking();

  // Append user message
  appendMessage('user', text);
  showTypingIndicator();

  const useRag = ragToggle.checked;

  // Show "Searching knowledge base" status when RAG enabled
  if (useRag && !ragBadge.classList.contains('hidden')) {
    statusDot.className = 'status-dot blue';
    statusText.textContent = 'Searching docs…';
  }

  const payload = {
    message: text,
    history: chatHistory,
    api_key: apiKeyInput.value.trim(),
    model: modelSelect.value,
    system_instruction: systemInstruction.value.trim(),
    use_rag: useRag,
  };

  chatHistory.push({ role: 'user', text });

  try {
    statusDot.className = 'status-dot blue';
    statusText.textContent = 'Thinking';

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    removeTypingIndicator();

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Request failed');
    }

    const data = await response.json();
    const reply = data.response;
    const sources = data.sources || [];
    const ragUsed = data.rag_used || false;

    // Append bot reply with source citations
    appendMessage('assistant', reply, sources, ragUsed);
    chatHistory.push({ role: 'model', text: reply });

    if (autoSpeakToggle.checked) {
      const playButtons = document.querySelectorAll('.play-btn');
      const lastPlayBtn = playButtons[playButtons.length - 1];
      speakText(reply, lastPlayBtn);
    } else {
      statusDot.className = 'status-dot green';
      statusText.textContent = 'Ready';
    }

  } catch (error) {
    removeTypingIndicator();
    statusDot.className = 'status-dot red';
    statusText.textContent = 'Error';
    showSystemMessage(`Error: ${error.message}`);
  }
}

// ---------------------------------------------------------------------------
// Append Message
// ---------------------------------------------------------------------------
function appendMessage(role, text, sources = [], ragUsed = false) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;

  const avatarDiv = document.createElement('div');
  avatarDiv.className = 'message-avatar';
  avatarDiv.innerHTML = role === 'user' ? '<i data-lucide="user"></i>' : '<i data-lucide="bot"></i>';

  const wrapperDiv = document.createElement('div');
  wrapperDiv.className = 'message-content-wrapper';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';

  if (role === 'assistant') {
    contentDiv.innerHTML = marked.parse(text);
  } else {
    const p = document.createElement('p');
    p.textContent = text;
    contentDiv.appendChild(p);
  }

  const metaDiv = document.createElement('div');
  metaDiv.className = 'message-meta';
  const now = new Date();
  let metaHTML = `<span>${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>`;
  if (ragUsed) metaHTML += `<span class="rag-pill"><i data-lucide="database"></i> RAG</span>`;
  metaDiv.innerHTML = metaHTML;

  const actionsDiv = document.createElement('div');
  actionsDiv.className = 'message-actions';

  // Copy button
  const copyBtn = document.createElement('button');
  copyBtn.className = 'msg-action-btn copy-btn';
  copyBtn.innerHTML = '<i data-lucide="copy"></i> Copy';
  copyBtn.onclick = () => {
    navigator.clipboard.writeText(text);
    copyBtn.innerHTML = '<i data-lucide="check"></i> Copied!';
    lucide.createIcons();
    setTimeout(() => { copyBtn.innerHTML = '<i data-lucide="copy"></i> Copy'; lucide.createIcons(); }, 1200);
  };
  actionsDiv.appendChild(copyBtn);

  // Speak button (assistant only)
  if (role === 'assistant') {
    const playBtn = document.createElement('button');
    playBtn.className = 'msg-action-btn play-btn';
    playBtn.innerHTML = '<i data-lucide="volume-2"></i> Speak';
    playBtn.onclick = () => speakText(text, playBtn);
    actionsDiv.appendChild(playBtn);
  }

  wrapperDiv.appendChild(contentDiv);
  wrapperDiv.appendChild(metaDiv);
  wrapperDiv.appendChild(actionsDiv);

  // Source citations (RAG only)
  if (ragUsed && sources.length > 0) {
    const citationsDiv = document.createElement('div');
    citationsDiv.className = 'source-citations';
    citationsDiv.innerHTML = `
      <div class="citations-label"><i data-lucide="book-open"></i> Sources</div>
      <div class="citations-list">
        ${sources.map(s => `
          <div class="citation-chip">
            <i data-lucide="file-text"></i>
            <span>${s.filename}</span>
            <span class="citation-score">${Math.round(s.score * 100)}%</span>
          </div>
        `).join('')}
      </div>
    `;
    wrapperDiv.appendChild(citationsDiv);
  }

  messageDiv.appendChild(avatarDiv);
  messageDiv.appendChild(wrapperDiv);
  chatMessages.appendChild(messageDiv);
  scrollToBottom();
  lucide.createIcons();
}

// ---------------------------------------------------------------------------
// System Notice / Error Messages
// ---------------------------------------------------------------------------
function showSystemMessage(text) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message assistant';

  const avatarDiv = document.createElement('div');
  avatarDiv.className = 'message-avatar';
  avatarDiv.innerHTML = '<i data-lucide="alert-circle"></i>';
  avatarDiv.style.borderColor = 'var(--accent-rose)';
  avatarDiv.style.color = 'var(--accent-rose)';

  const wrapperDiv = document.createElement('div');
  wrapperDiv.className = 'message-content-wrapper';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.style.border = '1px solid rgba(244, 63, 94, 0.15)';
  contentDiv.style.background = 'rgba(244, 63, 94, 0.03)';
  contentDiv.innerHTML = `<p style="color: var(--accent-rose); font-weight: 500;">${text}</p>`;

  const metaDiv = document.createElement('div');
  metaDiv.className = 'message-meta';
  const now = new Date();
  metaDiv.innerHTML = `<span>${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>`;

  wrapperDiv.appendChild(contentDiv);
  wrapperDiv.appendChild(metaDiv);
  messageDiv.appendChild(avatarDiv);
  messageDiv.appendChild(wrapperDiv);
  chatMessages.appendChild(messageDiv);
  scrollToBottom();
  lucide.createIcons();
}

// ---------------------------------------------------------------------------
// Typing Indicator
// ---------------------------------------------------------------------------
function showTypingIndicator() {
  const indicatorDiv = document.createElement('div');
  indicatorDiv.id = 'typing-indicator';
  indicatorDiv.className = 'message assistant';
  indicatorDiv.innerHTML = `
    <div class="message-avatar"><i data-lucide="bot"></i></div>
    <div class="message-content-wrapper">
      <div class="message-content">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  chatMessages.appendChild(indicatorDiv);
  scrollToBottom();
  lucide.createIcons();
}

function removeTypingIndicator() {
  const indicator = document.getElementById('typing-indicator');
  if (indicator) indicator.remove();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
