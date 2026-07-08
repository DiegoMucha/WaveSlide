import './style.css';

import * as pdfjsLib from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const API_BASE_URL = 'http://127.0.0.1:8000';
const WS_EVENTS_URL = 'ws://127.0.0.1:8000/ws/events';
const VIDEO_STREAM_URL = `${API_BASE_URL}/video/stream`;
const CONFIDENCE_THRESHOLD = 0.90;
const ACTION_COOLDOWN_MS = 1200;
const ACTION_DECISION_WINDOW_MS = 1000;
const ACTION_DECISION_MIN_SPAN_MS = 800;
const ACTION_DECISION_MIN_SAMPLES = 6;
const ACTION_DECISION_MIN_RATIO = 0.60;
const NO_PREDICTION_TIMEOUT_MS = 900;
const DEFAULT_LABELS = ['call', 'fist', 'like', 'two_up', 'unknown'];

const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');

const statusEl = document.getElementById('status');
const gestureEl = document.getElementById('gesture');
const confidenceEl = document.getElementById('confidence');
const probabilitiesEl = document.getElementById('probabilities');
const lastActionEl = document.getElementById('last-action');
const gestureControlEl = document.getElementById('gesture-control');

const pdfInput = document.getElementById('pdf-input');
const pdfCanvas = document.getElementById('pdf-canvas');
const pdfCtx = pdfCanvas.getContext('2d');
const pageInfo = document.getElementById('page-info');

const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const zoomInBtn = document.getElementById('zoom-in-btn');
const zoomOutBtn = document.getElementById('zoom-out-btn');

const appEl = document.querySelector('.app');
const presentationBtn = document.getElementById('presentation-btn');
const toggleCameraBtn = document.getElementById('toggle-camera-btn');

let pdfDoc = null;
let currentPage = 1;
let zoomFactor = 1;
let rendering = false;
let pendingPage = null;

let lastActionAt = 0;
let predictionHistory = [];
let lastBackendPredictionAt = 0;
let eventsSocket = null;
let backendLabels = DEFAULT_LABELS;
let backendRetryTimer = null;

let gestureControlEnabled = false;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(message) {
  statusEl.textContent = message;
}

function updateGestureControlUI() {
  if (gestureControlEl) {
    gestureControlEl.textContent = gestureControlEnabled ? 'Activado' : 'Desactivado';
  }

  appEl?.classList.toggle('gesture-enabled', gestureControlEnabled);
}

async function loadPdfFromFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  currentPage = 1;
  zoomFactor = 1;
  await renderPage(currentPage);
}

async function renderPage(pageNumber) {
  if (!pdfDoc) return;

  if (rendering) {
    pendingPage = pageNumber;
    return;
  }

  rendering = true;

  try {
    const page = await pdfDoc.getPage(pageNumber);
    const pdfWrapper = document.querySelector('.pdf-wrapper');

    const baseViewport = page.getViewport({ scale: 1 });

    const availableWidth = Math.max(320, pdfWrapper.clientWidth - 16);
    const availableHeight = Math.max(240, pdfWrapper.clientHeight - 16);

    const fitScale = Math.min(
      availableWidth / baseViewport.width,
      availableHeight / baseViewport.height
    );

    const finalScale = fitScale * zoomFactor;
    const viewport = page.getViewport({ scale: finalScale });

    const outputScale = window.devicePixelRatio || 1;

    pdfCanvas.width = Math.floor(viewport.width * outputScale);
    pdfCanvas.height = Math.floor(viewport.height * outputScale);

    pdfCanvas.style.width = `${Math.floor(viewport.width)}px`;
    pdfCanvas.style.height = `${Math.floor(viewport.height)}px`;

    const transform = outputScale !== 1
      ? [outputScale, 0, 0, outputScale, 0, 0]
      : null;

    await page.render({
      canvasContext: pdfCtx,
      viewport,
      transform
    }).promise;

    pageInfo.textContent = `Página ${currentPage} de ${pdfDoc.numPages}`;
  } finally {
    rendering = false;
  }

  if (pendingPage !== null) {
    const nextPending = pendingPage;
    pendingPage = null;
    await renderPage(nextPending);
  }
}

async function nextPage() {
  if (!pdfDoc) {
    lastActionEl.textContent = 'Carga un PDF primero';
    return;
  }

  if (currentPage >= pdfDoc.numPages) {
    lastActionEl.textContent = 'Ya estás en la última página';
    return;
  }

  currentPage += 1;
  await renderPage(currentPage);
}

async function prevPage() {
  if (!pdfDoc) {
    lastActionEl.textContent = 'Carga un PDF primero';
    return;
  }

  if (currentPage <= 1) {
    lastActionEl.textContent = 'Ya estás en la primera página';
    return;
  }

  currentPage -= 1;
  await renderPage(currentPage);
}

async function zoomIn() {
  if (!pdfDoc) return;
  zoomFactor = Math.min(zoomFactor + 0.1, 2);
  await renderPage(currentPage);
}

async function zoomOut() {
  if (!pdfDoc) return;
  zoomFactor = Math.max(zoomFactor - 0.1, 0.5);
  await renderPage(currentPage);
}

async function togglePresentationMode() {
  if (!appEl) return;

  appEl.classList.toggle('presentation-mode');
  appEl.classList.remove('show-camera');

  await delay(80);

  if (pdfDoc) {
    await renderPage(currentPage);
  }
}

async function toggleCameraPanel() {
  if (!appEl) return;

  appEl.classList.toggle('show-camera');

  if (toggleCameraBtn) {
    toggleCameraBtn.textContent = appEl.classList.contains('show-camera')
      ? 'Ocultar cámara'
      : 'Mostrar cámara';
  }

  await delay(80);

  if (pdfDoc) {
    await renderPage(currentPage);
  }
}

async function executeGesture(gesture) {
  const now = performance.now();

  if (now - lastActionAt < ACTION_COOLDOWN_MS) {
    return;
  }

  // LIKE funciona como interruptor general.
  if (gesture === 'like') {
    gestureControlEnabled = !gestureControlEnabled;
    updateGestureControlUI();

    lastActionEl.textContent = gestureControlEnabled
      ? 'Control por gestos activado'
      : 'Control por gestos desactivado';

    lastActionAt = now;
    predictionHistory = [];

    return;
  }

  // Si el control está desactivado, ignora los demás gestos.
  if (!gestureControlEnabled) {
    lastActionEl.textContent = 'Gestos bloqueados: haz like para activar';
    predictionHistory = [];

    return;
  }

  if (gesture === 'two_up') {
    await nextPage();
    lastActionEl.textContent = 'Siguiente página';
  } else if (gesture === 'fist') {
    await prevPage();
    lastActionEl.textContent = 'Página anterior';
  } else if (gesture === 'call') {
    await togglePresentationMode();
    lastActionEl.textContent = 'Modo presentación';
  } else {
    return;
  }

  lastActionAt = now;
  predictionHistory = [];
}

function resizeOverlay() {
  const box = overlay.parentElement?.getBoundingClientRect();
  const width = Math.max(1, Math.floor(box?.width ?? 640));
  const height = Math.max(1, Math.floor(box?.height ?? 480));

  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
}

function clearCameraOverlay() {
  resizeOverlay();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function scheduleBackendRetry() {
  if (backendRetryTimer !== null) {
    return;
  }

  backendRetryTimer = window.setTimeout(() => {
    backendRetryTimer = null;
    init();
  }, 1500);
}

function recordGestureVote(gesture, now) {
  predictionHistory.push({ gesture, at: now });
  predictionHistory = predictionHistory.filter(
    (item) => now - item.at <= ACTION_DECISION_WINDOW_MS
  );

  if (predictionHistory.length < ACTION_DECISION_MIN_SAMPLES) {
    return null;
  }

  const windowSpan = predictionHistory.at(-1).at - predictionHistory[0].at;
  if (windowSpan < ACTION_DECISION_MIN_SPAN_MS) {
    return null;
  }

  const counts = new Map();
  for (const item of predictionHistory) {
    counts.set(item.gesture, (counts.get(item.gesture) ?? 0) + 1);
  }

  let topGesture = null;
  let topCount = 0;
  for (const [label, count] of counts.entries()) {
    if (count > topCount) {
      topGesture = label;
      topCount = count;
    }
  }

  if (topCount / predictionHistory.length < ACTION_DECISION_MIN_RATIO) {
    return null;
  }

  return topGesture;
}

async function configureBackend() {
  const response = await fetch(`${API_BASE_URL}/config`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      control_presentation: false,
      min_prediction_confidence: CONFIDENCE_THRESHOLD
    })
  });

  if (response.status === 409) {
    await stopBackendEngine();
    return configureBackend();
  }

  if (!response.ok) {
    throw new Error(`No se pudo configurar el backend (${response.status})`);
  }

  const config = await response.json();
  backendLabels = Array.isArray(config.labels) && config.labels.length
    ? config.labels
    : DEFAULT_LABELS;
}

async function startBackendEngine() {
  const response = await fetch(`${API_BASE_URL}/engine/start`, {
    method: 'POST'
  });

  if (!response.ok) {
    throw new Error(`No se pudo iniciar el backend (${response.status})`);
  }
}

async function stopBackendEngine() {
  const response = await fetch(`${API_BASE_URL}/engine/stop`, {
    method: 'POST'
  });

  if (!response.ok) {
    throw new Error(`No se pudo detener el backend (${response.status})`);
  }

  await delay(300);
}

function connectBackendEvents() {
  if (eventsSocket) {
    eventsSocket.close();
  }

  eventsSocket = new WebSocket(WS_EVENTS_URL);

  eventsSocket.addEventListener('open', () => {
    setStatus('Conectado al backend');
    clearCameraOverlay();
  });

  eventsSocket.addEventListener('message', async (message) => {
    const event = JSON.parse(message.data);
    await handleBackendEvent(event);
  });

  eventsSocket.addEventListener('close', () => {
    setStatus('Backend desconectado');
    clearCameraOverlay();
  });

  eventsSocket.addEventListener('error', () => {
    setStatus('Error de conexión con backend');
    clearCameraOverlay();
  });
}

async function handleBackendEvent(event) {
  if (event.type === 'error') {
    setStatus(`Error: ${event.message}`);
    clearCameraOverlay();
    return;
  }

  if (event.type === 'status') {
    setStatus(event.message === 'engine_started' ? 'Backend activo' : 'Backend detenido');
    return;
  }

  if (event.type !== 'prediction') {
    return;
  }

  lastBackendPredictionAt = performance.now();

  const gesture = event.gesture || '---';
  const confidence = Number(event.confidence ?? 0);
  const probabilitiesText = formatProbabilities(event.probabilities);

  gestureEl.textContent = gesture;
  confidenceEl.textContent = `${(confidence * 100).toFixed(2)}%`;
  probabilitiesEl.textContent = probabilitiesText || '---';
  clearCameraOverlay();

  if (gesture === 'unknown') {
    predictionHistory = [];
    lastActionEl.textContent = 'Gesto desconocido';
    return;
  }

  if (confidence < CONFIDENCE_THRESHOLD) {
    return;
  }

  const selectedGesture = recordGestureVote(gesture, performance.now());

  if (selectedGesture !== null) {
    await executeGesture(selectedGesture);
  }
}

function formatProbabilities(probabilities) {
  if (!probabilities) {
    return '';
  }

  return backendLabels
    .map((label) => `${label}: ${Number(probabilities[label] ?? 0).toFixed(4)}`)
    .join(' | ');
}

function watchPredictionTimeout() {
  window.setInterval(() => {
    if (!lastBackendPredictionAt) {
      return;
    }

    if (performance.now() - lastBackendPredictionAt < NO_PREDICTION_TIMEOUT_MS) {
      return;
    }

    gestureEl.textContent = 'sin mano';
    confidenceEl.textContent = '---';
    probabilitiesEl.textContent = '---';
    predictionHistory = [];
    clearCameraOverlay();
  }, 250);
}

async function init() {
  try {
    if (backendRetryTimer !== null) {
      window.clearTimeout(backendRetryTimer);
      backendRetryTimer = null;
    }

    updateGestureControlUI();

    video.removeAttribute('src');
    clearCameraOverlay();

    setStatus('Conectando backend...');
    await configureBackend();
    connectBackendEvents();

    setStatus('Iniciando backend...');
    await startBackendEngine();
    video.src = `${VIDEO_STREAM_URL}?t=${Date.now()}`;
    watchPredictionTimeout();
  } catch (error) {
    console.error(error);
    setStatus(`Error: ${error.message}. Reintentando...`);
    clearCameraOverlay();
    scheduleBackendRetry();
  }
}

pdfInput.addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    await loadPdfFromFile(file);
  } catch (error) {
    console.error(error);
    pageInfo.textContent = 'No se pudo cargar el PDF.';
  }
});

prevBtn.addEventListener('click', prevPage);
nextBtn.addEventListener('click', nextPage);
zoomInBtn.addEventListener('click', zoomIn);
zoomOutBtn.addEventListener('click', zoomOut);

presentationBtn?.addEventListener('click', togglePresentationMode);
toggleCameraBtn?.addEventListener('click', toggleCameraPanel);

window.addEventListener('resize', () => {
  if (pdfDoc) {
    renderPage(currentPage);
  }
});

document.addEventListener('keydown', async (event) => {
  const key = event.key.toLowerCase();

  if (key === 'p') {
    await togglePresentationMode();
  }

  if (key === 'c') {
    await toggleCameraPanel();
  }
});

init();
