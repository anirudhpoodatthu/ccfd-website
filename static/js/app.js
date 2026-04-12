/* ═══════════════════════════════════════════════════════════════
   CREDIT CARD FRAUD DETECTION – CLIENT-SIDE LOGIC
   ═══════════════════════════════════════════════════════════════ */

// ── Particles Background ──────────────────────────────────────
function createParticles() {
  const container = document.getElementById('particles-bg');
  if (!container) return;
  const count = 30;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left = Math.random() * 100 + '%';
    p.style.setProperty('--duration', (8 + Math.random() * 12) + 's');
    p.style.animationDelay = Math.random() * 10 + 's';
    p.style.width = p.style.height = (2 + Math.random() * 3) + 'px';
    container.appendChild(p);
  }
}

document.addEventListener('DOMContentLoaded', createParticles);


// ── Tab Switching ─────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));

  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  document.getElementById(tabId).classList.add('active');

  // Hide previous results
  hideResult();
  hideBatchResult();
}


// ── Manual Prediction ─────────────────────────────────────────
async function predictManual() {
  const btn = document.getElementById('predict-btn');
  const featureInputs = document.querySelectorAll('#manual-form input[data-feature]');
  const features = {};

  featureInputs.forEach(input => {
    features[input.dataset.feature] = parseFloat(input.value) || 0.0;
  });

  // Show loading
  btn.classList.add('loading');
  hideResult();

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });

    const data = await response.json();

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    showResult(data);
  } catch (err) {
    alert('Request failed: ' + err.message);
  } finally {
    btn.classList.remove('loading');
  }
}


// ── Random Sample ─────────────────────────────────────────────
function fillRandom(type) {
  const inputs = document.querySelectorAll('#manual-form input[data-feature]');

  inputs.forEach(input => {
    const name = input.dataset.feature;
    let val;

    if (type === 'normal') {
      val = randn() * 0.3;
    } else if (type === 'suspicious') {
      val = randn() * 2;
      if (name === 'V14') val = -5 + randn() * 0.5;
      if (name === 'V12') val = -4 + randn() * 0.5;
      if (name === 'V10') val = -3.5 + randn() * 0.5;
      if (name === 'V17') val = -3 + randn() * 0.5;
    } else {
      val = randn();
    }

    input.value = val.toFixed(6);
  });
}

function clearForm() {
  document.querySelectorAll('#manual-form input[data-feature]').forEach(input => {
    input.value = '0.000000';
  });
  hideResult();
}

// Box-Muller transform for normal distribution
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}


// ── Show Result ───────────────────────────────────────────────
function showResult(data) {
  const section = document.getElementById('result-section');
  const banner = document.getElementById('result-banner');
  const icon = document.getElementById('result-icon');
  const label = document.getElementById('result-label');
  const desc = document.getElementById('result-desc');

  // Banner
  if (data.prediction === 0) {
    banner.className = 'result-banner safe';
    icon.textContent = '✅';
    label.textContent = 'LEGITIMATE TRANSACTION';
    desc.textContent = 'This transaction appears to be legitimate based on model analysis.';
  } else {
    banner.className = 'result-banner fraud';
    icon.textContent = '🚨';
    label.textContent = 'FRAUDULENT TRANSACTION DETECTED';
    desc.textContent = '⚠️ This transaction has been flagged as potentially fraudulent!';
  }

  // Metrics
  document.getElementById('metric-legit').textContent = data.legit_probability_pct + '%';
  document.getElementById('metric-fraud').textContent = data.fraud_probability_pct + '%';

  const riskEl = document.getElementById('metric-risk');
  riskEl.textContent = '⚡ ' + data.risk_level;
  riskEl.className = 'metric-value ' + (
    data.risk_level === 'Low' ? 'success' :
    data.risk_level === 'Medium' ? 'warning' : 'danger'
  );

  // Probability bars
  setTimeout(() => {
    document.getElementById('bar-legit').style.width = data.legit_probability_pct + '%';
    document.getElementById('bar-fraud').style.width = data.fraud_probability_pct + '%';
  }, 100);

  document.getElementById('bar-legit-pct').textContent = data.legit_probability_pct + '%';
  document.getElementById('bar-fraud-pct').textContent = data.fraud_probability_pct + '%';

  // Gauge
  drawGauge(data.fraud_probability_pct);

  // Show section
  section.classList.add('visible');
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResult() {
  const section = document.getElementById('result-section');
  if (section) section.classList.remove('visible');
}


// ── SVG Gauge ─────────────────────────────────────────────────
function drawGauge(percent) {
  const gaugeEl = document.getElementById('gauge-arc');
  const gaugeText = document.getElementById('gauge-text');
  if (!gaugeEl || !gaugeText) return;

  // Arc from 180° to 0° (left to right semicircle)
  const radius = 85;
  const cx = 110, cy = 100;
  const startAngle = Math.PI;
  const endAngle = Math.PI - (percent / 100) * Math.PI;

  const x1 = cx + radius * Math.cos(startAngle);
  const y1 = cy + radius * Math.sin(startAngle);
  const x2 = cx + radius * Math.cos(endAngle);
  const y2 = cy + radius * Math.sin(endAngle);

  const largeArc = percent > 50 ? 1 : 0;

  gaugeEl.setAttribute('d', `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`);

  // Color based on percentage
  let color;
  if (percent < 30) color = '#2ecc71';
  else if (percent < 70) color = '#f39c12';
  else color = '#e74c3c';

  gaugeEl.setAttribute('stroke', color);
  gaugeText.textContent = percent + '%';
  gaugeText.setAttribute('fill', color);
}


// ── CSV Upload ────────────────────────────────────────────────
function handleFileSelect(input) {
  const fileInfo = document.getElementById('file-info');
  const fileName = document.getElementById('file-name');

  if (input.files.length > 0) {
    const file = input.files[0];
    fileName.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.classList.add('visible');
  } else {
    fileInfo.classList.remove('visible');
  }
}

async function predictBatch() {
  const fileInput = document.getElementById('csv-file');
  if (!fileInput.files.length) {
    alert('Please select a CSV file first.');
    return;
  }

  const btn = document.getElementById('batch-btn');
  btn.classList.add('loading');
  hideBatchResult();

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const response = await fetch('/predict-batch', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      alert('Error: ' + data.error);
      return;
    }

    showBatchResult(data);
  } catch (err) {
    alert('Request failed: ' + err.message);
  } finally {
    btn.classList.remove('loading');
  }
}

function showBatchResult(data) {
  const section = document.getElementById('batch-result-section');

  // Summary
  document.getElementById('batch-total').textContent = data.total.toLocaleString();
  document.getElementById('batch-legit').textContent = data.legit_count.toLocaleString();
  document.getElementById('batch-fraud').textContent = data.fraud_count.toLocaleString();

  // Table
  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = '';

  data.results.forEach(row => {
    const tr = document.createElement('tr');
    const statusClass = row.prediction === 0 ? 'legit' : 'fraud';
    const statusIcon = row.prediction === 0 ? '✅' : '🚨';

    tr.innerHTML = `
      <td>${row.index}</td>
      <td><span class="status-badge ${statusClass}">${statusIcon} ${row.label}</span></td>
      <td>${row.fraud_probability}%</td>
      <td><span class="metric-value ${
        row.risk_level === 'Low' ? 'success' :
        row.risk_level === 'Medium' ? 'warning' : 'danger'
      }" style="font-size: 0.85rem; font-weight: 600;">${row.risk_level}</span></td>
    `;
    tbody.appendChild(tr);
  });

  section.classList.add('visible');
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideBatchResult() {
  const section = document.getElementById('batch-result-section');
  if (section) section.classList.remove('visible');
}


// ── Upload Zone Drag & Drop ───────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const zone = document.querySelector('.upload-zone');
  if (!zone) return;

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('dragover');
  });

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const fileInput = document.getElementById('csv-file');
    fileInput.files = e.dataTransfer.files;
    handleFileSelect(fileInput);
  });
});
