/* ═══════════════════════════════════════════════════════════════
   PredictaCore — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════ */

// ─── Chart instances ──────────────────────────────────────────────────────────
let gaugeChart = null;
let featureChart = null;
let probChart = null;

// ─── Status polling ───────────────────────────────────────────────────────────
function pollStatus() {
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      const dot  = document.querySelector('.status-dot');
      const text = document.getElementById('statusText');
      if (data.status === 'ready') {
        dot.className  = 'status-dot ready';
        text.textContent = 'System Ready';
        document.getElementById('predictBtn').disabled  = false;
        document.getElementById('simulateBtn').disabled = false;
      } else if (data.status === 'training') {
        dot.className  = 'status-dot';
        text.textContent = 'Training Models…';
        document.getElementById('predictBtn').disabled  = true;
        document.getElementById('simulateBtn').disabled = true;
        setTimeout(pollStatus, 2000);
      } else if (data.status === 'error') {
        dot.className  = 'status-dot error';
        text.textContent = 'Model Error';
      } else {
        setTimeout(pollStatus, 1500);
      }
    })
    .catch(() => setTimeout(pollStatus, 3000));
}

// ─── Type button toggle ───────────────────────────────────────────────────────
document.querySelectorAll('.type-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('typeInput').value = btn.dataset.val;
  });
});

// ─── Wear bar live update ─────────────────────────────────────────────────────
document.getElementById('toolWear').addEventListener('input', updateWearBar);
function updateWearBar() {
  const val = parseInt(document.getElementById('toolWear').value) || 0;
  const pct = Math.min((val / 253) * 100, 100);
  const bar = document.getElementById('wearBar');
  bar.style.width = pct + '%';
  if (pct > 80) bar.style.background = 'var(--red)';
  else if (pct > 55) bar.style.background = 'var(--amber)';
  else bar.style.background = 'var(--green)';
}
updateWearBar();

// ─── Validation ───────────────────────────────────────────────────────────────
function validateInputs() {
  const errs = [];
  const air  = parseFloat(document.getElementById('airTemp').value);
  const proc = parseFloat(document.getElementById('procTemp').value);
  const rpm  = parseInt(document.getElementById('rpm').value);
  const torq = parseFloat(document.getElementById('torque').value);
  const wear = parseInt(document.getElementById('toolWear').value);

  if (isNaN(air)  || air  < 295.3 || air  > 304.5) errs.push('Air Temp must be 295.3–304.5 K');
  if (isNaN(proc) || proc < 305.7 || proc > 313.8) errs.push('Process Temp must be 305.7–313.8 K');
  if (isNaN(rpm)  || rpm  < 1168  || rpm  > 2886)  errs.push('RPM must be 1168–2886');
  if (isNaN(torq) || torq < 3.8   || torq > 76.6)  errs.push('Torque must be 3.8–76.6 Nm');
  if (isNaN(wear) || wear < 0     || wear > 253)    errs.push('Tool Wear must be 0–253 min');

  const errDiv = document.getElementById('formError');
  if (errs.length) {
    errDiv.textContent = errs.join(' · ');
    errDiv.classList.remove('hidden');
    return null;
  }
  errDiv.classList.add('hidden');
  return { type: document.getElementById('typeInput').value, air_temp: air,
           proc_temp: proc, rpm, torque: torq, tool_wear: wear };
}

// ─── Charts init ──────────────────────────────────────────────────────────────
function initCharts() {
  Chart.defaults.color = '#6a7d92';
  Chart.defaults.font.family = "'Space Mono', monospace";
  Chart.defaults.font.size   = 10;

  // Gauge (doughnut)
  const gaugeCtx = document.getElementById('gaugeChart').getContext('2d');
  gaugeChart = new Chart(gaugeCtx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: ['#00d4aa', '#1e2530'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      cutout: '75%',
      animation: { duration: 800, easing: 'easeOutCubic' },
    }
  });

  // Feature importance bar
  const featCtx = document.getElementById('featureChart').getContext('2d');
  featureChart = new Chart(featCtx, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{
        data: [],
        backgroundColor: 'rgba(0,153,204,0.5)',
        borderColor: '#0099cc',
        borderWidth: 1,
        borderRadius: 3,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => ` Impact: ${ctx.parsed.x.toFixed(3)}` }
      }},
      scales: {
        x: { grid: { color: '#1e2530' }, ticks: { font: { size: 9 } } },
        y: { grid: { display: false }, ticks: { font: { size: 9 } } }
      },
      animation: { duration: 600 },
    }
  });

  // Probability horizontal bar
  const probCtx = document.getElementById('probChart').getContext('2d');
  probChart = new Chart(probCtx, {
    type: 'bar',
    data: {
      labels: ['Failure Prob'],
      datasets: [
        { data: [0],   backgroundColor: '#ef4444', borderRadius: 3, borderSkipped: false },
        { data: [100], backgroundColor: '#1e2530', borderRadius: 3, borderSkipped: false },
      ]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { stacked: true, max: 100, grid: { color: '#1e2530' },
             ticks: { callback: v => v + '%', font: { size: 9 } } },
        y: { stacked: true, grid: { display: false } }
      },
      animation: { duration: 500 },
    }
  });
}

// ─── Update charts ────────────────────────────────────────────────────────────
function updateCharts(result) {
  const score = result.final_score;
  const risk  = result.risk_level;

  // Gauge color based on risk
  const gaugeColor = risk === 'HIGH' ? '#ef4444' : risk === 'MEDIUM' ? '#f59e0b' : '#22c55e';
  gaugeChart.data.datasets[0].data = [score, 100 - score];
  gaugeChart.data.datasets[0].backgroundColor[0] = gaugeColor;
  gaugeChart.update();
  document.getElementById('gaugeCenterLabel').textContent = score + '%';
  document.getElementById('gaugeCenterLabel').style.color = gaugeColor;

  // Feature chart
  if (result.feature_chart && result.feature_chart.length) {
    const sorted = [...result.feature_chart].sort((a,b) => b.impact - a.impact);
    featureChart.data.labels   = sorted.map(f => truncFeature(f.feature));
    featureChart.data.datasets[0].data = sorted.map(f => f.impact);
    const maxImpact = Math.max(...sorted.map(f => f.impact));
    featureChart.data.datasets[0].backgroundColor = sorted.map(f => {
      const intensity = maxImpact > 0 ? f.impact / maxImpact : 0;
      return `rgba(0,153,204,${0.3 + 0.7 * intensity})`;
    });
    featureChart.update();
  }

  // Probability bar
  const p = result.probability;
  probChart.data.datasets[0].data = [p];
  probChart.data.datasets[1].data = [100 - p];
  const probColor = p > 70 ? '#ef4444' : p > 40 ? '#f59e0b' : '#22c55e';
  probChart.data.datasets[0].backgroundColor = probColor;
  probChart.update();
}

function truncFeature(name) {
  const map = {
    'Air temperature [K]': 'Air Temp',
    'Process temperature [K]': 'Proc Temp',
    'Rotational speed [rpm]': 'RPM',
    'Torque [Nm]': 'Torque',
    'Tool wear [min]': 'Tool Wear',
    'Torque_per_speed': 'Torq/Speed',
    'Wear_rate': 'Wear Rate',
    'Temp_diff': 'Temp Diff',
    'Power': 'Power',
    'Type': 'Type',
  };
  return map[name] || name;
}

// ─── Display results ──────────────────────────────────────────────────────────
function displayResults(result) {
  document.getElementById('emptyState').classList.add('hidden');
  document.getElementById('resultsContent').classList.remove('hidden');

  // Prediction badge
  const badge = document.getElementById('predBadge');
  badge.textContent = result.prediction;
  badge.className   = 'prediction-badge ' + result.prediction.toLowerCase();

  // Probability
  document.getElementById('probVal').textContent = result.probability + '%';

  // Risk
  const cardRisk = document.getElementById('cardRisk');
  cardRisk.className = 'metric-card risk-' + result.risk_level.toLowerCase();
  document.getElementById('riskVal').textContent = result.risk_level;
  const riskPct = result.risk_level === 'HIGH' ? 95 : result.risk_level === 'MEDIUM' ? 60 : 20;
  document.getElementById('riskBar').style.width = riskPct + '%';

  // Confidence
  const cardConf = document.getElementById('cardConf');
  const confClass = result.confidence.includes('HIGH') ? 'conf-high'
                  : result.confidence.includes('MODERATE') ? 'conf-moderate' : 'conf-uncertain';
  cardConf.className = 'metric-card ' + confClass;
  document.getElementById('confVal').textContent = result.confidence;

  // Anomaly
  const cardAnom = document.getElementById('cardAnomaly');
  cardAnom.className = 'metric-card anomaly-' + result.anomaly.toLowerCase();
  document.getElementById('anomalyVal').textContent = result.anomaly;

  // Score
  document.getElementById('scoreVal').textContent = result.final_score + '%';

  // Root causes
  const causesList = document.getElementById('causesList');
  causesList.innerHTML = '';
  (result.root_causes || []).forEach(c => {
    const div = document.createElement('div');
    div.className = 'cause-item';
    div.innerHTML = `<span class="cause-name">${truncFeature(c.feature)}</span>
                     <span class="cause-val">${(c.impact * 100).toFixed(2)}%</span>`;
    causesList.appendChild(div);
  });

  // Action
  const actionEl = document.getElementById('actionText');
  actionEl.textContent = result.action;
  if (result.prediction === 'FAILURE') {
    actionEl.className = result.risk_level === 'HIGH' ? 'action-text critical' : 'action-text warn';
  } else {
    actionEl.className = 'action-text ok';
  }

  updateCharts(result);
}

// ─── Run prediction ───────────────────────────────────────────────────────────
async function runPredict() {
  const payload = validateInputs();
  if (!payload) return;

  const overlay = document.getElementById('loadingOverlay');
  overlay.classList.remove('hidden');
  document.getElementById('loadingText').textContent = 'Running prediction engine…';
  document.getElementById('predictBtn').disabled = true;

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await resp.json();
    if (result.error) throw new Error(result.error);
    overlay.classList.add('hidden');
    displayResults(result);
  } catch (err) {
    overlay.classList.add('hidden');
    const errDiv = document.getElementById('formError');
    errDiv.textContent = 'Prediction failed: ' + err.message;
    errDiv.classList.remove('hidden');
  } finally {
    document.getElementById('predictBtn').disabled = false;
  }
}

// ─── Simulation ───────────────────────────────────────────────────────────────
let simInterval = null;

async function startSimulation() {
  const drawer   = document.getElementById('simDrawer');
  const backdrop = document.getElementById('simBackdrop');
  const grid     = document.getElementById('simGrid');

  grid.innerHTML = '';
  document.getElementById('simCounter').textContent = '0 / 8';
  drawer.classList.add('open');
  backdrop.classList.add('open');

  document.getElementById('simulateBtn').disabled = true;

  try {
    const resp = await fetch('/api/simulate');
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    const samples = data.samples;
    let idx = 0;

    function showNext() {
      if (idx >= samples.length) {
        document.getElementById('simulateBtn').disabled = false;
        return;
      }
      const s = samples[idx++];
      document.getElementById('simCounter').textContent = idx + ' / ' + samples.length;

      const card = document.createElement('div');
      const cls  = s.prediction === 'FAILURE' ? 'failure' : 'normal';
      card.className = `sim-card ${cls}`;
      card.innerHTML = `
        <div class="sim-card-header">
          <span class="sim-machine-id">${s.machine_id}</span>
          <span class="sim-status-badge ${cls}">${s.prediction}</span>
        </div>
        <div class="sim-metrics">
          <div class="sim-metric-row">
            <span class="sim-metric-key">Risk</span>
            <span class="sim-metric-val">${s.risk_level}</span>
          </div>
          <div class="sim-metric-row">
            <span class="sim-metric-key">Probability</span>
            <span class="sim-metric-val">${s.probability}%</span>
          </div>
          <div class="sim-metric-row">
            <span class="sim-metric-key">Anomaly</span>
            <span class="sim-metric-val">${s.anomaly}</span>
          </div>
          <div class="sim-metric-row">
            <span class="sim-metric-key">Score</span>
            <span class="sim-metric-val">${s.final_score}%</span>
          </div>
          <div class="sim-metric-row">
            <span class="sim-metric-key">Action</span>
            <span class="sim-metric-val" style="font-size:9px;max-width:110px;text-align:right;">${s.action.slice(0, 35)}…</span>
          </div>
        </div>`;
      grid.appendChild(card);

      // Animate in
      requestAnimationFrame(() => {
        requestAnimationFrame(() => card.classList.add('visible'));
      });

      // Also update main dashboard with last prediction
      displayResults(s);

      simInterval = setTimeout(showNext, 900);
    }

    showNext();

  } catch (err) {
    document.getElementById('simulateBtn').disabled = false;
    alert('Simulation error: ' + err.message);
    closeSimulation();
  }
}

function closeSimulation() {
  if (simInterval) clearTimeout(simInterval);
  document.getElementById('simDrawer').classList.remove('open');
  document.getElementById('simBackdrop').classList.remove('open');
  document.getElementById('simulateBtn').disabled = false;
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  pollStatus();
});
