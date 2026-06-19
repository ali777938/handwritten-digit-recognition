// Theme Toggle
(function(){
  var t = document.querySelector('[data-theme-toggle]');
  var r = document.documentElement;
  var d = matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light';
  r.setAttribute('data-theme', d);
  if(t) t.addEventListener('click', function() {
    d = d === 'dark' ? 'light' : 'dark';
    r.setAttribute('data-theme', d);
    initCanvas();
  });
})();

// Canvas Drawing
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var isDrawing = false;
var currentMode = 'single';
var strokes = [];
var currentStroke = [];

function getCanvasColor() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? '#111111' : '#ffffff';
}
function getCanvasBg() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? '#ffffff' : '#111111';
}

function initCanvas() {
  ctx.fillStyle = getCanvasBg();
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  // Redraw existing strokes
  strokes.forEach(function(stroke) {
    if (stroke.length === 0) return;
    ctx.beginPath();
    ctx.strokeStyle = getCanvasColor();
    ctx.lineWidth = parseInt(document.getElementById('brushSize').value);
    ctx.moveTo(stroke[0].x, stroke[0].y);
    stroke.forEach(function(p) { ctx.lineTo(p.x, p.y); });
    ctx.stroke();
  });
}
initCanvas();

function getPos(e) {
  var rect = canvas.getBoundingClientRect();
  var scaleX = canvas.width / rect.width;
  var scaleY = canvas.height / rect.height;
  if (e.touches) {
    return { x: (e.touches[0].clientX - rect.left) * scaleX, y: (e.touches[0].clientY - rect.top) * scaleY };
  }
  return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
}

canvas.addEventListener('mousedown', function(e) {
  isDrawing = true; currentStroke = [];
  var p = getPos(e);
  ctx.strokeStyle = getCanvasColor();
  ctx.lineWidth = parseInt(document.getElementById('brushSize').value);
  ctx.beginPath(); ctx.moveTo(p.x, p.y);
  currentStroke.push(p);
});
canvas.addEventListener('mousemove', function(e) {
  if (!isDrawing) return;
  var p = getPos(e);
  ctx.lineTo(p.x, p.y); ctx.stroke();
  currentStroke.push(p);
});
canvas.addEventListener('mouseup', function() {
  isDrawing = false;
  if(currentStroke.length) strokes.push(currentStroke.slice());
});
canvas.addEventListener('mouseleave', function() {
  if(isDrawing) { isDrawing = false; if(currentStroke.length) strokes.push(currentStroke.slice()); }
});
canvas.addEventListener('touchstart', function(e) {
  e.preventDefault(); isDrawing = true; currentStroke = [];
  var p = getPos(e);
  ctx.strokeStyle = getCanvasColor();
  ctx.lineWidth = parseInt(document.getElementById('brushSize').value);
  ctx.beginPath(); ctx.moveTo(p.x, p.y);
  currentStroke.push(p);
}, { passive: false });
canvas.addEventListener('touchmove', function(e) {
  e.preventDefault();
  if (!isDrawing) return;
  var p = getPos(e);
  ctx.lineTo(p.x, p.y); ctx.stroke();
  currentStroke.push(p);
}, { passive: false });
canvas.addEventListener('touchend', function() {
  isDrawing = false;
  if(currentStroke.length) strokes.push(currentStroke.slice());
});

document.getElementById('brushSize').addEventListener('input', function() {
  document.getElementById('brushVal').textContent = this.value;
});

function clearCanvas() {
  strokes = []; currentStroke = [];
  initCanvas();
  showEmpty();
}

function undoLast() {
  if (strokes.length === 0) return;
  strokes.pop();
  initCanvas();
}

function setMode(m) {
  currentMode = m;
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
  document.querySelector('[data-mode="' + m + '"]').classList.add('active');
  document.getElementById('mode-hint').textContent = m === 'single'
    ? 'Draw a single digit (0-9)'
    : 'Draw multiple digits left to right (e.g. 1 2 3)';
  clearCanvas();
}

function loadImage(event) {
  var file = event.target.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) {
    var img = new Image();
    img.onload = function() { clearCanvas(); ctx.drawImage(img, 0, 0, canvas.width, canvas.height); };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function showEmpty() {
  document.getElementById('resultArea').innerHTML =
    '<div class="result-empty">' +
    '<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>' +
    '<p>Draw a digit and click "Recognize Digit"</p></div>';
}

function showLoading() {
  document.getElementById('resultArea').innerHTML =
    '<div style="display:flex;flex-direction:column;align-items:center;gap:12px;color:var(--color-text-muted);font-size:var(--text-sm)">' +
    '<div class="loading-spinner"></div><p>Analyzing...</p></div>';
}

function showError(msg) {
  document.getElementById('resultArea').innerHTML =
    '<div style="color:var(--color-error);text-align:center;font-size:var(--text-sm);padding:var(--space-4)">' +
    '<strong>Error:</strong> ' + msg + '</div>';
}

function showResult(data) {
  var area = document.getElementById('resultArea');
  if (data.mode === 'multi' && data.results) {
    var chips = data.results.map(function(r) {
      return '<div class="multi-digit-chip"><div class="digit">' + r.digit + '</div>' +
             '<div class="conf">' + r.confidence.toFixed(1) + '%</div></div>';
    }).join('');
    area.innerHTML = '<div class="result-content">' +
      '<div class="result-digit-display">' +
      '<div class="result-number">' + data.number + '</div>' +
      '<div class="result-conf">Recognized number</div></div>' +
      '<div class="multi-results">' + chips + '</div></div>';
  } else {
    var probHTML = data.all_probs.map(function(p, i) {
      return '<div class="prob-cell' + (i === data.digit ? ' top' : '') + '">' +
             '<div class="prob-cell-digit">' + i + '</div>' +
             '<div class="prob-cell-val">' + p.toFixed(1) + '%</div></div>';
    }).join('');
    area.innerHTML = '<div class="result-content">' +
      '<div class="result-digit-display">' +
      '<div class="result-digit">' + data.digit + '</div>' +
      '<div class="result-conf">Confidence: ' + data.confidence.toFixed(1) + '%</div>' +
      '<div class="conf-bar-wrap"><div class="conf-bar-bg">' +
      '<div class="conf-bar-fill" style="width:' + data.confidence + '%"></div>' +
      '</div></div></div>' +
      '<div class="prob-grid">' + probHTML + '</div></div>';
  }
}

function predictDigit() {
  var imageData = canvas.toDataURL('image/png');
  showLoading();
  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData, mode: currentMode })
  })
  .then(function(response) {
    return response.text().then(function(text) {
      var data = {};
      try {
        data = text ? JSON.parse(text) : {};
      } catch (e) {
        throw new Error('Invalid server response');
      }
      if (!response.ok) {
        showError(data.error || 'Prediction failed');
        return;
      }
      showResult(data);
    });
  })
  .catch(function(err) { showError('Network error: ' + err.message); });
}
