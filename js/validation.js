/*
 * Page script for validation.html. Renders the embedded validation report
 * (<script id="validation-data" type="application/json">, filled in at build
 * time from backend/validation-results.json by scripts/copy-static.js).
 *
 * Built with DOM methods + textContent only (no innerHTML), so nothing in the
 * report data can inject markup, and no inline script is needed (CSP).
 */
(function () {
  function el(tag, className, text) {
    var e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  var statsEl = document.getElementById('stats');
  var dataEl = document.getElementById('validation-data');
  var raw = dataEl ? dataEl.textContent.trim() : '';
  var report;
  try {
    // Guard against the un-substituted placeholder still being present
    // (happens on `open validation.html` before any build ran).
    if (raw.indexOf('%%') === 0) throw new Error('placeholder not substituted');
    report = JSON.parse(raw);
  } catch (e) {
    if (!statsEl) return;
    clear(statsEl);
    var card = el('div', 'stat-card');
    card.style.gridColumn = '1/-1';
    var label = el('div', 'stat-label', 'Validation report not available');
    label.style.color = '#F87171';
    var note = el('div', 'stat-note');
    note.appendChild(document.createTextNode('Run '));
    note.appendChild(el('code', null, 'pytest -m validation'));
    note.appendChild(document.createTextNode(' in the backend to generate '));
    note.appendChild(el('code', null, 'backend/validation-results.json'));
    note.appendChild(document.createTextNode(', then rebuild the site.'));
    card.appendChild(label);
    card.appendChild(note);
    statsEl.appendChild(card);
    return;
  }

  // ── Stat cards ──
  var d = new Date(report.generated_at);
  var dateStr = isNaN(d.getTime()) ? report.generated_at : d.toISOString().slice(0, 10);
  var stats = [
    { value: report.n_methods_validated, label: 'Methods validated', accent: true },
    { value: report.n_total_checks, label: 'Independent checks' },
    { value: dateStr, label: 'Last verified' },
  ];
  if (statsEl) {
    clear(statsEl);
    stats.forEach(function (s) {
      var c = el('div', 'stat-card' + (s.accent ? ' accent' : ''));
      c.appendChild(el('div', 'stat-value', s.value));
      c.appendChild(el('div', 'stat-label', s.label));
      statsEl.appendChild(c);
    });
  }

  // ── Coverage note ──
  var covEl = document.getElementById('coverage-summary');
  if (covEl) covEl.textContent = report.n_methods_validated + ' of those';

  // ── Table rows ──
  var kindClass = function (k) {
    return String(k).toLowerCase().replace(/[^a-z]+/g, '-');
  };
  var kindLabel = function (k) {
    return k === 'nist' ? 'NIST' : k === 'r' ? 'R' : k === 'scipy-ref' ? 'scipy-ref' : 'analytical';
  };
  var tbody = document.getElementById('method-rows');
  if (tbody) {
    clear(tbody);
    (report.methods || []).forEach(function (m) {
      var tr = document.createElement('tr');

      var tdMethod = document.createElement('td');
      tdMethod.appendChild(el('strong', null, m.method));
      tdMethod.appendChild(el('code', null, m.endpoint));
      (m.references || []).forEach(function (x) {
        var r = el('div', null, x);
        r.style.fontSize = '.75rem';
        r.style.color = 'var(--text3)';
        r.style.marginTop = '4px';
        tdMethod.appendChild(r);
      });

      var tdKinds = document.createElement('td');
      (m.reference_kinds || []).forEach(function (k) {
        tdKinds.appendChild(el('span', 'chip ' + kindClass(k), kindLabel(k)));
      });

      var tdData = document.createElement('td');
      (m.datasets || []).forEach(function (x) {
        var ds = el('div', null, x);
        ds.style.fontSize = '.82rem';
        ds.style.color = 'var(--text3)';
        tdData.appendChild(ds);
      });

      var tdChecks = el('td', 'n-checks', m.n_checks);

      tr.appendChild(tdMethod);
      tr.appendChild(tdKinds);
      tr.appendChild(tdData);
      tr.appendChild(tdChecks);
      tbody.appendChild(tr);
    });
  }

  // ── Footer gen-time ──
  var gen = document.getElementById('gen-time');
  if (gen) {
    gen.textContent = isNaN(d.getTime())
      ? report.generated_at
      : d.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  }
})();
