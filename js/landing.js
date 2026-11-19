/* Page script for landing.html (moved out of inline <script> so the CSP needs no 'unsafe-inline'). */

        // Auto-updates whenever the page is edited & deployed.
        (function(){
          var d = new Date(document.lastModified || Date.now());
          var el = document.getElementById('compare-last-reviewed');
          if (el) el.textContent = 'Last reviewed ' + d.toISOString().slice(0,10) + '. ';
        })();

  (function () {
    var form = document.getElementById('betaForm');
    if (!form || !window.fetch) return;
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var btn = form.querySelector('button[type=submit]');
      var msg = document.getElementById('betaFormMsg');
      btn.disabled = true; msg.textContent = '';
      fetch(form.action, { method: 'POST', body: new FormData(form), headers: { Accept: 'application/json' } })
        .then(function (r) {
          if (!r.ok) throw new Error();
          form.reset();
          msg.style.color = '#10B981';
          msg.textContent = "Thanks! We'll email you a beta code when a spot opens.";
        })
        .catch(function () {
          msg.style.color = '#EF4444';
          msg.textContent = 'Something went wrong. Please try again.';
        })
        .finally(function () { btn.disabled = false; });
    });
  })();

// ── Module grid ─────────────────────────────────────────────────
const MODULES = [
  { cat: 'Core Statistics', items: [
    { name: 'Stats Studio', sub: 'T-tests, ANOVA, regression, power', icon: '\u{1F4CA}', color: '#3B82F6' },
    { name: 'Psychometrics', sub: 'Cronbach\'s \u03B1, factor analysis, IRT', icon: '\u{1F4CF}', color: '#8B5CF6' },
    { name: 'Bayesian Lab', sub: 'Bayes factors, MCMC, posteriors', icon: '\u{1F52C}', color: '#06B6D4' },
    { name: 'Meta-Analysis', sub: 'Effect pooling, forest plots, bias', icon: '\u{1F4DA}', color: '#D946EF' },
    { name: 'Causal Inference', sub: 'DAGs, IV, g-computation', icon: '\u{1F517}', color: '#FB923C' },
    { name: 'Multiverse', sub: 'Specification curves, robustness', icon: '\u{1F30D}', color: '#E11D48' },
    { name: 'Dual Report', sub: 'Frequentist + Bayesian side-by-side', icon: '\u2194\uFE0F', color: '#0D9488' },
    { name: 'Effect Sizes', sub: 'Calculator, converter, interpreter', icon: '\u{1F4D0}', color: '#2DD4BF' },
  ]},
  { cat: 'Specialized Domains', items: [
    { name: 'Genomics', sub: 'SNP, HWE, polygenic risk', icon: '\u{1F9EC}', color: '#10B981' },
    { name: 'Criminology', sub: 'Geographic profiling, recidivism', icon: '\u{1F5FA}\uFE0F', color: '#EF4444' },
    { name: 'Social Graph', sub: 'Network metrics, diffusion', icon: '\u{1F465}', color: '#EC4899' },
    { name: 'Medical Stats', sub: 'NNT, ROC, survival, Bland-Altman', icon: '\u{1FA7A}', color: '#14B8A6' },
    { name: 'Time Series', sub: 'ARIMA, VAR, changepoints', icon: '\u{1F4C8}', color: '#F43F5E' },
    { name: 'Text Analysis', sub: 'Sentiment, topics, LIWC, readability', icon: '\u{1F4DD}', color: '#A3E635' },
    { name: 'Machine Learning', sub: 'Classification, clustering, PCA', icon: '\u{1F916}', color: '#38BDF8' },
  ]},
  { cat: 'Data Collection', items: [
    { name: 'BioMetrics', sub: 'HRV, EDA, accelerometer', icon: '\u{1F493}', color: '#F59E0B' },
    { name: 'Cognitive Lab', sub: 'Score Stroop, N-Back, Flanker, Go/No-Go data', icon: '\u{1F4A1}', color: '#F97316' },
    { name: 'Ethnography', sub: 'Field notes, codebook, consensus', icon: '\u{1F4D3}', color: '#84CC16' },
  ]},
  { cat: 'Tools & Learning', items: [
    { name: 'Data Prep', sub: 'Outliers, normality, MICE', icon: '\u{1F9F9}', color: '#0EA5E9' },
    { name: 'Study Planner', sub: 'Power analysis, sample size', icon: '\u{1F4CB}', color: '#6366F1' },
    { name: 'Curriculum', sub: '6 interactive courses', icon: '\u{1F393}', color: '#8B5CF6' },
    { name: 'Pipeline Builder', sub: 'Reproducible workflows', icon: '\u{1F504}', color: '#7C3AED' },
    { name: 'Data Simulator', sub: 'Synthetic data, Monte Carlo', icon: '\u{1F3B2}', color: '#EA580C' },
    { name: 'Pre-Registration', sub: 'OSF / AsPredicted templates', icon: '\u{1F6E1}\uFE0F', color: '#4F46E5' },
  ]},
  { cat: 'Research Intelligence', items: [
    { name: 'Adversarial Reviewer', sub: 'AI rigor grading A\u2013F', icon: '\u{1F3AF}', color: '#E11D48' },
    { name: 'Method Triangulation', sub: '3 methods, check convergence', icon: '\u{1F52C}', color: '#7C3AED' },
    { name: 'Assumption Auto-Gate', sub: 'Pre-analysis checks', icon: '\u{1F6A7}', color: '#F59E0B' },
    { name: 'Contradiction Detector', sub: 'Cross-analysis consistency', icon: '\u26A0\uFE0F', color: '#EF4444' },
    { name: 'Equivalence Testing', sub: 'TOST, equivalence bounds', icon: '\u2194\uFE0F', color: '#0D9488' },
  ]},
  { cat: 'Research Lifecycle', items: [
    { name: 'Question Refiner', sub: '5-step hypothesis wizard', icon: '\u{1F4A1}', color: '#8B5CF6' },
    { name: 'Measurement Guide', sub: '150+ validated instruments', icon: '\u{1F4CF}', color: '#06B6D4' },
    { name: 'IRB/Ethics Generator', sub: 'AI-powered ethics drafts', icon: '\u{1F4DC}', color: '#10B981' },
    { name: 'Narrative Builder', sub: 'Auto APA method/results', icon: '\u{1F4DD}', color: '#3B82F6' },
    { name: 'Code Export', sub: 'R & Python for 17 tests', icon: '\u{1F4BB}', color: '#FB923C' },
  ]},
];

const grid = document.getElementById('module-grid');
MODULES.forEach(cat => {
  const row = document.createElement('div');
  row.className = 'cat-row';
  row.innerHTML = `<div class="cat-label">${cat.cat}</div><div class="cat-grid">${
    cat.items.map(m => `
      <div class="mod-card" style="--icon-color:${m.color}">
        <div class="mod-icon" style="background:${m.color}20">${m.icon}</div>
        <div>
          <div class="mod-name">${m.name}</div>
          <div class="mod-sub">${m.sub}</div>
        </div>
      </div>
    `).join('')
  }</div>`;
  grid.appendChild(row);
});

// ── Ticker ───────────────────────────────────────────────────────
const TICKS = [
  'T-Tests', 'ANOVA', 'Regression', 'SEM', 'CFA', 'IRT', 'MCMC', 'Bayes Factors',
  'Meta-Analysis', 'Forest Plots', 'Funnel Plots', 'Power Analysis', 'Effect Sizes',
  'Multiverse', 'Propensity Score', 'Kaplan-Meier', 'Cox Regression', 'HLM',
  'ARIMA', 'DAG Analysis', 'ROC Curves', 'Bland-Altman', 'Geographic Profiling',
  'Network Centrality', 'Sentiment Analysis', 'Topic Modeling', 'Machine Learning',
  'Bootstrap CI', 'Permutation Tests', 'MICE Imputation', 'Cronbach\'s Alpha',
  'Measurement Invariance', 'Latent Growth Curves',
  'Adversarial Reviewer', 'Method Triangulation', 'Equivalence Testing',
  'IRB Generator', 'Narrative Builder', 'Code Export (R/Python)',
  'ELI5 Mode', 'CLT Simulation', 'Effect Size Benchmarks',
  'Variable Tracker', 'Journal Formatting', 'Replication Package',
];
const tickerEl = document.getElementById('ticker');
const tickerHTML = TICKS.map(t => `<div class="tick-item"><span class="dot"></span>${t}</div>`).join('');
tickerEl.innerHTML = tickerHTML + tickerHTML; // duplicate for seamless loop

// ── FAQ toggle ──────────────────────────────────────────────────
function toggleFaq(btn) {
  const answer = btn.nextElementSibling;
  const open = answer.classList.contains('open');
  // close all
  document.querySelectorAll('.faq-a').forEach(a => a.classList.remove('open'));
  document.querySelectorAll('.faq-q').forEach(q => q.setAttribute('aria-expanded', 'false'));
  if (!open) {
    answer.classList.add('open');
    btn.setAttribute('aria-expanded', 'true');
  }
}

// ── Scroll reveal ───────────────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('revealed');
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));


// FAQ buttons (were inline onclick handlers)
document.querySelectorAll('.faq-q').forEach(function (btn) {
  btn.addEventListener('click', function () { toggleFaq(btn); });
});
