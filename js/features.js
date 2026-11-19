/* Page script for features.html (moved out of inline <script> so the CSP needs no 'unsafe-inline'). */

// ── Feature Data (plain-language only) ────────────────────────────
const FEATURES = [
  {
    name: "Stats Studio",
    category: "Core Statistics",
    icon: "\u{1F4CA}",
    color: "#3B82F6",
    methodCount: 27,
    plain: "The central hub for classical statistics. Compare groups, predict outcomes, test relationships, and run non-parametric alternatives \u2014 all with effect sizes, confidence intervals, and APA-ready output."
  },
  {
    name: "Psychometrics",
    category: "Psychometrics & Measurement",
    icon: "\u{1F4C8}",
    color: "#8B5CF6",
    methodCount: 18,
    plain: "Everything for test and scale development: reliability, factor structure, item-level modeling (IRT), adaptive testing, measurement fairness across groups, and cutting-edge network psychometrics and diagnostic classification."
  },
  {
    name: "SEM",
    category: "Structural Equation Modeling",
    icon: "\u{1F517}",
    color: "#7C3AED",
    methodCount: 6,
    plain: "Build and test complex theoretical models combining latent variables (things you cannot directly measure) with observed variables. Includes two-level (multilevel) models for clustered data."
  },
  {
    name: "Bayesian Lab",
    category: "Bayesian Analysis",
    icon: "\u{1F52C}",
    color: "#06B6D4",
    methodCount: 4,
    plain: "An alternative to traditional p-values. Quantify evidence strength with Bayes factors, explore full posterior distributions via MCMC, compare competing models, and verify your model can reproduce realistic data."
  },
  {
    name: "Causal Inference",
    category: "Causal Inference",
    icon: "\u{1F9ED}",
    color: "#FB923C",
    methodCount: 11,
    plain: "Move beyond correlation to causation. Draw causal diagrams, identify what to control for, estimate causal effects with propensity scores or instrumental variables, and test how sensitive your conclusions are to hidden confounders."
  },
  {
    name: "Meta-Analysis",
    category: "Meta-Analysis",
    icon: "\u{1F500}",
    color: "#D946EF",
    methodCount: 8,
    plain: "Synthesize findings from multiple studies into one precise estimate. Detect publication bias, explore what makes effects bigger or smaller across studies, and convert between effect size metrics."
  },
  {
    name: "HLM (Hierarchical Linear Models)",
    category: "Hierarchical & Mixed Models",
    icon: "\u{1F4DA}",
    color: "#0891B2",
    methodCount: 4,
    plain: "Analyze nested data \u2014 students within schools, patients within hospitals \u2014 at multiple levels simultaneously. Model individual growth trajectories over time."
  },
  {
    name: "Time Series",
    category: "Time Series",
    icon: "\u{1F4C8}",
    color: "#F43F5E",
    methodCount: 6,
    plain: "Analyze and forecast data collected over time. Detect where trends shift, test whether one time series predicts another, and decompose signals into trend, seasonal, and residual components."
  },
  {
    name: "Machine Learning",
    category: "Machine Learning",
    icon: "\u{1F916}",
    color: "#38BDF8",
    methodCount: 12,
    plain: "Classify cases, discover natural groupings, reduce data dimensions, and compare model performance \u2014 using algorithms from logistic regression to Random Forest and SVM."
  },
  {
    name: "Medical Stats",
    category: "Medical & Clinical",
    icon: "\u{1FA7A}",
    color: "#14B8A6",
    methodCount: 6,
    plain: "Clinical research tools: survival curves, hazard modeling, treatment efficacy (NNT), risk calculations, and diagnostic test evaluation with ROC curves."
  },
  {
    name: "Text Analysis",
    category: "Text Analysis & NLP",
    icon: "\u{1F4DD}",
    color: "#A3E635",
    methodCount: 5,
    plain: "Analyze text for emotional tone, discover latent themes, categorize words into psychological dimensions, measure reading difficulty, and code content with custom categories."
  },
  {
    name: "Cognitive Lab",
    category: "Cognitive Lab",
    icon: "\u{1F9E0}",
    color: "#F97316",
    methodCount: 9,
    plain: "Score data from five classic cognitive tasks (attention, working memory, focus, impulse control, flexibility) that you ran in your own experiment software, plus Signal Detection Theory analysis to separate sensitivity from response bias."
  },
  {
    name: "Social Graph",
    category: "Social & Network Analysis",
    icon: "\u{1F465}",
    color: "#EC4899",
    methodCount: 4,
    plain: "Map and measure social relationships. Identify the most connected or influential individuals, detect communities, and model how networks form and change over time."
  },
  {
    name: "Criminology",
    category: "Criminology & Spatial",
    icon: "\u{1F5FA}",
    color: "#EF4444",
    methodCount: 7,
    plain: "Analyze crime patterns in space and time. Predict offender residence from crime locations, identify hotspots, and run spatial statistics to understand geographic variation."
  },
  {
    name: "Ethnography",
    category: "Ethnography & Qualitative",
    icon: "\u{1F4D6}",
    color: "#84CC16",
    methodCount: 5,
    plain: "Digital fieldwork toolkit: test whether informants share a cultural model, check coder agreement, analyze pile-sort similarity, count themes, and run Q-methodology for subjective viewpoints."
  },
  {
    name: "Genomics",
    category: "Genomics & Biometrics",
    icon: "\u{1F9EC}",
    color: "#10B981",
    methodCount: 4,
    plain: "Analyze genetic variants: allele frequencies, Hardy-Weinberg equilibrium, aggregate genetic risk, and genome-wide association testing."
  },
  {
    name: "BioMetrics",
    category: "Genomics & Biometrics",
    icon: "\u{1F49A}",
    color: "#F59E0B",
    methodCount: 3,
    plain: "Process physiological sensor data: heart rate variability for stress, skin conductance for emotional arousal, and accelerometer data for physical activity levels."
  },
  {
    name: "EMA / ESM",
    category: "EMA / ESM",
    icon: "\u{1F514}",
    color: "#E879F9",
    methodCount: 4,
    plain: "Analyze experience-sampling data: track how emotions fluctuate moment-to-moment, measure emotional inertia (how feelings carry over), and assess participant compliance."
  },
  {
    name: "Conjoint Analysis",
    category: "Conjoint Analysis",
    icon: "\u{2696}",
    color: "#C084FC",
    methodCount: 4,
    plain: "Understand how people make trade-offs between product features. Estimate the value of each attribute level, rank what matters most, and predict choices for new configurations."
  },
  {
    name: "Sociology",
    category: "Sociology",
    icon: "\u{1F30D}",
    color: "#4ADE80",
    methodCount: 6,
    plain: "Measure inequality, residential segregation, intergenerational mobility, and analyze complex survey data with proper weighting for representative estimates."
  },
  {
    name: "Political Science",
    category: "Political Science",
    icon: "\u{1F3F3}",
    color: "#60A5FA",
    methodCount: 5,
    plain: "Quasi-experimental designs for policy evaluation (RDD, diff-in-diff), legislator ideology estimation, vote pattern analysis, and panel data methods for comparative politics."
  },
  {
    name: "Effect Sizes",
    category: "Effect Sizes",
    icon: "\u{1F4CF}",
    color: "#2DD4BF",
    methodCount: 6,
    plain: "Compute, convert, and interpret effect sizes with domain-specific benchmarks. A \u201Cmedium\u201D effect in clinical psych differs from social psych \u2014 this tool knows the difference."
  },
  {
    name: "Data Prep",
    category: "Data Preparation",
    icon: "\u{1F527}",
    color: "#0EA5E9",
    methodCount: 10,
    plain: "Clean and prepare your data before analysis: detect outliers, test normality, apply transformations, diagnose missing data patterns, and impute missing values with MICE."
  },
  {
    name: "Robustness Checks",
    category: "Robustness Checks",
    icon: "\u{1F9EA}",
    color: "#F472B6",
    methodCount: 4,
    plain: "Four checks on how much a result depends on your choices. None is a new method; each packages published work. Summarise a multiverse of analyses (Specification Summary), compare simple and complex models (Complexity Ladder), see whether a p-value and a Bayes factor agree (Frequentist vs Bayesian), and see which correlations keep their sign when cases are dropped (Correlation Stability). All reuse one sample, so they describe sensitivity; they do not show a result will replicate."
  },
  {
    name: "Pipeline Builder",
    category: "Tools & Workflows",
    icon: "\u{1F504}",
    color: "#7C3AED",
    methodCount: 2,
    plain: "Chain multiple analysis steps into a reusable, reproducible workflow. Clean, check assumptions, analyze, and compute effect sizes \u2014 all in one automated sequence."
  },
  {
    name: "Multiverse Analysis",
    category: "Tools & Workflows",
    icon: "\u{1F30C}",
    color: "#E11D48",
    methodCount: 3,
    plain: "Runs every reasonable combination of analysis choices and shows all results at once. Reveals whether your conclusion depends on arbitrary decisions."
  },
  {
    name: "Dual Report",
    category: "Tools & Workflows",
    icon: "\u{1F501}",
    color: "#0D9488",
    methodCount: 2,
    plain: "Runs the same analysis two ways \u2014 traditional (p-values) and Bayesian (evidence strength) \u2014 side by side. Highlights when they agree or disagree."
  },
  {
    name: "Data Simulator",
    category: "Tools & Workflows",
    icon: "\u{1F3B2}",
    color: "#EA580C",
    methodCount: 3,
    plain: "Creates realistic fake datasets for teaching, testing, or planning. Specify the design and effect size; it generates matching data. Monte Carlo simulation estimates empirical power."
  },
  {
    name: "Pre-Registration Builder",
    category: "Tools & Workflows",
    icon: "\u{1F6E1}",
    color: "#4F46E5",
    methodCount: 3,
    plain: "Creates a locked-in study plan before data collection \u2014 hypotheses, methods, and analysis plan. Prevents HARKing and increases credibility."
  },
  {
    name: "Study Planner",
    category: "Tools & Workflows",
    icon: "\u{1F4CB}",
    color: "#6366F1",
    methodCount: 3,
    plain: "Plan your study: choose a design, calculate exactly how many participants you need, visualize your power curve, and check field-specific effect size benchmarks."
  },
  {
    name: "Curriculum",
    category: "Tools & Workflows",
    icon: "\u{1F393}",
    color: "#8B5CF6",
    methodCount: 3,
    plain: "Six self-paced courses (Psychometrics, Genetics, Bayesian, Social Networks, Geographic Profiling, Clinical Trials) with AI-generated lessons and progress tracking."
  },
  {
    name: "AI Statistical Advisor",
    category: "AI & Learning",
    icon: "\u{2728}",
    color: "#A855F7",
    methodCount: 4,
    plain: "Ask questions in plain English and get expert explanations, test recommendations, and result interpretations. Like a statistics professor available 24/7."
  },
  {
    name: "Interactive Learning",
    category: "AI & Learning",
    icon: "\u{25B6}",
    color: "#06B6D4",
    methodCount: 9,
    plain: "Interactive simulations and visualizations that build statistical intuition you can\u2019t get from a textbook. Touch, drag, and see statistics come alive."
  },
  {
    name: "Research Intelligence",
    category: "Research Intelligence",
    icon: "\u{1F6E1}",
    color: "#8B5CF6",
    methodCount: 15,
    plain: "Intelligent tools that automatically check your work, flag issues, and suggest improvements \u2014 like having a methodologist looking over your shoulder."
  },
  {
    name: "Research Lifecycle",
    category: "Research Tools",
    icon: "\u{1F680}",
    color: "#EC4899",
    methodCount: 11,
    plain: "Tools that cover the entire research lifecycle \u2014 from formulating your question to responding to reviewers. Everything a researcher needs without leaving the app."
  },
  {
    name: "Statistical Glossary",
    category: "Platform",
    icon: "\u{1F4D6}",
    color: "#8B5CF6",
    methodCount: 4,
    plain: "A searchable dictionary of 62 statistical terms organized by category, each with a definition, detailed explanation, formula, and which module uses it."
  },
  {
    name: "Interactive Tooltips & Cross-References",
    category: "Platform",
    icon: "\u{2139}",
    color: "#6366F1",
    methodCount: 2,
    plain: "Tap any statistical value for an instant explanation. Statistical terms are auto-linked to definitions throughout the app."
  },
  {
    name: "Assumption Dashboard",
    category: "Platform",
    icon: "\u{2705}",
    color: "#10B981",
    methodCount: 3,
    plain: "A visual report card for your analysis assumptions: green checkmarks for passes, red warnings for violations, with diagnostic plots and alternative test suggestions."
  },
  {
    name: "Sensitivity Analysis Panel",
    category: "Platform",
    icon: "\u{1F6E1}",
    color: "#F59E0B",
    methodCount: 2,
    plain: "Runs the same comparison four different ways (parametric, non-parametric, permutation, bootstrap) to check if they all agree."
  },
  {
    name: "Analysis History & Projects",
    category: "Platform",
    icon: "\u{1F552}",
    color: "#06B6D4",
    methodCount: 3,
    plain: "Every analysis is saved automatically. Browse, search, and filter past results. Organize work into projects with tags for easy retrieval."
  },
  {
    name: "Export (PDF, CSV, APA)",
    category: "Platform",
    icon: "\u{1F4E5}",
    color: "#3B82F6",
    methodCount: 3,
    plain: "Generate print-ready PDF reports, download spreadsheet files, or copy APA-formatted results to paste directly into your manuscript."
  },
  {
    name: "Result Knowledge Graph",
    category: "Platform",
    icon: "\u{1F578}",
    color: "#EC4899",
    methodCount: 2,
    plain: "Links your analysis results across modules into an interactive network diagram, showing how one analysis feeds into or validates the next."
  },
  {
    name: "Dark Theme & Accessibility",
    category: "Platform",
    icon: "\u{1F319}",
    color: "#475569",
    methodCount: 3,
    plain: "Dark-first design with optional light mode. All modules have unique signature colors. Full accessibility labels throughout."
  },
  {
    name: "Offline Detection",
    category: "Platform",
    icon: "\u{1F4F4}",
    color: "#EF4444",
    methodCount: 2,
    plain: "A banner warns you when you lose internet or cannot reach the analysis server, so you know if results might not be available."
  },
  {
    name: "UX & Accessibility",
    category: "UX & Accessibility",
    icon: "\u{267F}",
    color: "#F59E0B",
    methodCount: 20,
    plain: "Every interaction is polished \u2014 copy results with a tap, recover from mistakes, never lose your work. Fully accessible with colorblind palettes, font scaling, high contrast, and screen reader support."
  }
];

const CATEGORIES = [
  "All",
  "Core Statistics",
  "Psychometrics & Measurement",
  "Structural Equation Modeling",
  "Bayesian Analysis",
  "Causal Inference",
  "Meta-Analysis",
  "Hierarchical & Mixed Models",
  "Time Series",
  "Machine Learning",
  "Medical & Clinical",
  "Text Analysis & NLP",
  "Cognitive Lab",
  "Social & Network Analysis",
  "Criminology & Spatial",
  "Ethnography & Qualitative",
  "Genomics & Biometrics",
  "EMA / ESM",
  "Conjoint Analysis",
  "Sociology",
  "Political Science",
  "Effect Sizes",
  "Data Preparation",
  "Robustness Checks",
  "Tools & Workflows",
  "AI & Learning",
  "Research Intelligence",
  "Research Tools",
  "Platform",
  "UX & Accessibility"
];

// ── State ──────────────────────────────────────────────────────────
let activeCategory = "All";
let searchQuery = "";

// ── Count methods ──────────────────────────────────────────────────
const totalMethods = FEATURES.reduce((sum, f) => sum + f.methodCount, 0);
document.getElementById("methodCount").textContent = totalMethods + "+";
document.getElementById("moduleCount").textContent = FEATURES.length;

// ── Render category tabs ───────────────────────────────────────────
const tabsContainer = document.getElementById("categoryTabs");
CATEGORIES.forEach(cat => {
  const btn = document.createElement("button");
  btn.className = "cat-tab" + (cat === "All" ? " active" : "");
  btn.textContent = cat;
  btn.addEventListener("click", () => {
    activeCategory = cat;
    document.querySelectorAll(".cat-tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderCards();
  });
  tabsContainer.appendChild(btn);
});

// ── Search ─────────────────────────────────────────────────────────
const searchInput = document.getElementById("searchInput");
searchInput.addEventListener("input", e => {
  searchQuery = e.target.value.toLowerCase();
  renderCards();
});

// ── Filter logic ───────────────────────────────────────────────────
function getFiltered() {
  return FEATURES.filter(f => {
    const catMatch = activeCategory === "All" || f.category === activeCategory;
    if (!catMatch) return false;
    if (!searchQuery) return true;
    const q = searchQuery;
    return f.name.toLowerCase().includes(q)
        || f.category.toLowerCase().includes(q)
        || f.plain.toLowerCase().includes(q);
  });
}

// ── Render cards ───────────────────────────────────────────────────
const grid = document.getElementById("featuresGrid");
const countEl = document.getElementById("searchCount");

function renderCards() {
  const filtered = getFiltered();
  const methodTotal = filtered.reduce((s, f) => s + f.methodCount, 0);
  countEl.textContent = "Showing " + filtered.length + " module" + (filtered.length !== 1 ? "s" : "") + " with " + methodTotal + " methods";

  grid.innerHTML = "";

  if (filtered.length === 0) {
    grid.innerHTML = '<div class="no-results"><p>No modules match your search.</p><p style="font-size:.85rem">Try a different keyword or category.</p></div>';
    return;
  }

  filtered.forEach(function(f, i) {
    var card = document.createElement("div");
    card.className = "feature-card";
    card.style.transitionDelay = Math.min(i * 40, 600) + "ms";

    card.innerHTML = '<div class="card-header">' +
      '<div class="card-icon" style="background:' + f.color + '20; color:' + f.color + '">' + f.icon + '</div>' +
      '<div>' +
        '<div class="card-title">' + f.name + '</div>' +
        '<div class="card-cat">' + f.category + '</div>' +
      '</div>' +
    '</div>' +
    '<div class="card-method-count">' + f.methodCount + ' method' + (f.methodCount !== 1 ? 's' : '') + '</div>' +
    '<div class="card-plain">' + f.plain + '</div>';

    grid.appendChild(card);
  });

  // Trigger scroll reveal
  requestAnimationFrame(function() { observeCards(); });
}

// ── Scroll reveal ──────────────────────────────────────────────────
function observeCards() {
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll(".feature-card:not(.visible)").forEach(function(card) {
    observer.observe(card);
  });
}

// ── Initial render ─────────────────────────────────────────────────
renderCards();

