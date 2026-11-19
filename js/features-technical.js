/* Page script for features-technical.html (moved out of inline <script> so the CSP needs no 'unsafe-inline'). */

// ── Feature Data ───────────────────────────────────────────────────
const FEATURES = [
  {
    name: "Stats Studio",
    category: "Core Statistics",
    icon: "\u{1F4CA}",
    color: "#3B82F6",
    methods: ["Independent-samples t-test","Paired-samples t-test","One-way ANOVA","Mixed ANOVA","MANOVA","Chi-square test of independence","Linear regression","Multiple regression","Power analysis","Descriptive statistics","Correlation matrix","Simple mediation","Parallel mediation","Serial mediation","Moderation","Moderated mediation","Mann-Whitney U","Wilcoxon signed-rank","Kruskal-Wallis","Friedman test","Bootstrap CI (percentile & BCa)","Permutation tests","Logistic regression","ROC-AUC analysis","Bland-Altman agreement","Correspondence analysis","MCA"],
    plain: "The central hub for classical statistics. Compare groups, predict outcomes, test relationships, and run non-parametric alternatives \u2014 all with effect sizes, confidence intervals, and APA-ready output.",
    technical: "T-tests (Student/Welch), F-tests (one-way, mixed, MANOVA), OLS/logistic regression, chi-square, Pearson/Spearman/Kendall correlations, mediation (Baron & Kenny + Sobel), moderation/moderated-mediation, non-parametric suite (Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman), bootstrap CI, permutation tests, ROC-AUC, Bland-Altman LOA, CA/MCA. Full diagnostics: VIF, Cook\u2019s D, Durbin-Watson, Levene, Mauchly."
  },
  {
    name: "Psychometrics",
    category: "Psychometrics & Measurement",
    icon: "\u{1F4C8}",
    color: "#8B5CF6",
    methods: ["Cronbach\u2019s alpha (Feldt CI)","Exploratory Factor Analysis (maximum likelihood)","Confirmatory Factor Analysis (CFA)","IRT \u2014 Rasch / 1PL","IRT \u2014 2PL","IRT \u2014 3PL","IRT \u2014 Graded Response Model (GRM)","IRT \u2014 Partial Credit Model (PCM)","Differential Item Functioning (DIF)","Computerized Adaptive Testing (CAT) simulation","Test equating","Measurement invariance (configural, metric, scalar, strict)","Latent growth curves","Generalizability theory (G-study & D-study)","Latent Transition Analysis (LTA)","MTMM (Campbell & Fiske + CFA approach)","Network psychometrics (estimation, centrality, stability, bridge, comparison)","Diagnostic classification models (DINA, DINO, G-DINA, Q-matrix validation)"],
    plain: "Everything for test and scale development: reliability, factor structure, item-level modeling (IRT), adaptive testing, measurement fairness across groups, and cutting-edge network psychometrics and diagnostic classification.",
    technical: "Cronbach\u2019s alpha (raw and standardized, Feldt CI) with corrected item-total r and alpha-if-deleted; maximum-likelihood EFA as R\u2019s factanal (varimax/promax, likelihood-ratio test, KMO/MSA, Bartlett sphericity); IRT by marginal maximum likelihood (Rasch, 2PL, 3PL with a Beta prior on guessing, graded response, partial credit; EAP scores, infit/outfit), checked against mirt; DIF by Mantel-Haenszel, logistic regression (Nagelkerke \u0394R\u00B2) and Lord\u2019s chi-square with iterative purification, checked against difR; CAT simulation (maximum-information selection, EAP scoring); IRT linking (mean-mean, mean-sigma, Haebara, Stocking-Lord), checked against equateIRT; CFA, path models and latent growth curves by ML with lavaan\u2019s defaults (CFI/TLI/RMSEA/SRMR), multi-group invariance from configural to strict with partial invariance and alignment, checked against lavaan and sirt; G-theory G and D studies (p\u00D7i, p\u00D7i\u00D7o), checked against lme4; latent transition analysis (EM; time-specific or stationary transitions; mover-stayer), checked against depmixS4; MTMM matrices; network psychometrics as qgraph (EBICglasso, strength/closeness/betweenness/expected influence), bridge centrality, case-dropping CS-coefficient and the Network Comparison Test, checked against qgraph, networktools and NetworkComparisonTest; DCMs (DINA, DINO, G-DINA by EM, likelihood-ratio comparison, PVAF Q-matrix validation), checked against GDINA."
  },
  {
    name: "SEM",
    category: "Structural Equation Modeling",
    icon: "\u{1F517}",
    color: "#7C3AED",
    methods: ["Confirmatory Factor Analysis (CFA)","Path analysis","Full structural equation model","Multilevel SEM","Cross-level mediation","Measurement invariance"],
    plain: "Build and test complex theoretical models combining latent variables (things you cannot directly measure) with observed variables. Includes two-level (multilevel) models for clustered data.",
    technical: "ML estimation with lavaan\u2019s defaults, measurement + structural models, direct/indirect/total effects with delta-method SEs, fit indices (CFI, TLI, RMSEA with 90% CI, SRMR), multi-group measurement invariance; two-level SEM by full maximum likelihood (path models, CFA, 1-1-1 / 2-1-1 / 1-1-2 mediation with Monte Carlo CIs), checked against lavaan; equivalent R (lavaan) code templates."
  },
  {
    name: "Bayesian Lab",
    category: "Bayesian Analysis",
    icon: "\u{1F52C}",
    color: "#06B6D4",
    methods: ["Bayes factors (JZS prior)","MCMC posterior sampling (Metropolis-Hastings)","Bayesian model comparison (BIC weights)","Posterior predictive checks"],
    plain: "An alternative to traditional p-values. Quantify evidence strength with Bayes factors, explore full posterior distributions via MCMC, compare competing models, and verify your model can reproduce realistic data.",
    technical: "JZS Bayes Factor (numerical integration, Cauchy prior) for t-tests, correlations, proportions. MCMC (Metropolis-Hastings, multiple chains, R-hat, ESS, trace/autocorrelation plots, 95% HDI). BIC-based model comparison with weights. Posterior predictive checks with Bayesian p-values and discrepancy measures."
  },
  {
    name: "Causal Inference",
    category: "Causal Inference",
    icon: "\u{1F9ED}",
    color: "#FB923C",
    methods: ["DAG visualization & analysis","Backdoor adjustment sets","Instrumental variables / 2SLS estimation","E-value sensitivity analysis","G-computation (standardization)","Doubly robust AIPW","Propensity score estimation","Propensity score matching","IPTW (inverse probability of treatment weighting)","Overlap assessment","Covariate balance diagnostics"],
    plain: "Move beyond correlation to causation. Draw causal diagrams, identify what to control for, estimate causal effects with propensity scores or instrumental variables, and test how sensitive your conclusions are to hidden confounders.",
    technical: "DAG with backdoor criterion, collider detection, minimal adjustment sets. IV/2SLS with first-stage F-statistic. E-value sensitivity for unmeasured confounding. G-computation for ATE. AIPW combining propensity + outcome models (sandwich SE). Propensity score suite: logistic estimation, nearest-neighbor matching, IPTW, overlap/trimming, SMD balance diagnostics."
  },
  {
    name: "Meta-Analysis",
    category: "Meta-Analysis",
    icon: "\u{1F500}",
    color: "#D946EF",
    methods: ["Fixed-effects model","Random-effects model (DerSimonian-Laird)","Egger\u2019s regression test","Trim-and-fill","PET-PEESE","Moderator analysis (meta-regression)","Cumulative meta-analysis","Effect size conversion"],
    plain: "Synthesize findings from multiple studies into one precise estimate. Detect publication bias, explore what makes effects bigger or smaller across studies, and convert between effect size metrics.",
    technical: "Inverse-variance pooling, DerSimonian-Laird random-effects, Q/I\u00B2/\u03C4\u00B2 heterogeneity, forest plot. Publication bias: Egger\u2019s regression, trim-and-fill, PET-PEESE. Meta-regression with study-level covariates, subgroup Q-test. Cumulative meta-analysis (studies added chronologically). Effect size conversion utilities."
  },
  {
    name: "HLM (Hierarchical Linear Models)",
    category: "Hierarchical & Mixed Models",
    icon: "\u{1F4DA}",
    color: "#0891B2",
    methods: ["2-level HLM","3-level HLM","Growth curve models","Intraclass Correlation (ICC)"],
    plain: "Analyze nested data \u2014 students within schools, patients within hospitals \u2014 at multiple levels simultaneously. Model individual growth trajectories over time.",
    technical: "Linear mixed-effects via REML/ML. 2-level and 3-level models with random intercepts and slopes, cross-level interactions. Growth curves with time-coded loadings. ICC for variance partitioning."
  },
  {
    name: "Time Series",
    category: "Time Series",
    icon: "\u{1F4C8}",
    color: "#F43F5E",
    methods: ["ARIMA (auto-order selection)","VAR (vector autoregression)","Interrupted time series (segmented regression)","STL decomposition","Change-point detection","Granger causality"],
    plain: "Analyze and forecast data collected over time. Detect where trends shift, test whether one time series predicts another, and decompose signals into trend, seasonal, and residual components.",
    technical: "ARIMA(p,d,q) with ADF/KPSS stationarity tests, VAR with Granger causality and impulse response functions, interrupted time series (segmented regression with level/slope change), STL seasonal-trend decomposition, change-point detection, auto-order selection via AIC/BIC."
  },
  {
    name: "Machine Learning",
    category: "Machine Learning",
    icon: "\u{1F916}",
    color: "#38BDF8",
    methods: ["Logistic regression classifier","Random Forest","Support Vector Machine (SVM)","k-Nearest Neighbors (kNN)","Decision Tree","K-Means clustering","Hierarchical clustering","DBSCAN","PCA (dimensionality reduction)","t-SNE (dimensionality reduction)","Feature selection","Cross-validation & model comparison"],
    plain: "Classify cases, discover natural groupings, reduce data dimensions, and compare model performance \u2014 using algorithms from logistic regression to Random Forest and SVM.",
    technical: "Classification (Logistic, RF, SVM, kNN, Decision Tree) with accuracy, precision, recall, F1, AUC-ROC. Clustering (K-Means, hierarchical/agglomerative, DBSCAN). Dimensionality reduction (PCA scree + loadings, t-SNE 2D embedding). Feature importance ranking, k-fold cross-validation, model comparison dashboard."
  },
  {
    name: "Medical Stats",
    category: "Medical & Clinical",
    icon: "\u{1FA7A}",
    color: "#14B8A6",
    methods: ["Kaplan-Meier survival analysis","Cox proportional hazards regression","Number Needed to Treat (NNT)","Risk ratios (RR, OR, ARR, RRR)","Diagnostic accuracy (sensitivity, specificity)","ROC curve analysis"],
    plain: "Clinical research tools: survival curves, hazard modeling, treatment efficacy (NNT), risk calculations, and diagnostic test evaluation with ROC curves.",
    technical: "Kaplan-Meier estimator (lifelines) with log-rank test, Cox PH with concordance index and proportionality check, NNT/ARR/RRR, OR/RR with CI, diagnostic panel (sensitivity, specificity, PPV, NPV, LR+/\u2212, Youden\u2019s J), ROC-AUC with bootstrap CI."
  },
  {
    name: "Text Analysis",
    category: "Text Analysis & NLP",
    icon: "\u{1F4DD}",
    color: "#A3E635",
    methods: ["Sentiment analysis","Topic modeling (NMF)","LIWC-style category analysis","Readability indices","Content coding"],
    plain: "Analyze text for emotional tone, discover latent themes, categorize words into psychological dimensions, measure reading difficulty, and code content with custom categories.",
    technical: "Sentiment scoring (valence), NMF topic modeling with coherence scores, LIWC-style dictionary-based category frequencies, 6 readability indices (Flesch, Gunning Fog, SMOG, Coleman-Liau, ARI, Dale-Chall), word stats (TTR, hapax, Yule\u2019s K), custom category-keyword co-occurrence coding."
  },
  {
    name: "Cognitive Lab",
    category: "Cognitive Lab",
    icon: "\u{1F9E0}",
    color: "#F97316",
    methods: ["Stroop task","N-Back task","Flanker task","Go/No-Go task","Task Switching","Signal Detection Theory \u2014 d\u2019 (d-prime)","Signal Detection Theory \u2014 criterion (c)","Signal Detection Theory \u2014 A\u2019","Signal Detection Theory \u2014 B\u2032\u2032D"],
    plain: "Score data from five classic cognitive tasks (attention, working memory, focus, impulse control, flexibility) that you ran in your own experiment software, plus Signal Detection Theory analysis to separate sensitivity from response bias.",
    technical: "Scores pasted trial-level data (one participant per run): Stroop interference and Flanker congruency effect (incongruent \u2212 congruent RT on correct trials, Welch t), task-switching switch cost (RT and accuracy), N-Back and Go/No-Go d\u2019 and criterion c with the log-linear correction (Hautus, 1995), commission/omission errors. SDT from counts: d\u2019, c, \u03b2, A\u2019, B\u2032\u2032D."
  },
  {
    name: "Social Graph",
    category: "Social & Network Analysis",
    icon: "\u{1F465}",
    color: "#EC4899",
    methods: ["Network centrality (degree, betweenness, closeness)","Community detection","ERGM (exponential random graph model)","SIENA (Simulation Investigation for Empirical Network Analysis)"],
    plain: "Map and measure social relationships. Identify the most connected or influential individuals, detect communities, and model how networks form and change over time.",
    technical: "Network metrics (density, clustering coefficient, components) via networkx. Centrality measures (degree, betweenness, closeness, eigenvector). Community detection algorithms. ERGM for network structure modeling. SIENA for longitudinal network dynamics and co-evolution of networks and behavior."
  },
  {
    name: "Criminology",
    category: "Criminology & Spatial",
    icon: "\u{1F5FA}",
    color: "#EF4444",
    methods: ["Geographic profiling (Rossmo\u2019s CGT)","Spatial clustering (K-means)","Temporal pattern analysis","Moran\u2019s I (global spatial autocorrelation)","LISA (local spatial autocorrelation)","Spatial regression","Geographically Weighted Regression (GWR)"],
    plain: "Analyze crime patterns in space and time. Predict offender residence from crime locations, identify hotspots, and run spatial statistics to understand geographic variation.",
    technical: "Rossmo CGT formula for geographic profiling, K-means spatial clustering, temporal decomposition. Spatial statistics: Moran\u2019s I (global autocorrelation), LISA (local indicators of spatial association), spatial lag/error regression, GWR for spatially varying coefficients."
  },
  {
    name: "Ethnography",
    category: "Ethnography & Qualitative",
    icon: "\u{1F4D6}",
    color: "#84CC16",
    methods: ["Cultural consensus analysis","Inter-coder reliability (Cohen\u2019s kappa, Krippendorff\u2019s alpha)","Pile-sort analysis","Thematic frequency analysis","Q-methodology"],
    plain: "Digital fieldwork toolkit: test whether informants share a cultural model, check coder agreement, analyze pile-sort similarity, count themes, and run Q-methodology for subjective viewpoints.",
    technical: "Cultural consensus factor model (eigenvalue ratio > 3:1), intercoder reliability (percent agreement, Cohen\u2019s kappa, Krippendorff\u2019s alpha), pile-sort proximity matrices, thematic frequency with code co-occurrence, Q-methodology (Q-sort, factor extraction, distinguishing statements)."
  },
  {
    name: "Genomics",
    category: "Genomics & Biometrics",
    icon: "\u{1F9EC}",
    color: "#10B981",
    methods: ["SNP analysis (allele/genotype frequencies)","Hardy-Weinberg equilibrium test","Polygenic risk scores","GWAS (genome-wide association)"],
    plain: "Analyze genetic variants: allele frequencies, Hardy-Weinberg equilibrium, aggregate genetic risk, and genome-wide association testing.",
    technical: "Allele/genotype frequency tables, HWE chi-square test, polygenic risk scores (weighted dosage sums, percentile mapping), GWAS allelic association with 5\u00D710\u207B\u2078 genome-wide significance threshold."
  },
  {
    name: "BioMetrics",
    category: "Genomics & Biometrics",
    icon: "\u{1F49A}",
    color: "#F59E0B",
    methods: ["HRV analysis (time & frequency domain)","EDA / skin conductance analysis","Accelerometry (activity levels)"],
    plain: "Process physiological sensor data: heart rate variability for stress, skin conductance for emotional arousal, and accelerometer data for physical activity levels.",
    technical: "HRV time-domain (SDNN, RMSSD, pNN50) and frequency-domain (VLF, LF, HF power). EDA: tonic SCL and phasic SCR peak detection. Accelerometer: SMA, dominant frequency, epoch-level activity classification. Multi-sensor session summaries."
  },
  {
    name: "EMA / ESM",
    category: "EMA / ESM",
    icon: "\u{1F514}",
    color: "#E879F9",
    methods: ["Descriptive dynamics","Lagged analysis (emotional inertia)","Affective dynamics (MSSD, PAC)","Compliance analysis"],
    plain: "Analyze experience-sampling data: track how emotions fluctuate moment-to-moment, measure emotional inertia (how feelings carry over), and assess participant compliance.",
    technical: "Descriptive dynamics (person-level means, SDs, within-person variability). Lagged models for emotional inertia (autoregression). Affective dynamics: MSSD (mean squared successive difference), PAC (probability of acute change). Compliance rates, response latency, and signal-contingent analysis."
  },
  {
    name: "Conjoint Analysis",
    category: "Conjoint Analysis",
    icon: "\u{2696}",
    color: "#C084FC",
    methods: ["Choice-based conjoint (CBC)","Part-worth utilities estimation","Attribute importance ranking","Choice prediction / market simulation"],
    plain: "Understand how people make trade-offs between product features. Estimate the value of each attribute level, rank what matters most, and predict choices for new configurations.",
    technical: "Choice-based conjoint with multinomial logit. Part-worth utility estimation via maximum likelihood. Relative attribute importance from utility ranges. Choice simulation / market share prediction for hypothetical profiles."
  },
  {
    name: "Sociology",
    category: "Sociology",
    icon: "\u{1F30D}",
    color: "#4ADE80",
    methods: ["Gini coefficient","Atkinson index","Theil index","Segregation indices","Social mobility tables","Survey weights (design effects)"],
    plain: "Measure inequality, residential segregation, intergenerational mobility, and analyze complex survey data with proper weighting for representative estimates.",
    technical: "Inequality: Gini coefficient, Atkinson index (epsilon parameter), Theil (GE family). Segregation: dissimilarity, isolation, information theory indices. Social mobility: transition matrices, odds ratios, Altham distance. Survey: design weights, raking, design effects (DEFF), Taylor linearization for variance estimation."
  },
  {
    name: "Political Science",
    category: "Political Science",
    icon: "\u{1F3F3}",
    color: "#60A5FA",
    methods: ["Regression discontinuity design (RDD)","Difference-in-differences (DiD)","Ideal point scaling (IRT-based)","Vote scaling","Time-series cross-section (TSCS) panels"],
    plain: "Quasi-experimental designs for policy evaluation (RDD, diff-in-diff), legislator ideology estimation, vote pattern analysis, and panel data methods for comparative politics.",
    technical: "RDD: sharp/fuzzy designs with local polynomial regression, bandwidth selection (IK/CCT), McCrary density test. DiD: canonical 2x2, staggered treatment (Callaway-Sant\u2019Anna), parallel trends testing. Ideal points via IRT-based scaling. TSCS panels: fixed/random effects, PCSE, Driscoll-Kraay SE."
  },
  {
    name: "Effect Sizes",
    category: "Effect Sizes",
    icon: "\u{1F4CF}",
    color: "#2DD4BF",
    methods: ["Effect size calculator (12+ metrics)","Effect size converter","Effect size interpreter (field benchmarks)","NNT from effect size","Power-from-effect-size","Field-specific benchmarks (Cohen, Hattie, Richard)"],
    plain: "Compute, convert, and interpret effect sizes with domain-specific benchmarks. A \u201Cmedium\u201D effect in clinical psych differs from social psych \u2014 this tool knows the difference.",
    technical: "12+ metrics: Cohen\u2019s d, Hedges\u2019 g, Glass\u2019s \u0394, r, OR, \u03B7\u00B2, partial \u03B7\u00B2, \u03C9\u00B2, f, f\u00B2, w, h, CLES. Bidirectional conversion matrix. NNT derivation. Power computation from observed effect. Domain benchmarks (Cohen 1988, Hattie 2009, Richard et al. 2003)."
  },
  {
    name: "Data Prep",
    category: "Data Preparation",
    icon: "\u{1F527}",
    color: "#0EA5E9",
    methods: ["Outlier detection (IQR, Z, Modified Z, Grubbs)","Normality testing (Shapiro-Wilk, D\u2019Agostino, KS, Anderson-Darling)","Transformations (log, sqrt, Box-Cox, z-score, rank)","Missing data analysis (Little\u2019s MCAR test)","MICE imputation (Rubin\u2019s rules, FMI)","Visual Spreadsheet Data Editor","Column type detection (numeric/categorical/ordinal)","Cell-level editing with sort and filter","CSV import/export with auto-parsing","Data Simulator (between/within/correlation/regression)"],
    plain: "Clean and prepare your data before analysis: detect outliers, test normality, apply transformations, diagnose missing data patterns, and impute missing values with MICE.",
    technical: "Outlier methods: IQR fence, Z-score, Modified Z (MAD-based), Grubbs test. Normality: Shapiro-Wilk, D\u2019Agostino-Pearson, Kolmogorov-Smirnov, Anderson-Darling. Transforms: log, sqrt, Box-Cox, z-standardization, rank-based inverse normal. Missing data: Little\u2019s MCAR test, MICE with Rubin\u2019s pooling rules and fraction of missing information (FMI). Import CSV, Excel, SPSS, Stata, R files."
  },
  {
    name: "Robustness Checks",
    category: "Robustness Checks",
    icon: "\u{1F9EA}",
    color: "#F472B6",
    methods: ["Specification Summary","Complexity Ladder","Frequentist vs Bayesian","Correlation Stability"],
    plain: "Four checks on how much a result depends on your choices. None is a new method; each packages published work. Summarise a multiverse of analyses (Specification Summary), compare simple and complex models (Complexity Ladder), see whether a p-value and a Bayes factor agree (Frequentist vs Bayesian), and see which correlations keep their sign when cases are dropped (Correlation Stability). All reuse one sample, so they describe sensitivity; they do not show a result will replicate.",
    technical: "Specification Summary: sign agreement (default 95% threshold), median, IQR, range, 5% trimmed mean, share significant, and which labelled decision levels reverse the dominant sign; descriptive, not inferential (Young & Holsteen, 2017; Simonsohn et al., 2020; Steegen et al., 2016). Complexity Ladder: intercept, linear, polynomial degree 2\u20135; AIC, AICc, BIC, adjusted R\u00b2, 5-fold CV RMSE; lowest-BIC model plus simplest model with \u0394BIC < 2 (Kass & Raftery, 1995). Frequentist vs Bayesian: exact JZS Bayes factor for t-tests (Rouder et al., 2009; r = 0.707), Ly et al. (2016) Bayes factor for correlations; agreement classified as full / partial / disagreement, with prior sensitivity (cf. Wetzels et al., 2011). Correlation Stability: Pearson r, Holm-adjusted p, 2,000-resample bootstrap CI, and a case-dropping stability coefficient adapted from Epskamp et al. (2018); tracks the precision of the sign, not replicability."
  },
  {
    name: "Pipeline Builder",
    category: "Tools & Workflows",
    icon: "\u{1F504}",
    color: "#7C3AED",
    methods: ["14-step reproducible analysis pipeline","Template library (Basic, Regression, ANOVA, Psychometric)"],
    plain: "Chain multiple analysis steps into a reusable, reproducible workflow. Clean, check assumptions, analyze, and compute effect sizes \u2014 all in one automated sequence.",
    technical: "14 pre-configured steps across 5 categories (cleaning, assumptions, analysis, effect, transform). Templates for Basic Analysis, Regression Prep, ANOVA, Psychometric. Export pipeline as JSON."
  },
  {
    name: "Multiverse Analysis",
    category: "Tools & Workflows",
    icon: "\u{1F30C}",
    color: "#E11D48",
    methods: ["Specification curve","Decision-point enumeration","Robustness verdicts"],
    plain: "Runs every reasonable combination of analysis choices and shows all results at once. Reveals whether your conclusion depends on arbitrary decisions.",
    technical: "Specification curve across decision points (outlier methods, missing data, transformations, test variants). Robustness verdict: fragile / moderately robust / highly robust."
  },
  {
    name: "Dual Report",
    category: "Tools & Workflows",
    icon: "\u{1F501}",
    color: "#0D9488",
    methods: ["Parallel frequentist + Bayesian analysis","Convergence assessment"],
    plain: "Runs the same analysis two ways \u2014 traditional (p-values) and Bayesian (evidence strength) \u2014 side by side. Highlights when they agree or disagree.",
    technical: "Parallel analysis: frequentist (test stat, p, CI, effect size) + Bayesian (BF\u2081\u2080; JZS for t-tests, Ly et al. stretched-beta for correlation, BIC approximation for ANOVA/regression; posterior mean and 95% equal-tailed credible interval where a posterior is computed). Convergence assessment. APA combined report."
  },
  {
    name: "Data Simulator",
    category: "Tools & Workflows",
    icon: "\u{1F3B2}",
    color: "#EA580C",
    methods: ["Design-based data generation","Monte Carlo power simulation","Missing data injection (MCAR/MAR/MNAR)"],
    plain: "Creates realistic fake datasets for teaching, testing, or planning. Specify the design and effect size; it generates matching data. Monte Carlo simulation estimates empirical power.",
    technical: "Simulation for between, within, factorial, correlational, regression designs. MCAR/MAR/MNAR missing mechanisms. Monte Carlo empirical power estimation across sample sizes."
  },
  {
    name: "Pre-Registration Builder",
    category: "Tools & Workflows",
    icon: "\u{1F6E1}",
    color: "#4F46E5",
    methods: ["Structured pre-registration","Completeness checklist","Export (OSF, AsPredicted, Markdown)"],
    plain: "Creates a locked-in study plan before data collection \u2014 hypotheses, methods, and analysis plan. Prevents HARKing and increases credibility.",
    technical: "Structured sections: study info, hypotheses (directional/two-tailed), sample planning (power justification), measures, analysis plans, data handling. 11-item completeness checklist. Three export formats."
  },
  {
    name: "Study Planner",
    category: "Tools & Workflows",
    icon: "\u{1F4CB}",
    color: "#6366F1",
    methods: ["Power-based sample size","Power curves","Effect size benchmarks"],
    plain: "Plan your study: choose a design, calculate exactly how many participants you need, visualize your power curve, and check field-specific effect size benchmarks.",
    technical: "Power-based sample size for 8+ designs (t-test, ANOVA, correlation, regression, chi-square, mixed, logistic, equivalence). Power curves, effect size benchmarks (Cohen 1988, Hattie 2009)."
  },
  {
    name: "Curriculum",
    category: "Tools & Workflows",
    icon: "\u{1F393}",
    color: "#8B5CF6",
    methods: ["6 self-paced courses","AI-generated lessons","Research workflow templates"],
    plain: "Six self-paced courses (Psychometrics, Genetics, Bayesian, Social Networks, Geographic Profiling, Clinical Trials) with AI-generated lessons and progress tracking.",
    technical: "AI lesson generation via Claude API. Course progression (locked > started > in-progress > completed). Research workflow templates: Pre/Post Intervention, Factor Analysis, Mediation, Growth Curves."
  },
  {
    name: "AI Statistical Advisor",
    category: "AI & Learning",
    icon: "\u{2728}",
    color: "#A855F7",
    methods: ["Natural language Q&A","Test recommendation engine","Result interpretation","Context-aware follow-ups"],
    plain: "Ask questions in plain English and get expert explanations, test recommendations, and result interpretations. Like a statistics professor available 24/7.",
    technical: "Claude API with behavioral science system prompt. Context-aware: module list, capabilities, last analysis, conversation history. Message persistence. Smart follow-up suggestions."
  },
  {
    name: "Interactive Learning",
    category: "AI & Learning",
    icon: "\u{25B6}",
    color: "#06B6D4",
    methods: ["Central Limit Theorem Simulator","P-Value Distribution Explorer","Confidence Interval Simulator (100 experiments)","Power & Error Tradeoff Visualizer","Correlation Visualizer (Cholesky generation)","Statistical Glossary (62 terms)","Module Help Guides (27 modules, plain + technical)","Guided Analysis Wizard (5-step test selection)","Command Palette (Cmd+K search)"],
    plain: "Interactive simulations and visualizations that build statistical intuition you can\u2019t get from a textbook. Touch, drag, and see statistics come alive.",
    technical: "Real-time statistical simulations using Monte Carlo sampling, Cholesky decomposition for correlated data generation, and empirical demonstration of sampling distributions and coverage probabilities."
  },
  {
    name: "Research Intelligence",
    category: "Research Intelligence",
    icon: "\u{1F6E1}",
    color: "#8B5CF6",
    methods: ["Reviewer 2 checklist (rule-based)","Method Triangulation (3-method convergence)","What Would It Take? (sample size & sensitivity)","Interpretation Confidence (PPV calculator)","Contradiction Detector (cross-analysis)","Assumption Auto-Gate (pre-analysis checks)","Analytic Decision Sensitivity","Equivalence Testing (TOST)","ELI5 plain-language translations","Misconception Detector (20 common errors)","Effect Size Field Benchmarking (15 fields)","Cross-Module Smart Suggestions","Analysis Journal (audit trail)","Result Dependency Graph"],
    plain: "Intelligent tools that automatically check your work, flag issues, and suggest improvements \u2014 like having a methodologist looking over your shoulder.",
    technical: "Methodological checks including sensitivity power analysis with exact noncentral distributions (no observed power), PPV for an assumed effect (Ioannidis 2005), TOST equivalence (Lakens 2017), method triangulation, a 36-specification decision multiverse, and cross-analysis contradiction detection."
  },
  {
    name: "Research Lifecycle",
    category: "Research Tools",
    icon: "\u{1F680}",
    color: "#EC4899",
    methods: ["Research Question Refiner (5-step wizard)","Measurement Selection Guide (50+ constructs, 150+ instruments)","IRB/Ethics Section Generator (AI-powered)","Pre-Registration (living, connected to analyses)","Replication Package Generator (data + code + narrative)","Research Narrative Builder (APA method/results)","Reviewer Response Generator (AI-powered)","Variable Tracker (cross-module usage)","Journal-Specific Formatting (10 journals)","Code Export (R & Python for 17 test types)","Data Story Presentation Mode"],
    plain: "Tools that cover the entire research lifecycle \u2014 from formulating your question to responding to reviewers. Everything a researcher needs without leaving the app.",
    technical: "End-to-end research workflow support including validated instrument database, OSF-compatible pre-registration with deviation tracking, APA 7th narrative generation, and reproducible analysis packages."
  },
  {
    name: "Statistical Glossary",
    category: "Platform",
    icon: "\u{1F4D6}",
    color: "#8B5CF6",
    methods: ["62 terms","6 category filters","Formula rendering","Module cross-references"],
    plain: "A searchable dictionary of 62 statistical terms organized by category, each with a definition, detailed explanation, formula, and which module uses it.",
    technical: "6 category filters, search by term/definition, expandable cards with formula rendering and module cross-references."
  },
  {
    name: "Interactive Tooltips & Cross-References",
    category: "Platform",
    icon: "\u{2139}",
    color: "#6366F1",
    methods: ["StatTooltip (8 built-in terms)","CrossRefText (99-term glossary)"],
    plain: "Tap any statistical value for an instant explanation. Statistical terms are auto-linked to definitions throughout the app.",
    technical: "StatTooltip (8 built-in terms, severity coloring, multi-section modal). CrossRefText (99-term glossary, longest-first regex matching, module navigation)."
  },
  {
    name: "Assumption Dashboard",
    category: "Platform",
    icon: "\u{2705}",
    color: "#10B981",
    methods: ["Pass/warning/fail cards","Diagnostic plots","Alternative test suggestions"],
    plain: "A visual report card for your analysis assumptions: green checkmarks for passes, red warnings for violations, with diagnostic plots and alternative test suggestions.",
    technical: "Assumption cards (pass/warning/fail), Q-Q plot, residuals vs. fitted, Cook\u2019s D bar plot, box plots for homogeneity, and alternative test recommendations."
  },
  {
    name: "Sensitivity Analysis Panel",
    category: "Platform",
    icon: "\u{1F6E1}",
    color: "#F59E0B",
    methods: ["4-method robustness check","Convergence verdict"],
    plain: "Runs the same comparison four different ways (parametric, non-parametric, permutation, bootstrap) to check if they all agree.",
    technical: "Four-method robustness: t-test, Mann-Whitney U, permutation, bootstrap CI. Verdict: Robust (4/4), Mostly Robust (3/4), Sensitive (\u22642/4)."
  },
  {
    name: "Analysis History & Projects",
    category: "Platform",
    icon: "\u{1F552}",
    color: "#06B6D4",
    methods: ["Auto-save","Search & filter","Project CRUD with tags"],
    plain: "Every analysis is saved automatically. Browse, search, and filter past results. Organize work into projects with tags for easy retrieval.",
    technical: "AsyncStorage + Supabase sync. Search by module/test/notes, tag filtering, sort, long-press delete. Project CRUD with module association."
  },
  {
    name: "Export (PDF, CSV, APA)",
    category: "Platform",
    icon: "\u{1F4E5}",
    color: "#3B82F6",
    methods: ["PDF reports","CSV export","APA 7th edition formatting"],
    plain: "Generate print-ready PDF reports, download spreadsheet files, or copy APA-formatted results to paste directly into your manuscript.",
    technical: "PDF via expo-print (native) / browser print (web). CSV with multi-section layout. APA 7th edition formatting via /api/apa/format. Annotation layer for researcher notes."
  },
  {
    name: "Result Knowledge Graph",
    category: "Platform",
    icon: "\u{1F578}",
    color: "#EC4899",
    methods: ["Node-edge visualization","5 relationship types"],
    plain: "Links your analysis results across modules into an interactive network diagram, showing how one analysis feeds into or validates the next.",
    technical: "Node-edge graph with 5 relationship types (uses_data_from, follows_up, validates, contradicts, input_to). Visual canvas with zoom, drag, directional arrows."
  },
  {
    name: "Dark Theme & Accessibility",
    category: "Platform",
    icon: "\u{1F319}",
    color: "#475569",
    methods: ["Dark-first design","Module-specific accent colors","Full ARIA accessibility"],
    plain: "Dark-first design with optional light mode. All modules have unique signature colors. Full accessibility labels throughout.",
    technical: "Slate/indigo palette (bg #0F172A). useTheme hook. Module-specific accents. ARIA roles, labels, and hints on all interactive elements."
  },
  {
    name: "Offline Detection",
    category: "Platform",
    icon: "\u{1F4F4}",
    color: "#EF4444",
    methods: ["Network health monitoring","Animated status banner"],
    plain: "A banner warns you when you lose internet or cannot reach the analysis server, so you know if results might not be available.",
    technical: "useNetwork hook pinging /health every 30s. Red banner (no internet), amber (backend unreachable). Animated slide-in/out."
  },
  {
    name: "UX & Accessibility",
    category: "UX & Accessibility",
    icon: "\u{267F}",
    color: "#F59E0B",
    methods: ["Tap-to-copy any result value","Result caching (30-min LRU, instant repeat analyses)","Auto-save input drafts (persists across navigation)","Undo clear (5-second recovery window)","Contextual progress messages during computation","Colorblind-safe chart palettes (deuteranopia, protanopia, tritanopia)","Adjustable font scaling (0.8x to 1.5x)","High contrast mode","Screen reader result announcements","Hover tooltips on result values (desktop)","Right-click / long-press context menus","Analysis tags and labels","Favorite analyses (pin important results)","Side-by-side result comparison view","8 pre-formatted data templates (2-group, pre-post, factorial, survey, etc.)","Bulk multi-select export (CSV, JSON, APA)","Citation generator (APA 7th, BibTeX, Chicago, Vancouver)","ORCID integration","PowerPoint/slides export","Visual spreadsheet data editor with cell editing, sort, filter"],
    plain: "Every interaction is polished \u2014 copy results with a tap, recover from mistakes, never lose your work. Fully accessible with colorblind palettes, font scaling, high contrast, and screen reader support.",
    technical: "LRU result caching with deterministic input hashing, debounced input draft persistence via AsyncStorage, WCAG 2.1 AA accessibility patterns including prefers-reduced-motion, ARIA roles, and live region announcements."
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
const totalMethods = FEATURES.reduce((sum, f) => sum + f.methods.length, 0);
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
        || f.plain.toLowerCase().includes(q)
        || f.technical.toLowerCase().includes(q)
        || f.methods.some(m => m.toLowerCase().includes(q));
  });
}

// ── Render cards ───────────────────────────────────────────────────
const grid = document.getElementById("featuresGrid");
const countEl = document.getElementById("searchCount");

function renderCards() {
  const filtered = getFiltered();
  const methodTotal = filtered.reduce((s, f) => s + f.methods.length, 0);
  countEl.textContent = `Showing ${filtered.length} module${filtered.length !== 1 ? "s" : ""} with ${methodTotal} methods`;

  grid.innerHTML = "";

  if (filtered.length === 0) {
    grid.innerHTML = '<div class="no-results"><p>No modules match your search.</p><p style="font-size:.85rem">Try a different keyword or category.</p></div>';
    return;
  }

  filtered.forEach((f, i) => {
    const card = document.createElement("div");
    card.className = "feature-card";
    card.style.transitionDelay = Math.min(i * 40, 600) + "ms";

    card.innerHTML = `
      <div class="card-header">
        <div class="card-icon" style="background:${f.color}20; color:${f.color}">${f.icon}</div>
        <div>
          <div class="card-title">${f.name}</div>
          <div class="card-cat">${f.category}</div>
        </div>
      </div>
      <div class="card-method-count">${f.methods.length} method${f.methods.length !== 1 ? "s" : ""}</div>
      <div class="card-plain">${f.plain}</div>
      <button class="card-toggle" type="button">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        Technical details & methods
      </button>
      <div class="card-detail">
        <div class="card-technical"><strong>Technical:</strong> ${f.technical}</div>
        <div class="methods-list">${f.methods.map(m => `<span class="method-tag">${m}</span>`).join("")}</div>
      </div>
    `;
    grid.appendChild(card);
  });

  // Trigger scroll reveal
  requestAnimationFrame(() => observeCards());
}

// ── Toggle detail ──────────────────────────────────────────────────
function toggleDetail(btn) {
  btn.classList.toggle("open");
  const detail = btn.nextElementSibling;
  detail.classList.toggle("open");
}

// ── Scroll reveal ──────────────────────────────────────────────────
function observeCards() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll(".feature-card:not(.visible)").forEach(card => {
    observer.observe(card);
  });
}

// ── Initial render ─────────────────────────────────────────────────
renderCards();


// Card toggles are rendered dynamically; delegate instead of inline onclick.
document.addEventListener('click', function (e) {
  var btn = e.target.closest && e.target.closest('.card-toggle');
  if (btn) toggleDetail(btn);
});
