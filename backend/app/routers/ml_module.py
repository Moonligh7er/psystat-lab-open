"""Machine Learning module — classification, clustering, dimensionality
reduction, model comparison, regression prediction, feature selection,
and cross-validation for behavioral science research."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
from scipy import stats as sp_stats

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    learning_curve,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.inspection import permutation_importance

router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────

class ClassificationRequest(BaseModel):
    features: list[list[float]]
    labels: list
    algorithm: str = "logistic_regression"  # logistic_regression/random_forest/svm/knn/decision_tree
    test_size: float = 0.2
    hyperparameters: Optional[dict] = None
    feature_names: Optional[list[str]] = None


class ClusteringRequest(BaseModel):
    features: list[list[float]]
    algorithm: str = "kmeans"  # kmeans/hierarchical/dbscan
    n_clusters: Optional[int] = 3
    eps: Optional[float] = 0.5           # DBSCAN
    min_samples: Optional[int] = 5       # DBSCAN
    linkage: Optional[str] = "ward"      # hierarchical
    labels: Optional[list] = None        # optional ground-truth for viz
    feature_names: Optional[list[str]] = None


class DimensionalityReductionRequest(BaseModel):
    features: list[list[float]]
    method: str = "pca"  # pca/tsne/umap_approx
    n_components: int = 2
    perplexity: float = 30.0  # t-SNE
    labels: Optional[list] = None
    feature_names: Optional[list[str]] = None


class ModelComparisonRequest(BaseModel):
    features: list[list[float]]
    labels: list
    algorithms: list[str]
    feature_names: Optional[list[str]] = None


class PredictionRequest(BaseModel):
    features: list[list[float]]
    target: list[float]
    algorithm: str = "linear"  # linear/ridge/lasso/elastic_net/random_forest/svr
    test_size: float = 0.2
    hyperparameters: Optional[dict] = None
    feature_names: Optional[list[str]] = None


class FeatureSelectionRequest(BaseModel):
    features: list[list[float]]
    target: list
    task: str = "classification"  # classification/regression
    top_k: Optional[int] = None
    feature_names: Optional[list[str]] = None


class CrossValidationRequest(BaseModel):
    features: list[list[float]]
    target: list
    algorithm: str = "logistic_regression"
    cv_strategy: str = "kfold"  # kfold/stratified/loocv/repeated
    n_splits: int = 5
    n_repeats: int = 10
    task: str = "classification"  # classification/regression
    feature_names: Optional[list[str]] = None


# ── Helpers ───────────────────────────────────────────────────────────

def _build_classifier(algorithm: str, hyperparameters: Optional[dict] = None):
    """Instantiate a sklearn classifier by name."""
    hp = hyperparameters or {}
    constructors = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=hp.get("max_iter", 1000),
            C=hp.get("C", 1.0),
            solver=hp.get("solver", "lbfgs"),
            random_state=42,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            min_samples_split=hp.get("min_samples_split", 2),
            random_state=42,
        ),
        "svm": lambda: SVC(
            C=hp.get("C", 1.0),
            kernel=hp.get("kernel", "rbf"),
            probability=True,
            random_state=42,
        ),
        "knn": lambda: KNeighborsClassifier(
            n_neighbors=hp.get("n_neighbors", 5),
            weights=hp.get("weights", "uniform"),
        ),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=hp.get("max_depth", None),
            min_samples_split=hp.get("min_samples_split", 2),
            random_state=42,
        ),
    }
    if algorithm not in constructors:
        raise HTTPException(
            400,
            f"Unknown classification algorithm '{algorithm}'. "
            f"Choose from: {list(constructors.keys())}",
        )
    return constructors[algorithm]()


def _build_regressor(algorithm: str, hyperparameters: Optional[dict] = None):
    """Instantiate a sklearn regressor by name."""
    hp = hyperparameters or {}
    constructors = {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(alpha=hp.get("alpha", 1.0)),
        "lasso": lambda: Lasso(alpha=hp.get("alpha", 1.0), max_iter=hp.get("max_iter", 5000)),
        "elastic_net": lambda: ElasticNet(
            alpha=hp.get("alpha", 1.0),
            l1_ratio=hp.get("l1_ratio", 0.5),
            max_iter=hp.get("max_iter", 5000),
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            random_state=42,
        ),
        "svr": lambda: SVR(
            C=hp.get("C", 1.0),
            kernel=hp.get("kernel", "rbf"),
        ),
    }
    if algorithm not in constructors:
        raise HTTPException(
            400,
            f"Unknown regression algorithm '{algorithm}'. "
            f"Choose from: {list(constructors.keys())}",
        )
    return constructors[algorithm]()


def _safe_float(v):
    """Convert numpy scalar to Python float, handling NaN/Inf."""
    v = float(v)
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _feature_importances(model, algorithm: str, feature_names: list[str]):
    """Extract feature importances where available."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        ranked = sorted(
            zip(feature_names, imp.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return {"method": "gini_importance", "importances": ranked}
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        ranked = sorted(
            zip(feature_names, coef.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return {"method": "absolute_coefficient", "importances": ranked}
    return None


# ── 1. Classification ────────────────────────────────────────────────

@router.post("/classification")
def classification(req: ClassificationRequest):
    """Train and evaluate a classifier on behavioral science data.

    Supports logistic regression, random forest, SVM, KNN, and decision tree
    with stratified train/test splitting and 5-fold cross-validation.
    """
    try:
        X = np.array(req.features, dtype=float)
        y = np.array(req.labels)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid feature/label data: {exc}")

    if X.shape[0] != y.shape[0]:
        raise HTTPException(400, "Number of samples in features and labels must match.")
    if X.shape[0] < 10:
        raise HTTPException(400, "At least 10 samples are required for reliable classification.")

    n_samples, n_features = X.shape
    feature_names = req.feature_names or [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise HTTPException(400, "Length of feature_names must match number of feature columns.")

    classes = np.unique(y)
    n_classes = len(classes)
    is_binary = n_classes == 2

    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=42, stratify=y,
        )
    except ValueError as exc:
        raise HTTPException(400, f"Train/test split failed — likely too few samples per class: {exc}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Build and fit model
    model = _build_classifier(req.algorithm, req.hyperparameters)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_s)

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ROC AUC
    roc_auc = None
    roc_curve_data = None
    if y_prob is not None:
        try:
            if is_binary:
                roc_auc = _safe_float(roc_auc_score(y_test, y_prob[:, 1]))
                fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=classes[1])
                roc_curve_data = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }
            else:
                y_bin = label_binarize(y_test, classes=classes)
                roc_auc = _safe_float(
                    roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
                )
                roc_curve_data = {}
                for i, cls in enumerate(classes):
                    fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_curve_data[str(cls)] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                    }
        except Exception:
            pass

    # Feature importances
    importances = _feature_importances(model, req.algorithm, feature_names)

    # Cross-validation (5-fold stratified)
    cv_scores = cross_val_score(
        _build_classifier(req.algorithm, req.hyperparameters),
        scaler.fit_transform(X), y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    ).tolist()

    # Interpretation
    interp_parts = []
    interp_parts.append(
        f"The {req.algorithm.replace('_', ' ')} classifier achieved {acc:.1%} accuracy "
        f"on the held-out test set ({int((1 - req.test_size) * 100)}/{int(req.test_size * 100)} "
        f"train/test split, n = {n_samples})."
    )
    if f1_macro < 0.5:
        interp_parts.append(
            "The macro-averaged F1 score is below 0.50, indicating the model struggles to "
            "reliably distinguish between groups. Consider whether your measured variables "
            "capture the behavioral construct of interest, or whether additional predictors "
            "are needed."
        )
    elif f1_macro < 0.7:
        interp_parts.append(
            "The macro-averaged F1 is moderate. The model detects some group differences "
            "but may not be strong enough for individual-level prediction in clinical or "
            "applied settings."
        )
    else:
        interp_parts.append(
            "The macro-averaged F1 indicates good discriminative ability. Verify this "
            "holds with cross-validation and consider whether the effect is theoretically "
            "meaningful for your behavioral construct."
        )
    cv_mean = float(np.mean(cv_scores))
    cv_sd = float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0
    interp_parts.append(
        f"Five-fold cross-validation accuracy: M = {cv_mean:.3f}, SD = {cv_sd:.3f}. "
        f"A large gap between test accuracy and CV accuracy may signal overfitting."
    )

    return {
        "accuracy": _safe_float(acc),
        "precision_macro": _safe_float(prec_macro),
        "recall_macro": _safe_float(rec_macro),
        "f1_macro": _safe_float(f1_macro),
        "precision_weighted": _safe_float(prec_weighted),
        "recall_weighted": _safe_float(rec_weighted),
        "f1_weighted": _safe_float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
        "roc_curve_data": roc_curve_data,
        "feature_importances": importances,
        "cross_validation_scores": cv_scores,
        "cross_validation_mean": _safe_float(cv_mean),
        "cross_validation_sd": _safe_float(cv_sd),
        "predictions": y_pred.tolist(),
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "classes": classes.tolist(),
        "algorithm": req.algorithm,
        "test_size": req.test_size,
        "interpretation": " ".join(interp_parts),
        "notes": (
            "Classification in behavioral science is useful for diagnostic group assignment "
            "(e.g., clinical vs. non-clinical), identifying response profiles, or predicting "
            "categorical outcomes from survey/test data. Always report cross-validation results "
            "alongside test-set accuracy, and be cautious about overfitting with small samples "
            "common in psychological research."
        ),
    }


# ── 2. Clustering ────────────────────────────────────────────────────

@router.post("/clustering")
def clustering(req: ClusteringRequest):
    """Unsupervised clustering for identifying latent subgroups in behavioral data.

    Supports KMeans, Agglomerative (hierarchical), and DBSCAN.
    """
    try:
        X = np.array(req.features, dtype=float)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid feature data: {exc}")

    if X.shape[0] < 4:
        raise HTTPException(400, "At least 4 samples are required for clustering.")

    n_samples, n_features = X.shape
    feature_names = req.feature_names or [f"feature_{i}" for i in range(n_features)]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    algorithm = req.algorithm.lower()

    if algorithm == "kmeans":
        if req.n_clusters is None or req.n_clusters < 2:
            raise HTTPException(400, "n_clusters must be >= 2 for KMeans.")
        model = KMeans(n_clusters=req.n_clusters, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(X_s)
        centers = scaler.inverse_transform(model.cluster_centers_).tolist()
        inertia = _safe_float(model.inertia_)

        # Elbow data: k = 2..10
        max_k = min(10, n_samples - 1)
        elbow_k = list(range(2, max_k + 1))
        elbow_inertia = []
        for k in elbow_k:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_s)
            elbow_inertia.append(_safe_float(km.inertia_))
        elbow_data = {"k_values": elbow_k, "inertia": elbow_inertia}

    elif algorithm == "hierarchical":
        if req.n_clusters is None or req.n_clusters < 2:
            raise HTTPException(400, "n_clusters must be >= 2 for hierarchical clustering.")
        model = AgglomerativeClustering(
            n_clusters=req.n_clusters, linkage=req.linkage or "ward",
        )
        cluster_labels = model.fit_predict(X_s)
        centers = None
        inertia = None
        elbow_data = None

        # Dendrogram data via scipy
        from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram as scipy_dendro

        Z = scipy_linkage(X_s, method=req.linkage or "ward")
        dendro = scipy_dendro(Z, no_plot=True)
        dendrogram_data = {
            "icoord": [list(map(float, row)) for row in dendro["icoord"]],
            "dcoord": [list(map(float, row)) for row in dendro["dcoord"]],
            "leaves": [int(v) for v in dendro["leaves"]],
            "linkage_matrix": Z.tolist(),
        }

    elif algorithm == "dbscan":
        model = DBSCAN(
            eps=req.eps or 0.5,
            min_samples=req.min_samples or 5,
        )
        cluster_labels = model.fit_predict(X_s)
        centers = None
        inertia = None
        elbow_data = None

    else:
        raise HTTPException(
            400,
            f"Unknown clustering algorithm '{algorithm}'. "
            f"Choose from: kmeans, hierarchical, dbscan.",
        )

    labels_arr = np.array(cluster_labels)
    unique_labels = np.unique(labels_arr)
    n_clusters_found = int(len(unique_labels[unique_labels >= 0]))  # exclude noise (-1)
    cluster_sizes = {
        str(int(lbl)): int(np.sum(labels_arr == lbl)) for lbl in unique_labels
    }

    # Evaluation metrics (need at least 2 clusters, no all-noise)
    sil = None
    ch = None
    db = None
    if n_clusters_found >= 2 and np.sum(labels_arr >= 0) >= 2:
        non_noise = labels_arr >= 0
        if np.sum(non_noise) > n_clusters_found:
            sil = _safe_float(silhouette_score(X_s[non_noise], labels_arr[non_noise]))
            ch = _safe_float(calinski_harabasz_score(X_s[non_noise], labels_arr[non_noise]))
            db = _safe_float(davies_bouldin_score(X_s[non_noise], labels_arr[non_noise]))

    # Within-cluster sum of squares for KMeans
    wcss = None
    if algorithm == "kmeans":
        wcss = {}
        for c in range(req.n_clusters):
            mask = labels_arr == c
            if np.any(mask):
                dists = np.sum((X_s[mask] - model.cluster_centers_[c]) ** 2)
                wcss[str(c)] = _safe_float(dists)

    # Interpretation
    interp_parts = []
    interp_parts.append(
        f"{algorithm.replace('_', ' ').title()} identified {n_clusters_found} cluster(s) "
        f"from {n_samples} observations across {n_features} variables."
    )
    if sil is not None:
        if sil > 0.7:
            interp_parts.append(
                f"The silhouette score ({sil:.3f}) indicates strong cluster separation — "
                f"the identified subgroups are well-differentiated."
            )
        elif sil > 0.5:
            interp_parts.append(
                f"The silhouette score ({sil:.3f}) suggests reasonable cluster structure, "
                f"though some overlap between groups exists."
            )
        elif sil > 0.25:
            interp_parts.append(
                f"The silhouette score ({sil:.3f}) indicates weak cluster structure. "
                f"The subgroups may not be strongly distinct — interpret with caution."
            )
        else:
            interp_parts.append(
                f"The silhouette score ({sil:.3f}) suggests little meaningful cluster "
                f"structure. The data may not contain discrete subgroups."
            )
    if algorithm == "dbscan" and -1 in cluster_sizes:
        noise_n = cluster_sizes["-1"]
        interp_parts.append(
            f"DBSCAN identified {noise_n} noise point(s) that did not belong to any cluster. "
            f"Consider adjusting eps or min_samples if this seems too high."
        )

    result = {
        "cluster_assignments": cluster_labels.tolist(),
        "n_clusters_found": n_clusters_found,
        "cluster_sizes": cluster_sizes,
        "silhouette_score": sil,
        "calinski_harabasz_index": ch,
        "davies_bouldin_index": db,
        "algorithm": algorithm,
        "n_samples": n_samples,
        "n_features": n_features,
        "interpretation": " ".join(interp_parts),
        "notes": (
            "Clustering can reveal latent subgroups in behavioral data — for example, "
            "distinct response patterns in surveys, typologies of coping strategies, or "
            "naturally occurring diagnostic profiles. Always validate clusters with "
            "theory-driven interpretation. The silhouette score ranges from -1 to 1 "
            "(higher = better separation). Davies-Bouldin is lower-is-better."
        ),
    }

    if algorithm == "kmeans":
        result["cluster_centers"] = centers
        result["inertia"] = inertia
        result["within_cluster_ss"] = wcss
        result["elbow_plot_data"] = elbow_data
    elif algorithm == "hierarchical":
        result["dendrogram_data"] = dendrogram_data
    elif algorithm == "dbscan":
        result["eps"] = req.eps
        result["min_samples"] = req.min_samples

    return result


# ── 3. Dimensionality Reduction ──────────────────────────────────────

@router.post("/dimensionality-reduction")
def dimensionality_reduction(req: DimensionalityReductionRequest):
    """Reduce high-dimensional behavioral data for visualization and analysis.

    Supports PCA, t-SNE, and a PCA-based UMAP approximation.
    """
    try:
        X = np.array(req.features, dtype=float)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid feature data: {exc}")

    n_samples, n_features = X.shape
    feature_names = req.feature_names or [f"feature_{i}" for i in range(n_features)]

    if req.n_components > n_features:
        raise HTTPException(
            400,
            f"n_components ({req.n_components}) cannot exceed the number of features ({n_features}).",
        )
    if req.n_components > n_samples:
        raise HTTPException(
            400,
            f"n_components ({req.n_components}) cannot exceed the number of samples ({n_samples}).",
        )

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    method = req.method.lower()

    if method == "pca":
        # Fit full PCA for scree plot, then extract requested components
        full_pca = PCA(n_components=min(n_features, n_samples))
        full_pca.fit(X_s)

        pca = PCA(n_components=req.n_components)
        X_transformed = pca.fit_transform(X_s)

        explained = pca.explained_variance_ratio_.tolist()
        cumulative = np.cumsum(full_pca.explained_variance_ratio_).tolist()
        loadings = pca.components_.tolist()  # shape (n_components, n_features)

        scree_data = {
            "component": list(range(1, len(full_pca.explained_variance_ratio_) + 1)),
            "explained_variance_ratio": full_pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": cumulative,
        }

        # Loadings table: feature x component
        loadings_table = []
        for j, fname in enumerate(feature_names):
            row = {"feature": fname}
            for c in range(req.n_components):
                row[f"PC{c + 1}"] = _safe_float(pca.components_[c, j])
            loadings_table.append(row)

        method_metrics = {
            "explained_variance_ratios": explained,
            "cumulative_variance": cumulative[: req.n_components],
            "total_variance_explained": _safe_float(sum(explained)),
            "scree_plot_data": scree_data,
            "loadings_matrix": loadings,
            "loadings_table": loadings_table,
            "eigenvalues": full_pca.explained_variance_.tolist(),
        }

    elif method == "tsne":
        if n_samples < 5:
            raise HTTPException(400, "t-SNE requires at least 5 samples.")
        perplexity = min(req.perplexity, (n_samples - 1) / 3.0)
        if perplexity < 1:
            perplexity = 1.0
        tsne = TSNE(
            n_components=req.n_components,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
        )
        X_transformed = tsne.fit_transform(X_s)

        method_metrics = {
            "kl_divergence": _safe_float(tsne.kl_divergence_),
            "perplexity_used": perplexity,
            "note": (
                "t-SNE is a non-linear method optimized for visualization. "
                "Distances between distant clusters are NOT meaningful — only "
                "local neighborhood structure is preserved."
            ),
        }

    elif method == "umap_approx":
        # PCA-based approximation when UMAP is not installed
        pca = PCA(n_components=min(50, n_features, n_samples))
        X_pca = pca.fit_transform(X_s)
        tsne = TSNE(
            n_components=req.n_components,
            perplexity=min(req.perplexity, (n_samples - 1) / 3.0),
            random_state=42,
            max_iter=1000,
        )
        X_transformed = tsne.fit_transform(X_pca)

        method_metrics = {
            "pca_variance_retained": _safe_float(sum(pca.explained_variance_ratio_)),
            "kl_divergence": _safe_float(tsne.kl_divergence_),
            "note": (
                "This is a PCA + t-SNE approximation of UMAP. For true UMAP, "
                "install the umap-learn package."
            ),
        }

    else:
        raise HTTPException(
            400,
            f"Unknown method '{method}'. Choose from: pca, tsne, umap_approx.",
        )

    # Build 2D scatter data for visualization
    scatter_data = []
    for i in range(n_samples):
        point = {"x": _safe_float(X_transformed[i, 0])}
        if req.n_components >= 2:
            point["y"] = _safe_float(X_transformed[i, 1])
        if req.n_components >= 3:
            point["z"] = _safe_float(X_transformed[i, 2])
        if req.labels is not None and i < len(req.labels):
            point["label"] = req.labels[i]
        scatter_data.append(point)

    # Interpretation
    interp_parts = []
    if method == "pca":
        total_var = sum(method_metrics["explained_variance_ratios"])
        interp_parts.append(
            f"PCA with {req.n_components} component(s) explains {total_var:.1%} of the "
            f"total variance in the {n_features} original variables."
        )
        if total_var < 0.5:
            interp_parts.append(
                "Less than 50% of variance is captured — the data may require more "
                "components, or the variables may not share a strong common structure."
            )
        interp_parts.append(
            "Examine the loadings matrix to interpret what each component represents "
            "in terms of your behavioral constructs."
        )
    elif method in ("tsne", "umap_approx"):
        interp_parts.append(
            f"{'t-SNE' if method == 'tsne' else 'UMAP approximation'} projected "
            f"{n_samples} observations from {n_features} dimensions to {req.n_components}D. "
            f"Look for visible clusters or gradients in the scatter plot, but remember "
            f"that axis distances are not directly interpretable."
        )

    return {
        "transformed_coordinates": X_transformed.tolist(),
        "scatter_data": scatter_data,
        "method": method,
        "n_components": req.n_components,
        "n_samples": n_samples,
        "n_features": n_features,
        "method_metrics": method_metrics,
        "interpretation": " ".join(interp_parts),
        "notes": (
            "Dimensionality reduction is valuable in behavioral science for visualizing "
            "high-dimensional questionnaire data, identifying latent constructs, and "
            "reducing multicollinearity before regression. PCA is preferred when you need "
            "interpretable components; t-SNE/UMAP are better for exploratory visualization "
            "of complex, nonlinear data structures."
        ),
    }


# ── 4. Model Comparison ─────────────────────────────────────────────

@router.post("/model-comparison")
def model_comparison(req: ModelComparisonRequest):
    """Compare multiple classifiers using stratified 10-fold cross-validation
    with statistical testing (Friedman + Nemenyi post-hoc)."""
    try:
        X = np.array(req.features, dtype=float)
        y = np.array(req.labels)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid data: {exc}")

    if X.shape[0] != y.shape[0]:
        raise HTTPException(400, "Features and labels must have the same number of samples.")
    if len(req.algorithms) < 2:
        raise HTTPException(400, "Provide at least 2 algorithms to compare.")
    if X.shape[0] < 20:
        raise HTTPException(400, "At least 20 samples recommended for model comparison.")

    n_samples, n_features = X.shape
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    classes = np.unique(y)
    is_binary = len(classes) == 2
    n_folds = min(10, min(np.bincount(y.astype(int) if np.issubdtype(y.dtype, np.integer) else
                                       np.unique(y, return_inverse=True)[1])))
    n_folds = max(n_folds, 2)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results_per_model = {}
    all_fold_accuracies = {}  # algorithm -> list of fold accuracies

    for alg in req.algorithms:
        try:
            model = _build_classifier(alg)
        except HTTPException:
            continue

        acc_scores = cross_val_score(model, X_s, y, cv=cv, scoring="accuracy")
        f1_scores = cross_val_score(model, X_s, y, cv=cv, scoring="f1_macro")

        auc_scores = None
        if is_binary:
            try:
                auc_scores = cross_val_score(model, X_s, y, cv=cv, scoring="roc_auc")
            except Exception:
                pass

        all_fold_accuracies[alg] = acc_scores.tolist()

        results_per_model[alg] = {
            "accuracy_mean": _safe_float(np.mean(acc_scores)),
            "accuracy_sd": _safe_float(np.std(acc_scores, ddof=1)),
            "f1_macro_mean": _safe_float(np.mean(f1_scores)),
            "f1_macro_sd": _safe_float(np.std(f1_scores, ddof=1)),
            "accuracy_folds": acc_scores.tolist(),
            "f1_folds": f1_scores.tolist(),
        }
        if auc_scores is not None:
            results_per_model[alg]["auc_mean"] = _safe_float(np.mean(auc_scores))
            results_per_model[alg]["auc_sd"] = _safe_float(np.std(auc_scores, ddof=1))
            results_per_model[alg]["auc_folds"] = auc_scores.tolist()

    if len(results_per_model) < 2:
        raise HTTPException(400, "Could not build at least 2 valid classifiers for comparison.")

    # Ranking by mean accuracy
    ranking = sorted(
        results_per_model.items(), key=lambda x: x[1]["accuracy_mean"], reverse=True
    )
    ranked_list = [{"rank": i + 1, "algorithm": alg, **metrics} for i, (alg, metrics) in enumerate(ranking)]

    # Statistical comparison: Friedman test
    alg_names = list(all_fold_accuracies.keys())
    fold_matrix = np.array([all_fold_accuracies[a] for a in alg_names])  # (n_algorithms, n_folds)

    friedman_stat = None
    friedman_p = None
    pairwise_comparisons = None

    if fold_matrix.shape[0] >= 3 and fold_matrix.shape[1] >= 2:
        try:
            friedman_stat_val, friedman_p_val = sp_stats.friedmanchisquare(
                *[fold_matrix[i, :] for i in range(fold_matrix.shape[0])]
            )
            friedman_stat = _safe_float(friedman_stat_val)
            friedman_p = _safe_float(friedman_p_val)
        except Exception:
            pass

        # Nemenyi post-hoc (rank-based pairwise)
        if friedman_p is not None and friedman_p < 0.05:
            pairwise_comparisons = []
            k = len(alg_names)
            n = fold_matrix.shape[1]  # number of folds
            # Compute average ranks
            ranks = np.zeros_like(fold_matrix)
            for j in range(n):
                fold_vals = fold_matrix[:, j]
                rank_order = sp_stats.rankdata(-fold_vals)  # higher is better -> lower rank
                ranks[:, j] = rank_order
            avg_ranks = ranks.mean(axis=1)

            # Critical difference (Nemenyi)
            # q_alpha for alpha=0.05, from studentized range distribution approx
            # Using conservative approximation: CD = q_alpha * sqrt(k*(k+1)/(6*n))
            from scipy.stats import studentized_range
            try:
                q_crit = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
            except Exception:
                # Fallback: approximate critical values
                q_crit = 2.569 if k <= 3 else 2.728 if k <= 4 else 2.850
            cd = q_crit * np.sqrt(k * (k + 1) / (6 * n))

            for i in range(k):
                for j in range(i + 1, k):
                    diff = abs(avg_ranks[i] - avg_ranks[j])
                    pairwise_comparisons.append({
                        "algorithm_1": alg_names[i],
                        "algorithm_2": alg_names[j],
                        "rank_difference": _safe_float(diff),
                        "critical_difference": _safe_float(cd),
                        "significant": bool(diff > cd),
                    })

            cd_diagram_data = {
                "algorithms": alg_names,
                "average_ranks": avg_ranks.tolist(),
                "critical_difference": _safe_float(cd),
                "significant_pairs": [
                    (p["algorithm_1"], p["algorithm_2"])
                    for p in pairwise_comparisons if p["significant"]
                ],
            }
        else:
            cd_diagram_data = None
    elif fold_matrix.shape[0] == 2 and fold_matrix.shape[1] >= 2:
        # Wilcoxon signed-rank for 2 algorithms
        try:
            stat, p = sp_stats.wilcoxon(fold_matrix[0], fold_matrix[1])
            friedman_stat = _safe_float(stat)
            friedman_p = _safe_float(p)
        except Exception:
            pass
        cd_diagram_data = None
    else:
        cd_diagram_data = None

    # Interpretation
    best = ranked_list[0]
    interp = (
        f"Comparing {len(alg_names)} classifiers with {n_folds}-fold stratified cross-validation "
        f"(n = {n_samples}). "
        f"The best-performing algorithm was {best['algorithm'].replace('_', ' ')} "
        f"with mean accuracy = {best['accuracy_mean']:.3f} (SD = {best['accuracy_sd']:.3f})."
    )
    if friedman_p is not None:
        if friedman_p < 0.05:
            interp += (
                f" The Friedman test was significant (p = {friedman_p:.4f}), indicating "
                f"statistically reliable differences between algorithms."
            )
        else:
            interp += (
                f" The Friedman test was not significant (p = {friedman_p:.4f}), suggesting "
                f"no reliable differences between algorithms. The simpler model may be preferred "
                f"for interpretability."
            )

    return {
        "model_results": results_per_model,
        "ranking": ranked_list,
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "pairwise_comparisons": pairwise_comparisons,
        "cd_diagram_data": cd_diagram_data,
        "n_folds": n_folds,
        "n_samples": n_samples,
        "interpretation": interp,
        "notes": (
            "Model comparison in behavioral science should prioritize interpretability "
            "alongside predictive accuracy. Simpler models (logistic regression, decision "
            "trees) are often preferred when performance differences are not statistically "
            "significant, because they allow researchers to understand which predictors "
            "matter. The Friedman test is a non-parametric alternative to repeated-measures "
            "ANOVA for comparing multiple algorithms across CV folds."
        ),
    }


# ── 5. Prediction (Regression) ──────────────────────────────────────

@router.post("/prediction")
def prediction(req: PredictionRequest):
    """Train and evaluate a regression model for predicting continuous
    behavioral outcomes."""
    try:
        X = np.array(req.features, dtype=float)
        y = np.array(req.target, dtype=float)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid data: {exc}")

    if X.shape[0] != y.shape[0]:
        raise HTTPException(400, "Features and target must have the same number of samples.")
    if X.shape[0] < 10:
        raise HTTPException(400, "At least 10 samples are required for regression.")

    n_samples, n_features = X.shape
    feature_names = req.feature_names or [f"feature_{i}" for i in range(n_features)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=42,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = _build_regressor(req.algorithm, req.hyperparameters)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - n_features - 1) if len(y_test) > n_features + 1 else r2
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = mean_absolute_error(y_test, y_pred)
    # MAPE — guard against zero targets
    nonzero = y_test != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100)
    else:
        mape = None

    # Residual diagnostics
    residuals = y_test - y_pred

    # Normality of residuals (Shapiro-Wilk)
    if len(residuals) >= 3 and len(residuals) <= 5000:
        sw_stat, sw_p = sp_stats.shapiro(residuals)
    else:
        sw_stat, sw_p = None, None

    # Homoscedasticity: Breusch-Pagan-like (correlation of abs residuals with predicted)
    if len(y_pred) > 2:
        bp_r, bp_p = sp_stats.pearsonr(np.abs(residuals), y_pred)
    else:
        bp_r, bp_p = None, None

    residual_diagnostics = {
        "residuals": residuals.tolist(),
        "shapiro_wilk_stat": _safe_float(sw_stat) if sw_stat is not None else None,
        "shapiro_wilk_p": _safe_float(sw_p) if sw_p is not None else None,
        "residuals_normal": bool(sw_p > 0.05) if sw_p is not None else None,
        "abs_resid_pred_correlation": _safe_float(bp_r) if bp_r is not None else None,
        "abs_resid_pred_p": _safe_float(bp_p) if bp_p is not None else None,
        "homoscedastic": bool(bp_p > 0.05) if bp_p is not None else None,
        "mean_residual": _safe_float(np.mean(residuals)),
        "std_residual": _safe_float(np.std(residuals, ddof=1)),
    }

    # Actual vs predicted plot data
    plot_data = {
        "actual": y_test.tolist(),
        "predicted": y_pred.tolist(),
    }

    # Feature importances
    importances = _feature_importances(model, req.algorithm, feature_names)

    # Cross-validation
    cv_model = _build_regressor(req.algorithm, req.hyperparameters)
    X_all_s = scaler.fit_transform(X)
    cv_scores = cross_val_score(
        cv_model, X_all_s, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
    ).tolist()

    # Interpretation
    interp_parts = []
    interp_parts.append(
        f"The {req.algorithm.replace('_', ' ')} model explained {r2:.1%} of variance "
        f"in the test set (R² = {r2:.3f}, adjusted R² = {adj_r2:.3f})."
    )
    if r2 < 0.1:
        interp_parts.append(
            "The very low R² suggests the predictors explain little variance in the outcome. "
            "Consider whether important variables are missing or whether the relationship "
            "is nonlinear."
        )
    elif r2 < 0.3:
        interp_parts.append(
            "In behavioral science, R² values in this range are common and may still be "
            "theoretically meaningful, though prediction at the individual level is limited."
        )
    else:
        interp_parts.append(
            "This is a relatively strong effect for behavioral data. Verify with "
            "cross-validation to ensure the result is not driven by overfitting."
        )

    if residual_diagnostics["residuals_normal"] is False:
        interp_parts.append(
            "Residuals deviate significantly from normality (Shapiro-Wilk p < .05), "
            "which may affect confidence intervals for linear models."
        )
    if residual_diagnostics["homoscedastic"] is False:
        interp_parts.append(
            "Evidence of heteroscedasticity detected — prediction errors vary systematically "
            "with predicted values. Consider robust standard errors or data transformation."
        )

    return {
        "r_squared": _safe_float(r2),
        "adjusted_r_squared": _safe_float(adj_r2),
        "rmse": _safe_float(rmse),
        "mae": _safe_float(mae),
        "mape": _safe_float(mape) if mape is not None else None,
        "residual_diagnostics": residual_diagnostics,
        "actual_vs_predicted": plot_data,
        "feature_importances": importances,
        "cross_validation_r2_scores": cv_scores,
        "cross_validation_r2_mean": _safe_float(np.mean(cv_scores)),
        "cross_validation_r2_sd": _safe_float(np.std(cv_scores, ddof=1)),
        "predictions": y_pred.tolist(),
        "algorithm": req.algorithm,
        "n_samples": n_samples,
        "n_features": n_features,
        "test_size": req.test_size,
        "interpretation": " ".join(interp_parts),
        "notes": (
            "Regression in behavioral science often yields lower R² values than in physical "
            "sciences due to inherent measurement noise and individual variability. An R² of "
            "0.10–0.30 can still represent a meaningful effect. Always check residual "
            "diagnostics — non-normality and heteroscedasticity can bias standard errors in "
            "linear models. Ridge/Lasso are recommended when multicollinearity is suspected "
            "among predictors (e.g., correlated questionnaire subscales)."
        ),
    }


# ── 6. Feature Selection ────────────────────────────────────────────

@router.post("/feature-selection")
def feature_selection(req: FeatureSelectionRequest):
    """Identify the most important predictors using multiple methods —
    particularly useful for reducing large item pools in behavioral research."""
    try:
        X = np.array(req.features, dtype=float)
        y = np.array(req.target)
        if req.task == "regression":
            y = y.astype(float)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid data: {exc}")

    if X.shape[0] != y.shape[0]:
        raise HTTPException(400, "Features and target must have the same number of samples.")
    if X.shape[0] < 10:
        raise HTTPException(400, "At least 10 samples are required for feature selection.")

    n_samples, n_features = X.shape
    feature_names = req.feature_names or [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise HTTPException(400, "Length of feature_names must match number of feature columns.")

    top_k = req.top_k or max(1, n_features // 2)
    top_k = min(top_k, n_features)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    is_classification = req.task == "classification"

    # 1. Correlation-based filter
    correlations = []
    for j in range(n_features):
        if is_classification:
            # Point-biserial or eta-squared approximation for multi-class
            classes = np.unique(y)
            if len(classes) == 2:
                r, p = sp_stats.pointbiserialr(y.astype(float), X[:, j])
                correlations.append({"feature": feature_names[j], "correlation": _safe_float(abs(r)), "p_value": _safe_float(p)})
            else:
                # Eta-squared from one-way ANOVA
                groups = [X[:, j][y == c] for c in classes if np.sum(y == c) > 0]
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    f_stat, p = sp_stats.f_oneway(*groups)
                    ss_between = sum(len(g) * (np.mean(g) - np.mean(X[:, j])) ** 2 for g in groups)
                    ss_total = np.sum((X[:, j] - np.mean(X[:, j])) ** 2)
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0
                    correlations.append({"feature": feature_names[j], "correlation": _safe_float(eta_sq), "p_value": _safe_float(p)})
                else:
                    correlations.append({"feature": feature_names[j], "correlation": 0, "p_value": 1.0})
        else:
            r, p = sp_stats.pearsonr(X[:, j], y)
            correlations.append({"feature": feature_names[j], "correlation": _safe_float(abs(r)), "p_value": _safe_float(p)})
    correlations.sort(key=lambda x: x["correlation"], reverse=True)
    corr_ranking = [c["feature"] for c in correlations]

    # 2. Mutual information
    if is_classification:
        mi = mutual_info_classif(X_s, y, random_state=42)
    else:
        mi = mutual_info_regression(X_s, y, random_state=42)
    mi_ranked = sorted(
        zip(feature_names, mi.tolist()), key=lambda x: x[1], reverse=True
    )
    mi_ranking = [name for name, _ in mi_ranked]

    # 3. Recursive Feature Elimination (RFE)
    if is_classification:
        rfe_estimator = LogisticRegression(max_iter=1000, random_state=42)
    else:
        rfe_estimator = Ridge(alpha=1.0)
    rfe = RFE(rfe_estimator, n_features_to_select=top_k, step=1)
    rfe.fit(X_s, y)
    rfe_ranking_raw = sorted(
        zip(feature_names, rfe.ranking_.tolist()), key=lambda x: x[1]
    )
    rfe_ranking = [name for name, _ in rfe_ranking_raw]
    rfe_selected = [feature_names[i] for i in range(n_features) if rfe.support_[i]]

    # 4. Permutation importance
    if is_classification:
        perm_model = RandomForestClassifier(n_estimators=100, random_state=42)
        scoring = "accuracy"
    else:
        perm_model = RandomForestRegressor(n_estimators=100, random_state=42)
        scoring = "r2"

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42,
        stratify=y if is_classification else None,
    )
    perm_model.fit(X_train, y_train)
    perm_result = permutation_importance(
        perm_model, X_test, y_test, n_repeats=10, random_state=42, scoring=scoring,
    )
    perm_ranked = sorted(
        zip(feature_names, perm_result.importances_mean.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    perm_ranking = [name for name, _ in perm_ranked]

    # Consensus ranking (average rank across methods)
    method_rankings = {
        "correlation": corr_ranking,
        "mutual_information": mi_ranking,
        "rfe": rfe_ranking,
        "permutation_importance": perm_ranking,
    }
    avg_rank = {}
    for fname in feature_names:
        ranks = []
        for method_name, ranking in method_rankings.items():
            if fname in ranking:
                ranks.append(ranking.index(fname) + 1)
        avg_rank[fname] = np.mean(ranks) if ranks else n_features
    consensus = sorted(avg_rank.items(), key=lambda x: x[1])
    consensus_ranking = [name for name, _ in consensus]
    selected_features = consensus_ranking[:top_k]

    # Performance comparison: all features vs selected
    if is_classification:
        eval_model_all = LogisticRegression(max_iter=1000, random_state=42)
        eval_model_sel = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score_metric = "accuracy"
    else:
        eval_model_all = Ridge(alpha=1.0)
        eval_model_sel = Ridge(alpha=1.0)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        score_metric = "r2"

    scores_all = cross_val_score(eval_model_all, X_s, y, cv=cv, scoring=score_metric)
    sel_indices = [feature_names.index(f) for f in selected_features]
    X_sel = X_s[:, sel_indices]
    scores_sel = cross_val_score(eval_model_sel, X_sel, y, cv=cv, scoring=score_metric)

    performance_comparison = {
        "all_features": {
            "n_features": n_features,
            "mean_score": _safe_float(np.mean(scores_all)),
            "sd_score": _safe_float(np.std(scores_all, ddof=1)),
            "metric": score_metric,
        },
        "selected_features": {
            "n_features": top_k,
            "mean_score": _safe_float(np.mean(scores_sel)),
            "sd_score": _safe_float(np.std(scores_sel, ddof=1)),
            "metric": score_metric,
        },
    }

    # Interpretation
    diff = float(np.mean(scores_all) - np.mean(scores_sel))
    interp_parts = [
        f"Feature selection reduced the predictor set from {n_features} to {top_k} features."
    ]
    if abs(diff) < 0.02:
        interp_parts.append(
            f"Performance was nearly identical with the reduced set "
            f"({np.mean(scores_sel):.3f} vs {np.mean(scores_all):.3f}), suggesting "
            f"the removed features contributed little unique predictive information."
        )
    elif diff > 0:
        interp_parts.append(
            f"The selected features performed slightly worse "
            f"({np.mean(scores_sel):.3f} vs {np.mean(scores_all):.3f}). "
            f"Consider retaining more features if the performance drop matters for your application."
        )
    else:
        interp_parts.append(
            f"The selected features actually outperformed the full set "
            f"({np.mean(scores_sel):.3f} vs {np.mean(scores_all):.3f}), suggesting "
            f"the removed features introduced noise."
        )
    interp_parts.append(
        f"The top-ranked features by consensus are: {', '.join(selected_features[:5])}."
    )

    return {
        "rankings": {
            "correlation": correlations,
            "mutual_information": mi_ranked,
            "rfe": rfe_ranking_raw,
            "permutation_importance": perm_ranked,
        },
        "consensus_ranking": consensus,
        "selected_features": selected_features,
        "top_k": top_k,
        "rfe_selected": rfe_selected,
        "performance_comparison": performance_comparison,
        "n_samples": n_samples,
        "n_features": n_features,
        "task": req.task,
        "interpretation": " ".join(interp_parts),
        "notes": (
            "Feature selection is critical in behavioral science where studies often measure "
            "many correlated variables (e.g., questionnaire items, physiological channels). "
            "Multiple methods are used because no single method is universally best: "
            "correlation-based filters capture linear relationships, mutual information "
            "detects nonlinear associations, RFE considers feature interactions within a "
            "model, and permutation importance measures real predictive contribution. "
            "The consensus ranking averages across methods for a robust selection."
        ),
    }


# ── 7. Cross-Validation ─────────────────────────────────────────────

@router.post("/cross-validation")
def cross_validation(req: CrossValidationRequest):
    """Flexible cross-validation with learning curve analysis for behavioral
    science models."""
    try:
        X = np.array(req.features, dtype=float)
        y = np.array(req.target)
        if req.task == "regression":
            y = y.astype(float)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid data: {exc}")

    if X.shape[0] != y.shape[0]:
        raise HTTPException(400, "Features and target must have the same number of samples.")

    n_samples, n_features = X.shape
    is_classification = req.task == "classification"

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Build model
    if is_classification:
        model = _build_classifier(req.algorithm)
        scoring = "accuracy"
    else:
        model = _build_regressor(req.algorithm)
        scoring = "r2"

    # Build CV strategy
    strategy = req.cv_strategy.lower()
    if strategy == "kfold":
        if is_classification:
            cv = KFold(n_splits=req.n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=req.n_splits, shuffle=True, random_state=42)
    elif strategy == "stratified":
        if not is_classification:
            raise HTTPException(400, "Stratified CV is only available for classification tasks.")
        cv = StratifiedKFold(n_splits=req.n_splits, shuffle=True, random_state=42)
    elif strategy == "loocv":
        cv = LeaveOneOut()
    elif strategy == "repeated":
        if is_classification:
            cv = RepeatedStratifiedKFold(
                n_splits=req.n_splits, n_repeats=req.n_repeats, random_state=42,
            )
        else:
            cv = RepeatedKFold(
                n_splits=req.n_splits, n_repeats=req.n_repeats, random_state=42,
            )
    else:
        raise HTTPException(
            400,
            f"Unknown cv_strategy '{strategy}'. Choose from: kfold, stratified, loocv, repeated.",
        )

    # Run cross-validation
    scores = cross_val_score(model, X_s, y, cv=cv, scoring=scoring)

    # Per-fold metrics
    fold_results = []
    for i, score in enumerate(scores):
        fold_results.append({"fold": i + 1, "score": _safe_float(score)})

    # Learning curve (performance vs training size)
    lc_cv = StratifiedKFold(n_splits=min(5, n_samples // 2), shuffle=True, random_state=42) if is_classification else KFold(n_splits=min(5, n_samples // 2), shuffle=True, random_state=42)
    try:
        if is_classification:
            lc_model = _build_classifier(req.algorithm)
        else:
            lc_model = _build_regressor(req.algorithm)

        train_sizes_abs, train_scores, test_scores = learning_curve(
            lc_model, X_s, y,
            train_sizes=np.linspace(0.2, 1.0, 8),
            cv=lc_cv,
            scoring=scoring,
            random_state=42,
            n_jobs=1,
        )
        learning_curve_data = {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_sd": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_sd": np.std(test_scores, axis=1).tolist(),
        }
    except Exception:
        learning_curve_data = None

    # Interpretation
    mean_score = float(np.mean(scores))
    sd_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0

    interp_parts = [
        f"{'Stratified ' if strategy == 'stratified' else ''}"
        f"{'Leave-one-out' if strategy == 'loocv' else ''}"
        f"{'Repeated ' if strategy == 'repeated' else ''}"
        f"{'K-fold ' if strategy == 'kfold' else ''}"
        f"cross-validation ({len(scores)} iterations) yielded a mean "
        f"{scoring} of {mean_score:.3f} (SD = {sd_score:.3f})."
    ]

    if sd_score > 0.1:
        interp_parts.append(
            "The high variability across folds suggests the model's performance is "
            "unstable — possibly due to small sample size or heterogeneous subgroups "
            "in the data."
        )
    elif sd_score < 0.02:
        interp_parts.append(
            "Performance is very stable across folds, indicating consistent model behavior."
        )

    if learning_curve_data is not None:
        train_final = learning_curve_data["train_scores_mean"][-1]
        test_final = learning_curve_data["test_scores_mean"][-1]
        gap = train_final - test_final
        if gap > 0.1:
            interp_parts.append(
                f"The learning curve shows a gap of {gap:.3f} between training and test "
                f"performance, suggesting overfitting. More data or regularization may help."
            )
        elif test_final < train_final and learning_curve_data["test_scores_mean"][-1] > learning_curve_data["test_scores_mean"][0]:
            interp_parts.append(
                "The learning curve shows test performance still improving with more data — "
                "collecting additional participants may meaningfully improve results."
            )

    return {
        "fold_results": fold_results,
        "mean_score": _safe_float(mean_score),
        "sd_score": _safe_float(sd_score),
        "min_score": _safe_float(float(np.min(scores))),
        "max_score": _safe_float(float(np.max(scores))),
        "scoring_metric": scoring,
        "n_folds": len(scores),
        "cv_strategy": strategy,
        "algorithm": req.algorithm,
        "learning_curve_data": learning_curve_data,
        "n_samples": n_samples,
        "n_features": n_features,
        "interpretation": " ".join(interp_parts),
        "notes": (
            "Cross-validation provides a more honest estimate of model performance than "
            "a single train/test split, which is especially important with the small samples "
            "common in behavioral research. Stratified CV preserves class proportions and is "
            "recommended for classification. Leave-one-out CV is useful for very small samples "
            "but can have high variance. Repeated CV (e.g., 10x10) gives the most stable "
            "estimates. The learning curve helps diagnose whether your model would benefit "
            "from more data (underfitting) or needs regularization (overfitting)."
        ),
    }
