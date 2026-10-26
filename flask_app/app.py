"""
Maharashtra Skill Align — Labour-Market Intelligence & Curriculum-Alignment Platform
Based on MSINS Problem Statement ID26134
Government of Maharashtra — Dept. of Skills, Employment, Entrepreneurship & Innovation
========================================================================================
Key Capabilities:
1. Load 'sample_jobs.csv' (or live 'scraped_maharashtra_jobs.csv') using Pandas.
2. Preprocess 'Required Skills' with MultiLabelBinarizer.
3. K-Means Clustering (k=4) to cluster skills into industry domain clusters.
4. Train KNN and Logistic Regression to predict 'Curriculum Status'
   (High Demand, Needs Update, Obsolete/Oversupplied) and calculate Accuracy, Precision, Recall, F1.
5. '/' route to render index.html with live dashboard metrics.
6. '/evaluate' POST route that accepts selected skills from the form, runs KNN prediction,
   and returns curriculum status, missing skill gaps, and recommendations.
"""

import io
import os
import re
import json
import random
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify, session,
    send_file, redirect, url_for,
)
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "maharashtra-skill-align-msins-id26134-key"

# ──────────────────────────────────────────────────────────────
# Constants & Maharashtra Geographies
# ──────────────────────────────────────────────────────────────
MAHARASHTRA_DISTRICTS = [
    "Pune", "Mumbai", "Nagpur", "Nashik", "Thane",
    "Chhatrapati Sambhajinagar", "Kolhapur", "Solapur"
]

STATUS_CLASSES = ["High Demand", "Needs Update", "Obsolete/Oversupplied"]

CLUSTER_NAMES = [
    "Cloud, DevOps & Distributed Infrastructure",
    "Data Science, Analytics & Applied AI",
    "Modern Fullstack & Web Application Engineering",
    "Enterprise Systems, Databases & Backend"
]

# Canonical fallback skills list
CANONICAL_SKILLS = [
    "python", "sql", "r", "java", "javascript", "typescript",
    "c++", "scala", "spark", "kafka", "airflow", "dbt",
    "tensorflow", "pytorch", "scikit-learn", "nlp",
    "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git",
    "ci/cd", "microservices", "cybersecurity",
    "tableau", "powerbi", "excel", "data analysis", "statistics",
    "react", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "mongodb", "postgresql", "redis",
]

# ──────────────────────────────────────────────────────────────
# Global In-Memory Model & Analytics Cache
# ──────────────────────────────────────────────────────────────
PIPELINE: Dict[str, Any] = {
    "df": None,
    "mlb": None,
    "kmeans": None,
    "knn": None,
    "lr": None,
    "le": None,
    "scaler": None,
    "feature_names": [],
    "metrics": {},
    "clusters": [],
    "districts": [],
    "hottest_roles": [],
    "status_dist": {},
    "initialized": False,
}

# ──────────────────────────────────────────────────────────────
# Data Preprocessing & Model Training Engine
# ──────────────────────────────────────────────────────────────
def clean_skill_token(token: str) -> str:
    s = str(token).lower().strip()
    s = re.sub(r"[^a-z0-9\+\#\/\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Canonical mappings
    if s in {"k8s"}: return "kubernetes"
    if s in {"postgres"}: return "postgresql"
    if s in {"power bi"}: return "powerbi"
    if s in {"ml"}: return "machine learning"
    if s in {"dl"}: return "deep learning"
    if s in {"nodejs"}: return "node.js"
    if s in {"cicd"}: return "ci/cd"
    if s in {"sklearn"}: return "scikit-learn"
    return s


def parse_skills_string(skills_raw: Any) -> List[str]:
    """Extract clean list of skills from comma/slash/pipe separated text."""
    if pd.isnull(skills_raw):
        return []
    raw = str(skills_raw).lower()
    # Split by comma, semicolon, or newline
    tokens = re.split(r"[,;\n|]+", raw)
    cleaned = []
    for t in tokens:
        st = clean_skill_token(t)
        if len(st) >= 2:
            cleaned.append(st)
    return sorted(list(set(cleaned)))


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers to standard names."""
    col_map = {}
    for c in df.columns:
        norm = re.sub(r"[^a-z0-9]", "", str(c).lower())
        if norm in {"jobrole", "jobtitle", "role", "title", "position", "coursename"}:
            col_map[c] = "Job Role"
        elif norm in {"requiredskills", "skillsraw", "skills", "skill", "keyskills", "technicalskills", "competencies"}:
            col_map[c] = "Required Skills"
        elif norm in {"districtlocation", "district", "location", "city", "region", "worklocation", "place"}:
            col_map[c] = "District/Location"
        elif norm in {"demandvolume", "openings", "vacancies", "demand", "volume", "postings", "count"}:
            col_map[c] = "Demand Volume"
        elif norm in {"curriculumstatus", "status", "demandstatus", "viability", "tier"}:
            col_map[c] = "Curriculum Status"
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    if "Job Role" not in df.columns:
        df["Job Role"] = "Software Developer"
    if "Required Skills" not in df.columns:
        df["Required Skills"] = "python, sql, git"
    if "District/Location" not in df.columns:
        random.seed(42)
        df["District/Location"] = [random.choice(MAHARASHTRA_DISTRICTS) for _ in range(len(df))]
    if "Demand Volume" not in df.columns:
        random.seed(42)
        df["Demand Volume"] = [random.randint(40, 280) for _ in range(len(df))]
    if "Curriculum Status" not in df.columns:
        df["Curriculum Status"] = "Needs Update"

    return df


def load_and_train_models(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    1. Loads dataset using Pandas.
    2. Preprocesses 'Required Skills' with MultiLabelBinarizer.
    3. Runs K-Means (k=4) clustering on skill feature vectors.
    4. Trains KNN and Logistic Regression to predict 'Curriculum Status'.
    5. Computes Accuracy, Precision, Recall, and F1-Score for both models.
    """
    # Check possible dataset paths
    candidate_paths = [
        csv_path,
        "scraped_maharashtra_jobs.csv",
        "sample_jobs.csv",
        os.path.join(os.path.dirname(__file__), "scraped_maharashtra_jobs.csv"),
        os.path.join(os.path.dirname(__file__), "sample_jobs.csv"),
    ]
    target_path = None
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.getsize(p) > 50:
            target_path = p
            break

    if not target_path:
        raise FileNotFoundError("Neither sample_jobs.csv nor scraped_maharashtra_jobs.csv could be found.")

    df = pd.read_csv(target_path)
    df = resolve_columns(df)

    # Preprocess skills column using MultiLabelBinarizer
    df["skills_list"] = df["Required Skills"].apply(parse_skills_string)

    # Fallback for any rows with empty skills
    for i, row in df.iterrows():
        if not row["skills_list"]:
            df.at[i, "skills_list"] = ["python", "sql", "git"]

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["skills_list"])
    feature_names = list(mlb.classes_)

    # Normalize Curriculum Status to standard 3 classes
    def normalize_status(val):
        s = str(val).strip().lower()
        if "high" in s: return "High Demand"
        if "update" in s or "mid" in s or "moderate" in s: return "Needs Update"
        if "obsolete" in s or "oversupplied" in s or "legacy" in s or "entry" in s: return "Obsolete/Oversupplied"
        return "Needs Update"

    df["Curriculum Status"] = df["Curriculum Status"].apply(normalize_status)

    # Ensure all 3 classes exist so supervised models train reliably
    counts = df["Curriculum Status"].value_counts()
    for cls in STATUS_CLASSES:
        if cls not in counts or counts[cls] < 2:
            # Overwrite cyclically
            sample_idxs = df.index.tolist()
            if sample_idxs:
                idx = sample_idxs[STATUS_CLASSES.index(cls) % len(sample_idxs)]
                df.at[idx, "Curriculum Status"] = cls

    y = df["Curriculum Status"]

    # ── K-Means Clustering (k=4) ────────────────────────────────
    n_unique_X = len(np.unique(X, axis=0))
    k_clusters = max(2, min(4, n_unique_X, len(df)))
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)
    df["cluster"] = cluster_ids

    cluster_summaries = []
    centroids = kmeans.cluster_centers_

    for cid in range(k_clusters):
        sub = df[df["cluster"] == cid]
        domain_title = CLUSTER_NAMES[cid % len(CLUSTER_NAMES)]

        # Find top skills in this cluster
        if len(sub) > 0:
            cluster_X = mlb.transform(sub["skills_list"])
            skill_sums = cluster_X.sum(axis=0)
            top_skill_indices = np.argsort(skill_sums)[::-1][:5]
            top_skills = [feature_names[i] for i in top_skill_indices if skill_sums[i] > 0]
            status_dist = sub["Curriculum Status"].value_counts().to_dict()
            total_openings = int(pd.to_numeric(sub["Demand Volume"], errors="coerce").fillna(50).sum())
            top_districts = sub["District/Location"].value_counts().head(3).to_dict()
        else:
            top_skills = []
            status_dist = {}
            total_openings = 0
            top_districts = {}

        cluster_summaries.append({
            "cluster_id": cid,
            "cluster_name": f"Cluster {cid + 1}: {domain_title}",
            "domain_label": domain_title,
            "size": len(sub),
            "total_openings": total_openings,
            "top_skills": top_skills,
            "status_dist": status_dist,
            "top_districts": top_districts,
            "centroid": centroids[cid].tolist(),
        })

    # ── Supervised Learning: KNN & Logistic Regression ──────────
    class_counts = y.value_counts()
    n_classes = len(class_counts)
    test_count = max(1, int(np.ceil(len(df) * 0.25)))
    train_count = len(df) - test_count
    can_stratify = (
        (class_counts.min() >= 2)
        and (n_classes > 1)
        and (test_count >= n_classes)
        and (train_count >= n_classes)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if can_stratify else None
    )

    # Train KNN
    k_neighbors = min(5, max(1, len(X_train) - 1))
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Metrics
    metrics = {
        "knn_accuracy":  round(accuracy_score(y_test, knn_pred) * 100, 1),
        "knn_precision": round(precision_score(y_test, knn_pred, average="weighted", zero_division=0) * 100, 1),
        "knn_recall":    round(recall_score(y_test, knn_pred, average="weighted", zero_division=0) * 100, 1),
        "knn_f1":        round(f1_score(y_test, knn_pred, average="weighted", zero_division=0) * 100, 1),

        "lr_accuracy":   round(accuracy_score(y_test, lr_pred) * 100, 1),
        "lr_precision":  round(precision_score(y_test, lr_pred, average="weighted", zero_division=0) * 100, 1),
        "lr_recall":     round(recall_score(y_test, lr_pred, average="weighted", zero_division=0) * 100, 1),
        "lr_f1":         round(f1_score(y_test, lr_pred, average="weighted", zero_division=0) * 100, 1),
    }

    # District Summary for Maharashtra
    district_list = []
    for dist, grp in df.groupby("District/Location"):
        total_openings = int(pd.to_numeric(grp["Demand Volume"], errors="coerce").fillna(50).sum())
        role_counts = grp["Job Role"].value_counts()
        top_roles = role_counts.head(2).index.tolist()
        grp_skills = mlb.transform(grp["skills_list"]).sum(axis=0)
        top_skill_idx = np.argsort(grp_skills)[::-1][:3]
        top_dist_skills = [feature_names[i] for i in top_skill_idx if grp_skills[i] > 0]
        high_share = round((grp["Curriculum Status"].value_counts().get("High Demand", 0) / len(grp)) * 100, 1)
        district_list.append({
            "district": dist,
            "postings_count": len(grp),
            "total_openings": total_openings,
            "top_roles": top_roles,
            "top_skills": top_dist_skills,
            "high_demand_share": high_share,
        })
    district_list.sort(key=lambda x: x["total_openings"], reverse=True)

    # Top Emerging Roles
    hottest_roles = []
    grouped_roles = df.groupby("Job Role")
    ranked_roles = grouped_roles["Demand Volume"].count().sort_values(ascending=False).head(5)
    for rank, (r_name, p_count) in enumerate(ranked_roles.items(), 1):
        sub_r = df[df["Job Role"] == r_name]
        openings = int(pd.to_numeric(sub_r["Demand Volume"], errors="coerce").fillna(50).sum())
        top_dist = sub_r["District/Location"].mode().iloc[0] if not sub_r["District/Location"].empty else "Pune"
        st = sub_r["Curriculum Status"].mode().iloc[0] if not sub_r["Curriculum Status"].empty else "High Demand"
        sub_skills = mlb.transform(sub_r["skills_list"]).sum(axis=0)
        top_s_idx = np.argsort(sub_skills)[::-1][:4]
        r_skills = [feature_names[i] for i in top_s_idx if sub_skills[i] > 0]
        hottest_roles.append({
            "rank": rank,
            "role": str(r_name).title(),
            "openings": openings,
            "postings": p_count,
            "primary_district": top_dist,
            "status": st,
            "top_skills": r_skills,
            "viability_index": round(min(100.0, (openings / 300) * 70.0 + (30.0 if st == "High Demand" else 15.0)), 1),
        })

    # Update Global Pipeline Cache
    PIPELINE.update({
        "df": df,
        "mlb": mlb,
        "kmeans": kmeans,
        "knn": knn,
        "lr": lr,
        "feature_names": feature_names,
        "metrics": metrics,
        "clusters": cluster_summaries,
        "districts": district_list,
        "hottest_roles": hottest_roles,
        "status_dist": df["Curriculum Status"].value_counts().to_dict(),
        "initialized": True,
    })

    return PIPELINE


# ──────────────────────────────────────────────────────────────
# Recommendations & Gap Detection Logic
# ──────────────────────────────────────────────────────────────
def compute_curriculum_gap(selected_skills: List[str], predicted_status: str) -> Dict[str, Any]:
    """Finds nearest K-Means cluster and determines missing competencies & action plans."""
    mlb = PIPELINE["mlb"]
    clusters = PIPELINE["clusters"]
    feature_names = PIPELINE["feature_names"]

    selected_set = set([s.lower().strip() for s in selected_skills])
    user_vec = mlb.transform([selected_skills])[0]

    # Find closest cluster via cosine similarity
    best_cluster = None
    best_sim = -1.0
    for c in clusters:
        centroid = np.array(c.get("centroid", []))
        if len(centroid) == len(user_vec) and np.linalg.norm(user_vec) > 0 and np.linalg.norm(centroid) > 0:
            sim = float(np.dot(user_vec, centroid) / (np.linalg.norm(user_vec) * np.linalg.norm(centroid)))
        else:
            sim = float(len(selected_set & set(c.get("top_skills", []))))
        if sim > best_sim:
            best_sim = sim
            best_cluster = c

    if not best_cluster and clusters:
        best_cluster = clusters[0]

    cluster_top_skills = best_cluster["top_skills"] if best_cluster else []
    matched_skills = [s for s in selected_skills if s in cluster_top_skills or s in feature_names]
    missing_skills = [s for s in cluster_top_skills if s not in selected_set]

    if predicted_status == "High Demand":
        status_color = "emerald"
        status_message = "Curriculum is strongly aligned with emerging industry demand in Maharashtra."
        readiness = 94
        action_plan = [
            f"Add advanced capstone modules in {', '.join(missing_skills[:2]) or 'System Architecture'}." if missing_skills else "Maintain current industry-validated curriculum.",
            "Establish industry micro-credentials and fast-track placement interviews with tech parks in Pune & Mumbai.",
            "Host technical hackathons sponsored by Maharashtra State Innovation Society partners."
        ]
        equipment_guidance = "Maintain cloud sandbox environments (AWS/Azure/GCP access) and modern Linux development workstations."
        trainer_guidance = "Certify faculty in containerization, cloud infrastructure, and real-world system architecture."
    elif predicted_status == "Needs Update":
        status_color = "amber"
        status_message = "Curriculum covers foundational skills but lacks industry-critical modern tooling."
        readiness = 68
        rec_add = missing_skills[:3] or ["docker", "aws", "ci/cd"]
        action_plan = [
            f"Integrate modern industry tools into course syllabus: add {', '.join(rec_add)}.",
            "Convert 35% of traditional classroom lectures into hands-on lab projects with Git version control.",
            "Align student capstone projects with active hiring requirements in Maharashtra district hubs."
        ]
        equipment_guidance = "Provision cloud container labs (Docker/Kubernetes) and updated Linux workstations."
        trainer_guidance = f"Conduct faculty upskilling workshops on {', '.join(rec_add)} via industry trainer partnerships."
    else:  # Obsolete/Oversupplied
        status_color = "rose"
        status_message = "Curriculum relies on legacy, oversaturated, or low-demand technologies with poor hiring velocity."
        readiness = 32
        pivotal = ["python", "sql", "git", "docker", "aws"]
        action_plan = [
            f"Phase out legacy modules and replace with contemporary technology stacks: {', '.join(pivotal[:3])}.",
            "Re-structure qualification framework to comply with NSQF & MSINS emerging technology standards.",
            "Conduct urgent curriculum revision workshop with Maharashtra State Innovation Society industry board."
        ]
        equipment_guidance = "Retire legacy desktop software; install modern Linux operating environments and high-speed network access."
        trainer_guidance = "Complete comprehensive faculty re-skilling program on Python, Cloud & Modern Web stacks."

    target_districts = list(best_cluster.get("top_districts", {}).keys()) if best_cluster else ["Pune", "Mumbai", "Nagpur"]

    return {
        "aligned_cluster": best_cluster["cluster_name"] if best_cluster else "General Industry Stack",
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "placement_readiness": readiness,
        "status_color": status_color,
        "status_message": status_message,
        "action_plan": action_plan,
        "equipment_guidance": equipment_guidance,
        "trainer_guidance": trainer_guidance,
        "target_districts": target_districts,
    }


# Initialize models at module load
try:
    load_and_train_models()
except Exception as e:
    print(f"Initial model loading deferred: {e}")


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Renders main dashboard with live metrics, clusters, and curriculum evaluator."""
    if not PIPELINE.get("initialized"):
        load_and_train_models()

    m = PIPELINE["metrics"]
    all_skills = sorted(list(set(PIPELINE["feature_names"] + CANONICAL_SKILLS)))

    return render_template(
        "index.html",
        skills=all_skills,
        selected_skills=[],
        course_name="",
        # Model metrics
        knn_accuracy=m.get("knn_accuracy", 100.0),
        knn_precision=m.get("knn_precision", 100.0),
        knn_recall=m.get("knn_recall", 100.0),
        knn_f1=m.get("knn_f1", 100.0),
        lr_accuracy=m.get("lr_accuracy", 100.0),
        lr_precision=m.get("lr_precision", 100.0),
        lr_recall=m.get("lr_recall", 100.0),
        lr_f1=m.get("lr_f1", 100.0),
        # Platform data
        clusters=PIPELINE.get("clusters", []),
        districts=PIPELINE.get("districts", []),
        hottest_roles=PIPELINE.get("hottest_roles", []),
        status_dist=PIPELINE.get("status_dist", {}),
        total_postings=len(PIPELINE["df"]) if PIPELINE["df"] is not None else 400,
        total_skills_count=len(PIPELINE["feature_names"]),
        # Evaluator results (None on initial load)
        predicted_status=None,
        knn_confidence=None,
        lr_confidence=None,
        matched_skills=[],
        missing_skills=[],
        aligned_cluster=None,
        placement_readiness=None,
        action_plan=[],
        equipment_guidance="",
        trainer_guidance="",
        target_districts=[],
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Accepts selected curriculum skills from frontend form.
    Runs KNN to predict curriculum status and passes results & metrics back to template.
    """
    if not PIPELINE.get("initialized"):
        load_and_train_models()

    # Support form submission and JSON fetch
    if request.is_json:
        data = request.get_json(silent=True) or {}
        selected_skills = [s.strip().lower() for s in data.get("skills", []) if s.strip()]
        course_name = data.get("course_name", "Evaluated Skill Curriculum")
    else:
        selected_skills = [s.strip().lower() for s in request.form.getlist("skills") if s.strip()]
        course_name = request.form.get("course_name", "").strip() or "Evaluated Skill Curriculum"

    m = PIPELINE["metrics"]
    all_skills = sorted(list(set(PIPELINE["feature_names"] + CANONICAL_SKILLS)))

    if not selected_skills:
        # No skills checked
        return render_template(
            "index.html",
            skills=all_skills,
            selected_skills=[],
            course_name=course_name,
            error_message="Please select at least one curriculum skill to evaluate course viability.",
            knn_accuracy=m.get("knn_accuracy", 100.0),
            knn_precision=m.get("knn_precision", 100.0),
            knn_recall=m.get("knn_recall", 100.0),
            knn_f1=m.get("knn_f1", 100.0),
            lr_accuracy=m.get("lr_accuracy", 100.0),
            lr_precision=m.get("lr_precision", 100.0),
            lr_recall=m.get("lr_recall", 100.0),
            lr_f1=m.get("lr_f1", 100.0),
            clusters=PIPELINE.get("clusters", []),
            districts=PIPELINE.get("districts", []),
            hottest_roles=PIPELINE.get("hottest_roles", []),
            status_dist=PIPELINE.get("status_dist", {}),
            total_postings=len(PIPELINE["df"]) if PIPELINE["df"] is not None else 400,
            total_skills_count=len(PIPELINE["feature_names"]),
            predicted_status=None,
        )

    # Encode selected skills using MultiLabelBinarizer
    mlb = PIPELINE["mlb"]
    knn = PIPELINE["knn"]
    lr = PIPELINE["lr"]

    user_vec = mlb.transform([selected_skills])

    # Predict with KNN and Logistic Regression
    knn_pred = knn.predict(user_vec)[0]
    knn_proba = knn.predict_proba(user_vec)[0]
    lr_pred = lr.predict(user_vec)[0]
    lr_proba = lr.predict_proba(user_vec)[0]

    # Primary predicted status from KNN
    predicted_status = str(knn_pred)
    knn_conf = round(float(np.max(knn_proba)) * 100, 1)
    lr_conf = round(float(np.max(lr_proba)) * 100, 1)

    # Compute skill gaps and recommendations
    gap_info = compute_curriculum_gap(selected_skills, predicted_status)

    # If request wants JSON (e.g. from fetch)
    if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({
            "predicted_status": predicted_status,
            "course_name": course_name,
            "knn_confidence": knn_conf,
            "lr_status": str(lr_pred),
            "lr_confidence": lr_conf,
            "matched_skills": gap_info["matched_skills"],
            "missing_skills": gap_info["missing_skills"],
            "aligned_cluster": gap_info["aligned_cluster"],
            "placement_readiness": gap_info["placement_readiness"],
            "action_plan": gap_info["action_plan"],
            "equipment_guidance": gap_info["equipment_guidance"],
            "trainer_guidance": gap_info["trainer_guidance"],
            "target_districts": gap_info["target_districts"],
            "metrics": m,
        })

    # Render template with live ML variables
    return render_template(
        "index.html",
        skills=all_skills,
        selected_skills=selected_skills,
        course_name=course_name,
        # Live ML predictions & gaps
        predicted_status=predicted_status,
        knn_confidence=knn_conf,
        lr_status=str(lr_pred),
        lr_confidence=lr_conf,
        matched_skills=gap_info["matched_skills"],
        missing_skills=gap_info["missing_skills"],
        aligned_cluster=gap_info["aligned_cluster"],
        placement_readiness=gap_info["placement_readiness"],
        status_color=gap_info["status_color"],
        status_message=gap_info["status_message"],
        action_plan=gap_info["action_plan"],
        equipment_guidance=gap_info["equipment_guidance"],
        trainer_guidance=gap_info["trainer_guidance"],
        target_districts=gap_info["target_districts"],
        # Model metrics
        knn_accuracy=m.get("knn_accuracy", 100.0),
        knn_precision=m.get("knn_precision", 100.0),
        knn_recall=m.get("knn_recall", 100.0),
        knn_f1=m.get("knn_f1", 100.0),
        lr_accuracy=m.get("lr_accuracy", 100.0),
        lr_precision=m.get("lr_precision", 100.0),
        lr_recall=m.get("lr_recall", 100.0),
        lr_f1=m.get("lr_f1", 100.0),
        # Platform data
        clusters=PIPELINE.get("clusters", []),
        districts=PIPELINE.get("districts", []),
        hottest_roles=PIPELINE.get("hottest_roles", []),
        status_dist=PIPELINE.get("status_dist", {}),
        total_postings=len(PIPELINE["df"]) if PIPELINE["df"] is not None else 400,
        total_skills_count=len(PIPELINE["feature_names"]),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Alias for /evaluate supporting JSON API requests."""
    return evaluate()


@app.route("/upload", methods=["POST"])
def upload():
    """Accepts uploaded CSV and retrains the entire pipeline."""
    if "csv_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["csv_file"]
    if file.filename == "" or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a valid .csv file."}), 400

    try:
        temp_path = os.path.join(os.path.dirname(__file__), "uploaded_temp.csv")
        file.save(temp_path)
        load_and_train_models(temp_path)
        m = PIPELINE["metrics"]
        return jsonify({
            "success": True,
            "n_rows": len(PIPELINE["df"]),
            "n_features": len(PIPELINE["feature_names"]),
            "metrics": m,
            "clusters": PIPELINE["clusters"],
            "districts": PIPELINE["districts"],
            "hottest_roles": PIPELINE["hottest_roles"],
            "status_dist": PIPELINE["status_dist"],
        })
    except Exception as e:
        return jsonify({"error": f"Upload processing error: {str(e)}"}), 400


@app.route("/load-sample", methods=["GET", "POST"])
def load_sample():
    """Quick loads sample_jobs.csv."""
    try:
        load_and_train_models("sample_jobs.csv")
        m = PIPELINE["metrics"]
        return jsonify({
            "success": True,
            "n_rows": len(PIPELINE["df"]),
            "n_features": len(PIPELINE["feature_names"]),
            "metrics": m,
            "clusters": PIPELINE["clusters"],
            "districts": PIPELINE["districts"],
            "hottest_roles": PIPELINE["hottest_roles"],
            "status_dist": PIPELINE["status_dist"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/sample-csv", methods=["GET"])
def sample_csv():
    """Serves sample_jobs.csv as downloadable attachment."""
    path = os.path.join(os.path.dirname(__file__), "sample_jobs.csv")
    if not os.path.exists(path):
        load_and_train_models()
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="maharashtra_job_market_sample.csv")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
