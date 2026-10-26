"""
Maharashtra Skill Align — REST API Backend
Based on MSINS Problem Statement ID26134
Government of Maharashtra — Dept. of Skills, Employment, Entrepreneurship & Innovation
========================================================================================
REST API Specifications:
1. Data Integration: Loads 'scraped_maharashtra_jobs.csv' from 'flask_app/' (fallback to sample_jobs.csv).
2. Explicit Column Handling: Maps and processes 'Job Title' and 'Required Skills' columns using MultiLabelBinarizer.
3. Unsupervised K-Means (k=4) clustering to discover industry skill domains.
4. Supervised KNN and Logistic Regression trained on 'Curriculum Status'.
5. REST Endpoints:
   - GET  /api/metrics   → Returns dataset statistics (total rows, unique skills, model accuracy, clusters)
   - POST /api/evaluate  → Accepts selected skills as JSON, runs KNN, returns curriculum status & recommendations
   - GET  /api/skills    → Returns list of all unique skills in the dataset
   - GET  /              → API status and sitemap
"""

import os
import re
import io
import json
import random
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ──────────────────────────────────────────────────────────────
# Flask App Setup
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "maharashtra-skill-align-api-secret-key"

# ──────────────────────────────────────────────────────────────
# Domain Constants
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

# Standard canonical technical skills reference
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
# Global In-Memory Model Cache
# ──────────────────────────────────────────────────────────────
MODEL_CACHE: Dict[str, Any] = {
    "df": None,
    "active_dataset_file": None,
    "mlb": None,
    "kmeans": None,
    "knn": None,
    "lr": None,
    "unique_skills": [],
    "metrics": {},
    "clusters": [],
    "districts": [],
    "hottest_roles": [],
    "status_distribution": {},
    "initialized": False,
}

# ──────────────────────────────────────────────────────────────
# Preprocessing Helpers
# ──────────────────────────────────────────────────────────────
def clean_skill_token(token: str) -> str:
    s = str(token).lower().strip()
    s = re.sub(r"[^a-z0-9\+\#\/\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Normalize common abbreviations
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
    """Tokenize and clean the 'Required Skills' column."""
    if pd.isnull(skills_raw):
        return []
    raw = str(skills_raw).lower()
    tokens = re.split(r"[,;\n|]+", raw)
    cleaned = []
    for t in tokens:
        st = clean_skill_token(t)
        if len(st) >= 2:
            cleaned.append(st)
    return sorted(list(set(cleaned)))


def explicit_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly detects, maps, and normalizes 'Job Title' and 'Required Skills'
    along with 'District/Location', 'Demand Volume', and 'Curriculum Status'.
    """
    rename_dict = {}
    for col in df.columns:
        norm = re.sub(r"[^a-z0-9]", "", str(col).lower())
        # Explicit mapping for Job Title
        if norm in {"jobtitle", "title", "jobrole", "role", "position", "designation"}:
            rename_dict[col] = "Job Title"
        # Explicit mapping for Required Skills
        elif norm in {"requiredskills", "skills", "skillsraw", "skill", "keyskills", "technicalskills", "competencies"}:
            rename_dict[col] = "Required Skills"
        elif norm in {"districtlocation", "district", "location", "city", "region", "worklocation"}:
            rename_dict[col] = "District/Location"
        elif norm in {"demandvolume", "openings", "vacancies", "demand", "volume", "postings", "count"}:
            rename_dict[col] = "Demand Volume"
        elif norm in {"curriculumstatus", "status", "demandstatus", "viability", "tier"}:
            rename_dict[col] = "Curriculum Status"

    df = df.rename(columns=rename_dict)

    # Ensure required columns are present with sensible defaults
    if "Job Title" not in df.columns:
        df["Job Title"] = "Software Engineer"
    if "Required Skills" not in df.columns:
        df["Required Skills"] = "python, sql, git"
    if "District/Location" not in df.columns:
        random.seed(42)
        df["District/Location"] = [random.choice(MAHARASHTRA_DISTRICTS) for _ in range(len(df))]
    if "Demand Volume" not in df.columns:
        random.seed(42)
        df["Demand Volume"] = [random.randint(50, 300) for _ in range(len(df))]
    if "Curriculum Status" not in df.columns:
        df["Curriculum Status"] = "Needs Update"

    return df


def normalize_status_label(val: Any) -> str:
    s = str(val).strip().lower()
    if "high" in s or "advanced" in s or "emerging" in s:
        return "High Demand"
    if "update" in s or "mid" in s or "moderate" in s:
        return "Needs Update"
    if "obsolete" in s or "oversupplied" in s or "legacy" in s:
        return "Obsolete/Oversupplied"
    return "Needs Update"


# ──────────────────────────────────────────────────────────────
# Data Loading & ML Training Pipeline
# ──────────────────────────────────────────────────────────────
def initialize_data_pipeline(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    1. Loads 'flask_app/scraped_maharashtra_jobs.csv' (or provided path, fallback to sample_jobs.csv).
    2. Explicitly maps 'Job Title' and 'Required Skills'.
    3. Fits MultiLabelBinarizer on 'Required Skills'.
    4. Runs K-Means (k=4) clustering.
    5. Trains KNN and Logistic Regression on 'Curriculum Status' and computes evaluation metrics.
    """
    candidate_paths = [
        file_path,
        "scraped_maharashtra_jobs.csv",
        "flask_app/scraped_maharashtra_jobs.csv",
        os.path.join(os.path.dirname(__file__), "scraped_maharashtra_jobs.csv"),
        "sample_jobs.csv",
        "flask_app/sample_jobs.csv",
        os.path.join(os.path.dirname(__file__), "sample_jobs.csv"),
    ]

    chosen_path = None
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.getsize(p) > 50:
            chosen_path = p
            break

    if not chosen_path:
        raise FileNotFoundError("Could not find scraped_maharashtra_jobs.csv or sample_jobs.csv.")

    df = pd.read_csv(chosen_path)
    df = explicit_column_mapping(df)

    # Preprocess Required Skills using MultiLabelBinarizer
    df["skills_list"] = df["Required Skills"].apply(parse_skills_string)

    # Fallback for any rows without detected skills
    for i, row in df.iterrows():
        if not row["skills_list"]:
            df.at[i, "skills_list"] = ["python", "sql", "git"]

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["skills_list"])
    unique_skills = sorted(list(mlb.classes_))

    # Preprocess Curriculum Status
    df["Curriculum Status"] = df["Curriculum Status"].apply(normalize_status_label)

    # Ensure class balance so all 3 classes exist for supervised learning
    status_counts = df["Curriculum Status"].value_counts()
    for cls in STATUS_CLASSES:
        if cls not in status_counts or status_counts[cls] < 2:
            sample_indices = df.index.tolist()
            if sample_indices:
                idx = sample_indices[STATUS_CLASSES.index(cls) % len(sample_indices)]
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
        domain_label = CLUSTER_NAMES[cid % len(CLUSTER_NAMES)]

        if len(sub) > 0:
            cluster_X = mlb.transform(sub["skills_list"])
            skill_sums = cluster_X.sum(axis=0)
            top_skill_indices = np.argsort(skill_sums)[::-1][:6]
            top_skills = [unique_skills[i] for i in top_skill_indices if skill_sums[i] > 0]
            status_dist = sub["Curriculum Status"].value_counts().to_dict()
            total_openings = int(pd.to_numeric(sub["Demand Volume"], errors="coerce").fillna(50).sum())
            top_districts = sub["District/Location"].value_counts().head(3).to_dict()
        else:
            top_skills = []
            status_dist = {}
            total_openings = 0
            top_districts = {}

        cluster_summaries.append({
            "cluster_id": cid + 1,
            "domain_label": domain_label,
            "cluster_name": f"Cluster {cid + 1}: {domain_label}",
            "size": len(sub),
            "total_openings": total_openings,
            "top_skills": top_skills,
            "status_distribution": status_dist,
            "top_districts": top_districts,
            "centroid": centroids[cid].tolist(),
        })

    # ── Supervised Models: KNN & Logistic Regression ───────────
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

    # K-Nearest Neighbors Classifier
    k_neighbors = min(5, max(1, len(X_train) - 1))
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    # Logistic Regression Classifier
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Compute Evaluation Metrics
    metrics = {
        "knn": {
            "model": "K-Nearest Neighbors",
            "k_neighbors": k_neighbors,
            "accuracy": round(float(accuracy_score(y_test, knn_pred)) * 100, 1),
            "precision": round(float(precision_score(y_test, knn_pred, average="weighted", zero_division=0)) * 100, 1),
            "recall": round(float(recall_score(y_test, knn_pred, average="weighted", zero_division=0)) * 100, 1),
            "f1_score": round(float(f1_score(y_test, knn_pred, average="weighted", zero_division=0)) * 100, 1),
        },
        "logistic_regression": {
            "model": "Logistic Regression",
            "accuracy": round(float(accuracy_score(y_test, lr_pred)) * 100, 1),
            "precision": round(float(precision_score(y_test, lr_pred, average="weighted", zero_division=0)) * 100, 1),
            "recall": round(float(recall_score(y_test, lr_pred, average="weighted", zero_division=0)) * 100, 1),
            "f1_score": round(float(f1_score(y_test, lr_pred, average="weighted", zero_division=0)) * 100, 1),
        }
    }

    # District Level Aggregations
    district_list = []
    for dist, grp in df.groupby("District/Location"):
        total_openings = int(pd.to_numeric(grp["Demand Volume"], errors="coerce").fillna(50).sum())
        role_counts = grp["Job Title"].value_counts()
        top_roles = role_counts.head(2).index.tolist()
        grp_skills = mlb.transform(grp["skills_list"]).sum(axis=0)
        top_skill_idx = np.argsort(grp_skills)[::-1][:4]
        top_dist_skills = [unique_skills[i] for i in top_skill_idx if grp_skills[i] > 0]
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

    # Top Emerging Industry Roles
    hottest_roles = []
    grouped_roles = df.groupby("Job Title")
    ranked_roles = grouped_roles["Demand Volume"].count().sort_values(ascending=False).head(5)
    for rank, (r_name, p_count) in enumerate(ranked_roles.items(), 1):
        sub_r = df[df["Job Title"] == r_name]
        openings = int(pd.to_numeric(sub_r["Demand Volume"], errors="coerce").fillna(50).sum())
        top_dist = sub_r["District/Location"].mode().iloc[0] if not sub_r["District/Location"].empty else "Pune"
        st = sub_r["Curriculum Status"].mode().iloc[0] if not sub_r["Curriculum Status"].empty else "High Demand"
        sub_skills = mlb.transform(sub_r["skills_list"]).sum(axis=0)
        top_s_idx = np.argsort(sub_skills)[::-1][:4]
        r_skills = [unique_skills[i] for i in top_s_idx if sub_skills[i] > 0]
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

    # Update Global Model Cache
    MODEL_CACHE.update({
        "df": df,
        "active_dataset_file": os.path.basename(chosen_path),
        "mlb": mlb,
        "kmeans": kmeans,
        "knn": knn,
        "lr": lr,
        "unique_skills": unique_skills,
        "metrics": metrics,
        "clusters": cluster_summaries,
        "districts": district_list,
        "hottest_roles": hottest_roles,
        "status_distribution": df["Curriculum Status"].value_counts().to_dict(),
        "initialized": True,
    })

    return MODEL_CACHE


# Run initial training on startup
try:
    initialize_data_pipeline()
    print(f"Data pipeline ready. Loaded dataset: {MODEL_CACHE['active_dataset_file']}")
except Exception as e:
    print(f"Startup pipeline notice: {e}")


# ──────────────────────────────────────────────────────────────
# REST API Endpoints
# ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def api_root():
    """Returns REST API overview and documentation."""
    if not MODEL_CACHE.get("initialized"):
        initialize_data_pipeline()

    return jsonify({
        "platform": "Maharashtra Skill Align — Labour-Market Intelligence REST API",
        "problem_statement": "MSINS ID26134 (Government of Maharashtra)",
        "version": "2.0.0",
        "active_dataset": MODEL_CACHE.get("active_dataset_file"),
        "total_rows": len(MODEL_CACHE["df"]) if MODEL_CACHE["df"] is not None else 0,
        "endpoints": {
            "GET /api/metrics": "Returns dataset statistics, unique skills, model accuracy, K-Means clusters, and district training data",
            "POST /api/evaluate": "Accepts {'skills': [...]} as JSON, runs KNN, returns predicted curriculum status, missing skill gaps, and actionable recommendations",
            "GET /api/skills": "Returns complete list of unique technical skills available in the dataset",
            "GET /api/clusters": "Returns the 4 K-Means industry skill domain clusters",
            "GET /api/districts": "Returns Maharashtra district-level capacity training plans",
        }
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    GET /api/metrics
    Returns dataset statistics (total rows, unique skills, status distribution)
    and supervised model evaluation metrics (Accuracy, Precision, Recall, F1-Score).
    """
    if not MODEL_CACHE.get("initialized"):
        initialize_data_pipeline()

    df = MODEL_CACHE["df"]
    unique_skills = MODEL_CACHE["unique_skills"]
    metrics = MODEL_CACHE["metrics"]

    return jsonify({
        "status": "success",
        "dataset": {
            "filename": MODEL_CACHE["active_dataset_file"],
            "total_rows": len(df),
            "total_unique_skills": len(unique_skills),
            "unique_skills": unique_skills,
            "status_distribution": MODEL_CACHE["status_distribution"],
        },
        "model_metrics": metrics,
        "clusters": MODEL_CACHE["clusters"],
        "districts": MODEL_CACHE["districts"],
        "hottest_roles": MODEL_CACHE["hottest_roles"],
    })


@app.route("/api/skills", methods=["GET"])
def get_skills():
    """GET /api/skills: Returns full list of unique skills in the dataset."""
    if not MODEL_CACHE.get("initialized"):
        initialize_data_pipeline()

    return jsonify({
        "status": "success",
        "total_skills": len(MODEL_CACHE["unique_skills"]),
        "skills": MODEL_CACHE["unique_skills"],
    })


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    POST /api/evaluate
    Accepts frontend skills as JSON payload:
      {
        "skills": ["docker", "kubernetes", "aws", "linux"],
        "course_name": "Diploma in Cloud Computing" (optional)
      }
    Runs KNN to predict 'Curriculum Status' (High Demand, Needs Update, Obsolete/Oversupplied),
    finds closest K-Means cluster, identifies missing skill gaps, and returns actionable recommendations.
    """
    if not MODEL_CACHE.get("initialized"):
        initialize_data_pipeline()

    # Extract input skills from JSON body (or form data fallback)
    if request.is_json:
        data = request.get_json(silent=True) or {}
        input_skills = data.get("skills", [])
        course_name = data.get("course_name", "Evaluated Skill Curriculum")
    else:
        input_skills = request.form.getlist("skills")
        course_name = request.form.get("course_name", "Evaluated Skill Curriculum")

    # Clean and filter input skills
    selected_skills = [clean_skill_token(s) for s in input_skills if s and len(clean_skill_token(s)) >= 2]

    if not selected_skills:
        return jsonify({
            "status": "error",
            "message": "Please provide a non-empty 'skills' array in the request body (e.g., {'skills': ['docker', 'kubernetes', 'aws']})."
        }), 400

    mlb = MODEL_CACHE["mlb"]
    knn = MODEL_CACHE["knn"]
    lr = MODEL_CACHE["lr"]
    clusters = MODEL_CACHE["clusters"]
    unique_skills = MODEL_CACHE["unique_skills"]

    # Transform selected skills using MultiLabelBinarizer
    user_vec = mlb.transform([selected_skills])

    # 1. Run KNN Prediction
    knn_pred = knn.predict(user_vec)[0]
    knn_proba = knn.predict_proba(user_vec)[0]
    knn_classes = knn.classes_.tolist()
    knn_proba_dict = {
        knn_classes[i]: round(float(knn_proba[i]) * 100, 1)
        for i in range(len(knn_classes))
    }

    # 2. Run Logistic Regression Prediction
    lr_pred = lr.predict(user_vec)[0]
    lr_proba = lr.predict_proba(user_vec)[0]
    lr_classes = lr.classes_.tolist()
    lr_proba_dict = {
        lr_classes[i]: round(float(lr_proba[i]) * 100, 1)
        for i in range(len(lr_classes))
    }

    predicted_status = str(knn_pred)
    confidence = round(float(np.max(knn_proba)) * 100, 1)

    # 3. Benchmark against K-Means Centroids for Gap Analysis
    selected_set = set(selected_skills)
    vec_flat = user_vec[0]

    best_cluster = None
    best_sim = -1.0
    for c in clusters:
        centroid = np.array(c.get("centroid", []))
        if len(centroid) == len(vec_flat) and np.linalg.norm(vec_flat) > 0 and np.linalg.norm(centroid) > 0:
            sim = float(np.dot(vec_flat, centroid) / (np.linalg.norm(vec_flat) * np.linalg.norm(centroid)))
        else:
            sim = float(len(selected_set & set(c.get("top_skills", []))))
        if sim > best_sim:
            best_sim = sim
            best_cluster = c

    if not best_cluster and clusters:
        best_cluster = clusters[0]

    cluster_top_skills = best_cluster["top_skills"] if best_cluster else []
    matched_skills = [s for s in selected_skills if s in cluster_top_skills or s in unique_skills]
    missing_skills = [s for s in cluster_top_skills if s not in selected_set]

    # 4. Actionable Curriculum Recommendations & Capacity Plans
    if predicted_status == "High Demand":
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
        status_message = "Curriculum covers foundational skills but lacks industry-critical modern tooling."
        readiness = 68
        rec_add = missing_skills[:3] or ["docker", "aws", "ci/cd"]
        action_plan = [
            f"Integrate modern industry tools into course syllabus: add {', '.join(rec_add)}.",
            "Convert 35% of traditional classroom lectures into hands-on lab projects with Git version control.",
            "Align student capstone projects with active hiring requirements in Maharashtra district hubs."
        ]
        equipment_guidance = "Provision cloud container labs (Docker/Kubernetes access) and updated Linux workstations."
        trainer_guidance = f"Conduct faculty upskilling workshops on {', '.join(rec_add)} via industry trainer partnerships."
    else:  # Obsolete/Oversupplied
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

    return jsonify({
        "status": "success",
        "course_name": course_name,
        "curriculum_status": predicted_status,
        "confidence": confidence,
        "probabilities": knn_proba_dict,
        "logistic_regression": {
            "status": str(lr_pred),
            "confidence": round(float(np.max(lr_proba)) * 100, 1),
            "probabilities": lr_proba_dict,
        },
        "aligned_cluster": best_cluster["cluster_name"] if best_cluster else "General Industry Stack",
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "placement_readiness": readiness,
        "recommendations": {
            "status_message": status_message,
            "action_plan": action_plan,
            "equipment_guidance": equipment_guidance,
            "trainer_guidance": trainer_guidance,
            "target_districts": target_districts,
        }
    })


# ──────────────────────────────────────────────────────────────
# Route Aliases for Backward Compatibility
# ──────────────────────────────────────────────────────────────
@app.route("/evaluate", methods=["POST"])
@app.route("/predict", methods=["POST"])
def evaluate_alias():
    return api_evaluate()


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    if not MODEL_CACHE.get("initialized"):
        initialize_data_pipeline()
    return jsonify({
        "status": "success",
        "clusters": MODEL_CACHE["clusters"],
    })


@app.route("/api/districts", methods=["GET"])
def get_districts():
    if not MODEL_CACHE.get("initialized"):
        initialize_data_pipeline()
    return jsonify({
        "status": "success",
        "districts": MODEL_CACHE["districts"],
    })


@app.route("/api/roles", methods=["GET"])
def get_roles():
    if not MODEL_CACHE.get("initialized"):
        initialize_data_pipeline()
    return jsonify({
        "status": "success",
        "hottest_roles": MODEL_CACHE["hottest_roles"],
    })


@app.route("/api/sample-csv", methods=["GET"])
@app.route("/sample-csv", methods=["GET"])
def download_sample_csv():
    path = os.path.join(os.path.dirname(__file__), "scraped_maharashtra_jobs.csv")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "sample_jobs.csv")
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="maharashtra_jobs.csv")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
