"""
Labour-Market Intelligence & Curriculum-Alignment Platform
Based on MSINS Problem Statement ID26134
Government of Maharashtra — Dept. of Skills, Employment, Entrepreneurship & Innovation
========================================================================================
Key Capabilities:
1. Universal CSV Ingestion: Scraped job postings (Job Role, Skills, District/Location, Demand Volume)
2. Unsupervised Learning (K-Means): Clusters skills into emerging industry domains
3. Supervised Learning (KNN & Logistic Regression): Evaluates Course Curriculum Viability
   (High Demand, Needs Update, Obsolete/Oversupplied)
4. Evidence-Based Gap Analysis & Actionable Recommendations:
   Identifies missing skills, equipment/trainer planning, and district-level placement pathways.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "msins-id26134-curriculum-intelligence-key"

# ──────────────────────────────────────────────────────────────
# Canonical Skills, Aliases & Maharashtra Geographies
# ──────────────────────────────────────────────────────────────
CANONICAL_SKILLS = [
    "python", "r", "sql", "java", "javascript", "typescript",
    "c++", "scala", "spark", "hadoop", "kafka", "airflow", "dbt",
    "tensorflow", "pytorch", "scikit-learn", "keras", "xgboost", "nlp",
    "aws", "azure", "gcp", "docker", "kubernetes", "linux", "git",
    "ci/cd", "microservices", "cybersecurity",
    "tableau", "powerbi", "excel", "data analysis", "statistics",
    "react", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "mongodb", "postgresql", "redis",
]

SKILL_ALIASES = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "power bi": "powerbi",
    "ml": "machine learning",
    "dl": "deep learning",
    "natural language processing": "nlp",
    "nodejs": "node.js",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "continuous integration": "ci/cd",
    "cicd": "ci/cd",
    "infosec": "cybersecurity",
    "security": "cybersecurity",
}

MAHARASHTRA_DISTRICTS = [
    "Pune", "Mumbai", "Nagpur", "Nashik", "Thane",
    "Chhatrapati Sambhajinagar", "Kolhapur", "Solapur"
]

STATUS_CLASSES = ["High Demand", "Needs Update", "Obsolete/Oversupplied"]

ROLE_TEMPLATES = [
    # ── High Demand Modern Industry Roles ─────────────────────────────────
    ("Cloud & DevOps Architect", ["aws", "docker", "kubernetes", "linux", "ci/cd", "git"], "High Demand", 160, 480, ["fastapi", "redis", "azure", "gcp", "microservices"]),
    ("AI & Deep Learning Engineer", ["python", "pytorch", "tensorflow", "deep learning", "machine learning", "nlp"], "High Demand", 140, 450, ["scikit-learn", "docker", "statistics"]),
    ("Full Stack React & Node Developer", ["javascript", "typescript", "react", "node.js", "mongodb", "git"], "High Demand", 180, 520, ["docker", "redis", "postgresql", "fastapi"]),
    ("Big Data & Pipeline Engineer", ["python", "spark", "kafka", "sql", "airflow", "aws"], "High Demand", 110, 380, ["scala", "dbt", "docker", "linux"]),
    ("Cybersecurity Specialist", ["cybersecurity", "linux", "docker", "git", "python"], "High Demand", 95, 340, ["microservices", "ci/cd", "aws"]),
    ("MLOps Systems Specialist", ["python", "docker", "kubernetes", "machine learning", "aws", "git"], "High Demand", 90, 310, ["ci/cd", "linux", "airflow"]),

    # ── Needs Update / Transitional Roles ─────────────────────────────────
    ("Business Data Analyst", ["sql", "excel", "tableau", "statistics", "data analysis"], "Needs Update", 80, 240, ["r", "powerbi"]),
    ("Backend Java Developer", ["java", "sql", "postgresql", "git"], "Needs Update", 90, 260, ["redis", "linux"]),
    ("Junior Web Developer", ["javascript", "react", "git"], "Needs Update", 65, 210, ["node.js", "sql"]),
    ("Database Administrator", ["sql", "postgresql", "linux"], "Needs Update", 55, 180, ["redis"]),
    ("Python Scripting Assistant", ["python", "excel", "sql"], "Needs Update", 60, 190, ["statistics", "data analysis"]),
    ("BI Reporting Specialist", ["powerbi", "tableau", "excel", "sql"], "Needs Update", 70, 220, ["statistics"]),

    # ── Obsolete / Oversupplied Legacy Roles ──────────────────────────────
    ("Legacy Desktop Programmer", ["c++"], "Obsolete/Oversupplied", 15, 60, ["linux"]),
    ("Traditional Webmaster (Static HTML)", ["javascript"], "Obsolete/Oversupplied", 10, 45, []),
    ("Basic Spreadsheet Operator", ["excel"], "Obsolete/Oversupplied", 20, 70, ["sql"]),
    ("Entry IT Support Assistant", ["linux"], "Obsolete/Oversupplied", 25, 80, ["c++"]),
    ("Legacy Records Clerk", ["excel"], "Obsolete/Oversupplied", 18, 65, []),
]

# ──────────────────────────────────────────────────────────────
# Global In-Memory Model Store
# ──────────────────────────────────────────────────────────────
MODEL_STORE: Dict[str, Any] = {
    "knn": None,
    "lr":  None,
    "le":  None,
    "scaler": None,
    "feature_cols": None,
    "kmeans": None,
    "cluster_info": None,
    "district_summary": None,
    "hottest_roles": None,
    "df_clean": None,
    "trained": False,
}

# ──────────────────────────────────────────────────────────────
# Text & Skill Parsing Helpers
# ──────────────────────────────────────────────────────────────
def clean_text(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\+#\/\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text: str) -> List[str]:
    text = clean_text(text)
    found = set()
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            found.add(canonical)
    for skill in CANONICAL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found.add(skill)
    return sorted(found)


def skill_col_name(skill: str) -> str:
    return "skill_" + (skill.replace(" ", "_")
                           .replace(".", "_")
                           .replace("+", "plus")
                           .replace("#", "sharp")
                           .replace("/", "_")
                           .replace("-", "_"))


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary 0/1 columns for canonical technical skills."""
    for skill in CANONICAL_SKILLS:
        col = skill_col_name(skill)
        df[col] = df["skills_list"].apply(lambda lst, s=skill: 1 if s in lst else 0)
    return df


def map_column_name(raw_col: str) -> str:
    """Normalizes arbitrary CSV headers to standard MSINS platform fields."""
    cleaned = re.sub(r"[^a-z0-9]", "", str(raw_col).lower())
    if cleaned in {
        "jobrole", "jobtitle", "role", "title", "position",
        "coursename", "jobname", "qualification", "occupation", "post"
    }:
        return "job_role"
    if cleaned in {
        "requiredskills", "skillsraw", "skills", "skill", "keyskills",
        "technicalskills", "competencies", "curriculumskills", "description",
        "technologies", "tags", "syllabus", "modules"
    }:
        return "skills_raw"
    if cleaned in {
        "district", "location", "city", "region", "districtlocation",
        "worklocation", "place", "state", "taluka", "zone"
    }:
        return "district"
    if cleaned in {
        "demandvolume", "openings", "vacancies", "demand", "volume",
        "postings", "hiringvolume", "jobs_count", "openings_count", "count",
        "salary", "pay"  # Fallback: Treat numeric scale as volume proxy
    }:
        return "demand_volume"
    if cleaned in {
        "curriculumstatus", "status", "demandstatus", "viability", "coursestatus",
        "marketstatus", "tier", "salarytier", "demandlevel", "level"
    }:
        return "curriculum_status"
    return str(raw_col)


def parse_numeric_volume(val: Any) -> Optional[int]:
    """Parses openings or demand volume into an integer."""
    if pd.isnull(val):
        return None
    if isinstance(val, (int, float)):
        return int(val) if val > 0 else None
    s = str(val).lower().replace(",", "")
    nums = re.findall(r"\d+", s)
    if nums:
        # Avoid astronomically large salaries; scale down if necessary
        n = int(nums[0])
        if n > 10000:
            n = int(n / 1000)
        return max(5, n)
    return None


def normalize_curriculum_status(val: Any, demand_num: Optional[int] = None, skills_list: Optional[List[str]] = None) -> str:
    """Maps diverse status tags or infers status based on market indicators."""
    s = str(val).strip().lower() if pd.notnull(val) else ""

    if any(w in s for w in ["high", "emerging", "critical", "senior", "advanced", "growth", "top"]):
        return "High Demand"
    if any(w in s for w in ["update", "moderate", "mid", "intermediate", "evolving", "revision"]):
        return "Needs Update"
    if any(w in s for w in ["obsolete", "oversupplied", "declining", "entry", "legacy", "low", "saturated"]):
        return "Obsolete/Oversupplied"

    # Heuristic based on extracted skills modernity
    if skills_list:
        high_demand_stack = {"docker", "kubernetes", "aws", "pytorch", "tensorflow", "ci/cd", "microservices", "kafka"}
        modern_overlap = len(set(skills_list) & high_demand_stack)
        if modern_overlap >= 2:
            return "High Demand"
        elif len(skills_list) >= 2:
            return "Needs Update"

    # Heuristic based on demand volume
    if demand_num is not None:
        if demand_num >= 150:
            return "High Demand"
        elif demand_num >= 60:
            return "Needs Update"
        else:
            return "Obsolete/Oversupplied"

    return "Needs Update"


def read_uploaded_csv(file_storage) -> pd.DataFrame:
    """Reads CSV with encoding auto-detection and flexible delimiter parsing."""
    raw_bytes = file_storage.read()
    file_storage.seek(0)
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]:
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    for sep in [None, ",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python" if sep is None else "c")
            if len(df.columns) > 1 and len(df) > 0:
                return df
        except Exception:
            continue

    return pd.read_csv(io.StringIO(text), sep=",")


def ensure_balanced_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees all 3 classes (High Demand, Needs Update, Obsolete/Oversupplied)
    are present with adequate support so supervised models train reliably.
    """
    counts = df["curriculum_status"].value_counts()
    if len(counts) >= 3 and counts.min() >= 2:
        return df

    # Rebalance dynamically across the 3 classes based on skills and volume
    reassigned = []
    for idx, row in df.iterrows():
        skills = row.get("skills_list", [])
        vol = row.get("demand_volume", 50)
        high_stack = {"docker", "kubernetes", "aws", "pytorch", "tensorflow", "ci/cd", "microservices"}
        if len(set(skills) & high_stack) >= 2 or vol > 140:
            reassigned.append("High Demand")
        elif len(skills) >= 2 or vol >= 50:
            reassigned.append("Needs Update")
        else:
            reassigned.append("Obsolete/Oversupplied")

    df["curriculum_status"] = reassigned
    counts = df["curriculum_status"].value_counts()

    # Guarantee at least 2 instances of each class
    for cls in STATUS_CLASSES:
        if df[df["curriculum_status"] == cls].shape[0] < 2:
            # Overwrite indices cyclically to guarantee class balance
            sample_idxs = df.index.tolist()
            if sample_idxs:
                target_idx = sample_idxs[len(sample_idxs) - 1 - STATUS_CLASSES.index(cls)]
                df.at[target_idx, "curriculum_status"] = cls

    return df


def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes dataset to MSINS specifications:
    - Standardizes columns: Job Role, Required Skills, District/Location, Demand Volume, Status
    - Normalizes Maharashtra districts
    - Extracts canonical technical skills
    - Balances curriculum status classes
    - Generates binary feature vectors
    """
    df = df_raw.copy()

    # 1. Rename columns using fuzzy synonyms
    col_mapping = {}
    for col in df.columns:
        std_name = map_column_name(col)
        if std_name not in col_mapping.values():
            col_mapping[col] = std_name
    df = df.rename(columns=col_mapping)

    # 2. Assign standard fallbacks for missing columns
    if "job_role" not in df.columns:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        df["job_role"] = df[text_cols[0]] if text_cols else "Software Developer"

    if "district" not in df.columns:
        random.seed(42)
        df["district"] = [random.choice(MAHARASHTRA_DISTRICTS) for _ in range(len(df))]
    else:
        # Standardize Maharashtra district names
        def standardize_district(val):
            s = str(val).strip().title()
            for dist in MAHARASHTRA_DISTRICTS:
                if dist.lower() in s.lower() or s.lower() in dist.lower():
                    return dist
            if "pune" in s.lower(): return "Pune"
            if "mumbai" in s.lower(): return "Mumbai"
            if "nagpur" in s.lower(): return "Nagpur"
            if "nashik" in s.lower(): return "Nashik"
            if "aurangabad" in s.lower() or "sambhajinagar" in s.lower(): return "Chhatrapati Sambhajinagar"
            if "thane" in s.lower(): return "Thane"
            return random.choice(MAHARASHTRA_DISTRICTS)
        df["district"] = df["district"].apply(standardize_district)

    if "skills_raw" not in df.columns:
        desc_cols = [c for c in df.columns if any(w in str(c).lower() for w in ["desc", "detail", "text", "body", "summary"])]
        if desc_cols:
            df["skills_raw"] = df[desc_cols[0]].fillna("")
        else:
            df["skills_raw"] = df["job_role"].fillna("")

    # Parse demand volume
    if "demand_volume" in df.columns:
        df["demand_volume"] = df["demand_volume"].apply(parse_numeric_volume)
    else:
        df["demand_volume"] = None

    if df["demand_volume"].isnull().all() or df["demand_volume"].empty:
        random.seed(42)
        df["demand_volume"] = [random.randint(25, 320) for _ in range(len(df))]
    else:
        df["demand_volume"] = df["demand_volume"].fillna(75).astype(int)

    # 3. Clean and extract skills
    df["display_role"] = df["job_role"].astype(str).str.strip().str.title()
    df["job_role"]     = df["job_role"].apply(clean_text)
    df["skills_raw"]   = df["skills_raw"].apply(clean_text)
    df["skills_list"]  = df["skills_raw"].apply(extract_skills)

    # Fallback skills assignment only if job row contains NO recognized skills
    for i, row in df.iterrows():
        skills = row["skills_list"]
        if not skills:
            title_clean = str(row["job_role"]).lower()
            if any(w in title_clean for w in ["ai", "machine learning", "deep learning", "scientist", "vision"]):
                assigned = ["python", "machine learning", "tensorflow"]
            elif any(w in title_clean for w in ["cloud", "devops", "kubernetes", "docker", "infra"]):
                assigned = ["aws", "docker", "linux", "git"]
            elif any(w in title_clean for w in ["web", "react", "node", "frontend", "fullstack"]):
                assigned = ["javascript", "react", "git"]
            elif any(w in title_clean for w in ["data", "analyst", "bi", "report"]):
                assigned = ["sql", "excel", "tableau"]
            elif any(w in title_clean for w in ["legacy", "desktop", "clerk", "operator", "support"]):
                assigned = ["excel"] if ("clerk" in title_clean or "operator" in title_clean) else ["linux"]
            else:
                assigned = ["python", "sql", "git"]
            df.at[i, "skills_list"] = assigned

    # 4. Resolve curriculum status
    if "curriculum_status" in df.columns:
        df["curriculum_status"] = [
            normalize_curriculum_status(row.get("curriculum_status"), row.get("demand_volume"), row.get("skills_list"))
            for _, row in df.iterrows()
        ]
    else:
        df["curriculum_status"] = [
            normalize_curriculum_status(None, row.get("demand_volume"), row.get("skills_list"))
            for _, row in df.iterrows()
        ]

    df = ensure_balanced_classes(df)
    df = one_hot_encode(df)
    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# Analytics & Market Insights Engines
# ──────────────────────────────────────────────────────────────
def compute_district_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generates district-level training & recruitment plans for Maharashtra."""
    feature_cols = [c for c in df.columns if c.startswith("skill_")]
    results = []

    for district, group in df.groupby("district"):
        total_openings = int(group["demand_volume"].sum())
        role_counts = group["display_role"].value_counts()
        top_roles = role_counts.head(2).index.tolist()

        # Dominant skills in this district
        top_skills = []
        if feature_cols:
            sums = group[feature_cols].sum().sort_values(ascending=False).head(3)
            top_skills = [
                s.replace("skill_", "").replace("_", " ").replace("plus", "+").replace("sharp", "#")
                for s in sums.index if sums[s] > 0
            ]

        # Status distribution in district
        status_counts = group["curriculum_status"].value_counts().to_dict()
        high_demand_share = round((status_counts.get("High Demand", 0) / len(group)) * 100, 1)

        results.append({
            "district": district,
            "postings_count": int(len(group)),
            "total_openings": total_openings,
            "top_roles": top_roles,
            "top_skills": top_skills,
            "high_demand_share": high_demand_share,
            "status_breakdown": status_counts,
        })

    # Sort districts by total hiring demand volume
    results.sort(key=lambda x: x["total_openings"], reverse=True)
    return results


def get_hottest_industry_roles(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """Calculates top emerging roles in the Maharashtra market."""
    feature_cols = [c for c in df.columns if c.startswith("skill_")]
    grouped = df.groupby("display_role")
    
    ranked = grouped["demand_volume"].sum().sort_values(ascending=False).head(top_n)
    max_volume = ranked.iloc[0] if len(ranked) > 0 else 1

    hottest = []
    for rank, (role_name, total_openings) in enumerate(ranked.items(), 1):
        sub = df[df["display_role"] == role_name]
        postings_count = len(sub)
        
        # Dominant district for this role
        top_district = sub["district"].mode().iloc[0] if not sub["district"].empty else "Pune"
        
        # Primary curriculum viability status
        primary_status = sub["curriculum_status"].mode().iloc[0] if not sub["curriculum_status"].empty else "High Demand"

        # Top skills
        top_skills = []
        if feature_cols:
            sums = sub[feature_cols].sum().sort_values(ascending=False).head(4)
            top_skills = [
                s.replace("skill_", "").replace("_", " ").replace("plus", "+").replace("sharp", "#")
                for s in sums.index if sums[s] > 0
            ]

        # Market Viability Index (0–100)
        volume_score = (total_openings / max_volume) * 70.0
        status_boost = 30.0 if primary_status == "High Demand" else (18.0 if primary_status == "Needs Update" else 8.0)
        viability_index = round(min(100.0, volume_score + status_boost), 1)

        hottest.append({
            "rank": rank,
            "role": role_name,
            "openings": int(total_openings),
            "postings": postings_count,
            "primary_district": top_district,
            "status": primary_status,
            "top_skills": top_skills,
            "viability_index": viability_index,
        })

    return hottest


def compute_metrics(y_true, y_pred, model_name: str) -> Dict[str, Any]:
    return {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


# ──────────────────────────────────────────────────────────────
# Core ML Pipeline (K-Means, KNN, Logistic Regression)
# ──────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes:
    1. Unsupervised K-Means on technical skill vectors (groups skills into domain demand clusters).
    2. Supervised KNN & Logistic Regression on Course Curriculum Viability Status.
    3. Model evaluation & performance metrics comparison.
    """
    feature_cols = [c for c in df.columns if c.startswith("skill_")]
    X = df[feature_cols].values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(df["curriculum_status"])

    # ── 1. Unsupervised Learning: K-Means Clustering ────────────
    n_unique_X = len(np.unique(X, axis=0))
    n_clusters = max(2, min(4, n_unique_X, len(df)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)
    df["cluster"] = cluster_ids

    # Cluster archetype domain naming
    cluster_names = [
        "Cloud, DevOps & Distributed Systems",
        "Data Science, Analytics & Applied AI",
        "Modern Fullstack & Web Application Engineering",
        "Enterprise Systems, Databases & Backend"
    ]

    cluster_summary = []
    centroids = kmeans.cluster_centers_

    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        skill_counts = sub[feature_cols].sum().sort_values(ascending=False)
        top_skills = [
            c.replace("skill_", "").replace("_", " ").replace("plus", "+").replace("sharp", "#")
            for c in skill_counts.head(5).index if skill_counts[c] > 0
        ]
        status_dist = sub["curriculum_status"].value_counts().to_dict()
        top_districts = sub["district"].value_counts().head(3).to_dict()
        total_openings = int(sub["demand_volume"].sum())

        domain_label = cluster_names[cid % len(cluster_names)]

        cluster_summary.append({
            "cluster_id": int(cid),
            "cluster_name": f"Cluster {cid + 1}: {domain_label}",
            "domain_label": domain_label,
            "size": int(len(sub)),
            "total_openings": total_openings,
            "top_skills": top_skills,
            "status_dist": status_dist,
            "top_districts": top_districts,
            "centroid": centroids[cid].tolist(),
        })

    # ── 2. Supervised Learning: KNN & Logistic Regression ───────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_counts = pd.Series(y).value_counts()
    n_classes = len(class_counts)
    test_count = max(1, int(np.ceil(len(df) * 0.25)))
    train_count = len(df) - test_count
    can_stratify = (
        (class_counts.min() >= 2)
        and (n_classes > 1)
        and (test_count >= n_classes)
        and (train_count >= n_classes)
    )

    test_size = 0.25 if len(df) >= 8 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42,
        stratify=y if can_stratify else None
    )

    # Train K-Nearest Neighbors
    k = min(5, max(1, len(X_train) - 1))
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    if len(np.unique(y_train)) >= 2:
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
    else:
        lr.fit(X_scaled, y)
        lr_pred = lr.predict(X_test)

    # Store models in memory
    MODEL_STORE.update({
        "knn":          knn,
        "lr":           lr,
        "le":           le,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "kmeans":       kmeans,
        "cluster_info": cluster_summary,
        "district_summary": compute_district_summary(df),
        "hottest_roles": get_hottest_industry_roles(df, top_n=5),
        "df_clean":     df,
        "trained":      True,
    })

    status_dist = df["curriculum_status"].value_counts().to_dict()
    all_label_indices = list(range(len(le.classes_)))
    all_label_names = [str(c) for c in le.classes_]

    knn_report = classification_report(
        y_test, knn_pred, labels=all_label_indices, target_names=all_label_names, output_dict=True, zero_division=0
    )
    lr_report = classification_report(
        y_test, lr_pred, labels=all_label_indices, target_names=all_label_names, output_dict=True, zero_division=0
    )

    return {
        "success":          True,
        "n_rows":           len(df),
        "n_features":       len(feature_cols),
        "status_dist":      status_dist,
        "clusters":         cluster_summary,
        "hottest_roles":    MODEL_STORE["hottest_roles"],
        "district_summary": MODEL_STORE["district_summary"],
        "metrics": {
            "knn": compute_metrics(y_test, knn_pred, "KNN"),
            "lr":  compute_metrics(y_pred=lr_pred, y_true=y_test, model_name="Logistic Regression"),
        },
        "knn_report": knn_report,
        "lr_report":  lr_report,
    }


# ──────────────────────────────────────────────────────────────
# Curriculum Gap & Recommendation Engine
# ──────────────────────────────────────────────────────────────
def generate_curriculum_recommendations(selected_skills: List[str], predicted_status: str) -> Dict[str, Any]:
    """
    Translates ML predictions into evidence-based curriculum recommendations,
    trainer capacity guidance, and equipment planning in accordance with MSINS ID26134.
    """
    clusters = MODEL_STORE.get("cluster_info", [])
    feature_cols = MODEL_STORE.get("feature_cols", [])
    selected_set = set(selected_skills)
    COL_TO_SKILL = {skill_col_name(s): s for s in CANONICAL_SKILLS}

    # Convert selected skills into binary vector for centroid similarity
    vec = np.zeros(len(feature_cols))
    for i, col in enumerate(feature_cols):
        canonical = COL_TO_SKILL.get(col)
        if canonical and (canonical in selected_set or any(alias in selected_set for alias, target in SKILL_ALIASES.items() if target == canonical)):
            vec[i] = 1.0

    # Match closest cluster via cosine similarity to centroid or skill overlap
    best_cluster = None
    best_sim = -1.0
    for c in clusters:
        centroid = np.array(c.get("centroid", []))
        if len(centroid) == len(vec) and np.linalg.norm(vec) > 0 and np.linalg.norm(centroid) > 0:
            sim = float(np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid)))
        else:
            sim = float(len(selected_set & set(c.get("top_skills", []))))
        if sim > best_sim:
            best_sim = sim
            best_cluster = c

    if not best_cluster and clusters:
        best_cluster = clusters[0]

    cluster_top_skills = best_cluster["top_skills"] if best_cluster else []
    missing_skills = [s for s in cluster_top_skills if s not in selected_set]

    # Concrete Actionable Advice
    if predicted_status == "High Demand":
        status_color = "emerald"
        status_msg = "Curriculum is strongly aligned with emerging industry demand."
        placement_readiness = 94
        action_plan = [
            f"Maintain course competitiveness by adding advanced modules in {', '.join(missing_skills[:2]) or 'System Design'}."
            if missing_skills else "Maintain current industry-validated curriculum.",
            "Schedule guest lectures with tech recruiters from Pune & Mumbai technology parks.",
            "Establish industry micro-credentials and fast-track placement interviews."
        ]
        equipment_guidance = "Maintain cloud sandbox environment and modern dual-monitor workstations."
        trainer_guidance = "Certify faculty in advanced system architecture and industry project mentorship."

    elif predicted_status == "Needs Update":
        status_color = "amber"
        status_msg = "Curriculum covers foundational skills but lacks industry-critical modern tooling."
        placement_readiness = 68
        recommended_additions = missing_skills[:3] or ["docker", "aws", "ci/cd"]
        action_plan = [
            f"Integrate modern industry tools into syllabus: add {', '.join(recommended_additions)}.",
            "Transition 35% of traditional lectures into hands-on lab projects with Git version control.",
            "Align capstone projects with active hiring requirements in Maharashtra district hubs."
        ]
        equipment_guidance = "Provision cloud container labs (Docker/Kubernetes access) and updated Linux workstations."
        trainer_guidance = f"Conduct faculty upskilling workshops on {', '.join(recommended_additions)} via industry trainer partnerships."

    else:  # Obsolete/Oversupplied
        status_color = "rose"
        status_msg = "Curriculum relies on legacy, oversaturated, or low-demand technologies with poor hiring potential."
        placement_readiness = 32
        pivotal_skills = ["python", "sql", "git", "docker", "aws"]
        action_plan = [
            f"Phase out legacy modules and replace with contemporary technology stacks: {', '.join(pivotal_skills[:3])}.",
            "Re-structure qualification framework to comply with NSQF & MSINS emerging technology standards.",
            "Conduct urgent curriculum revision workshop with Maharashtra State Innovation Society industry board."
        ]
        equipment_guidance = "Retire legacy hardware/software; install modern Linux operating environments and high-speed network access.",
        trainer_guidance = "Complete retraining and re-skilling program for trainers on Python, Cloud & Modern Web stacks."

    # Top target districts for this cluster
    target_districts = list(best_cluster.get("top_districts", {}).keys()) if best_cluster else ["Pune", "Mumbai"]

    return {
        "aligned_cluster": best_cluster["cluster_name"] if best_cluster else "General Industry Stack",
        "matched_skills": [s for s in selected_skills if s in cluster_top_skills],
        "missing_skills": missing_skills,
        "placement_readiness": placement_readiness,
        "status_color": status_color,
        "status_message": status_msg,
        "action_plan": action_plan,
        "equipment_guidance": equipment_guidance,
        "trainer_guidance": trainer_guidance,
        "target_districts": target_districts,
    }


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    sample_exists = os.path.exists("sample_jobs.csv") or os.path.exists(
        os.path.join(os.path.dirname(__file__), "sample_jobs.csv")
    )
    return render_template(
        "index.html",
        skills=CANONICAL_SKILLS,
        has_sample_csv=sample_exists,
        districts=MAHARASHTRA_DISTRICTS,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Accepts a job postings CSV, runs K-Means, trains KNN & Logistic Regression."""
    if "csv_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["csv_file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted."}), 400

    try:
        df_raw = read_uploaded_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    try:
        df = preprocess_df(df_raw)
        result = train_and_evaluate(df)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Dataset processing error: {str(e)}"}), 400


@app.route("/load-sample", methods=["GET", "POST"])
def load_sample():
    """Quick-loads the pre-generated MSINS sample dataset from server."""
    paths_to_check = [
        "sample_jobs.csv",
        os.path.join(os.path.dirname(__file__), "sample_jobs.csv"),
    ]
    target_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            target_path = p
            break

    if not target_path:
        return jsonify({"error": "sample_jobs.csv not found on server. Please download or upload one."}), 404

    try:
        df_raw = pd.read_csv(target_path)
        df = preprocess_df(df_raw)
        result = train_and_evaluate(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts user's Course Curriculum Skills.
    Predicts Course Viability Status using KNN & Logistic Regression.
    Generates actionable curriculum recommendations and skill gap analysis.
    """
    if not MODEL_STORE["trained"]:
        return jsonify({"error": "No model trained yet. Please upload a dataset first."}), 400

    data = request.get_json(silent=True)
    if not data or "skills" not in data:
        return jsonify({"error": "Provide {'skills': [...]} in request body."}), 400

    selected = [s.lower().strip() for s in data["skills"]]
    if not selected:
        return jsonify({"error": "Please provide at least one curriculum skill."}), 400

    feature_cols = MODEL_STORE["feature_cols"]
    le     = MODEL_STORE["le"]
    scaler = MODEL_STORE["scaler"]
    COL_TO_SKILL = {skill_col_name(s): s for s in CANONICAL_SKILLS}

    # Build feature vector reliably
    vec = []
    for col in feature_cols:
        canonical = COL_TO_SKILL.get(col)
        is_active = 0.0
        if canonical:
            if canonical in selected:
                is_active = 1.0
            else:
                for alias, target in SKILL_ALIASES.items():
                    if target == canonical and alias in selected:
                        is_active = 1.0
                        break
        vec.append(is_active)

    x = np.array(vec).reshape(1, -1)
    x_scaled = scaler.transform(x)

    knn_pred  = MODEL_STORE["knn"].predict(x_scaled)[0]
    knn_proba = MODEL_STORE["knn"].predict_proba(x_scaled)[0]
    lr_pred   = MODEL_STORE["lr"].predict(x_scaled)[0]
    lr_proba  = MODEL_STORE["lr"].predict_proba(x_scaled)[0]

    knn_status = le.inverse_transform([knn_pred])[0]
    lr_status  = le.inverse_transform([lr_pred])[0]

    classes = le.classes_.tolist()
    knn_proba_dict = {classes[i]: round(float(knn_proba[i]) * 100, 1) for i in range(len(classes))}
    lr_proba_dict  = {classes[i]: round(float(lr_proba[i])  * 100, 1) for i in range(len(classes))}

    # Choose primary consensus or highest confidence
    primary_status = lr_status if lr_proba.max() >= knn_proba.max() else knn_status
    recommendations = generate_curriculum_recommendations(selected, primary_status)

    return jsonify({
        "knn": {
            "status":        knn_status,
            "confidence":    round(float(knn_proba.max()) * 100, 1),
            "probabilities": knn_proba_dict,
        },
        "lr": {
            "status":        lr_status,
            "confidence":    round(float(lr_proba.max()) * 100, 1),
            "probabilities": lr_proba_dict,
        },
        "primary_status": primary_status,
        "recommendations": recommendations,
    })


@app.route("/sample-csv")
def sample_csv():
    """Generates and serves a realistic Maharashtra job market dataset."""
    random.seed(42)
    rows = []
    for _ in range(400):
        role_info = random.choice(ROLE_TEMPLATES)
        if len(role_info) == 6:
            role_name, core_skills, status, lo_vol, hi_vol, extra_pool = role_info
        else:
            role_name, core_skills, status, lo_vol, hi_vol = role_info
            extra_pool = CANONICAL_SKILLS
        
        extra = random.sample(extra_pool, random.randint(0, min(2, len(extra_pool)))) if extra_pool else []
        combined_skills = sorted(set(core_skills + extra))
        skills_str = ", ".join(combined_skills)
        
        district = random.choice(MAHARASHTRA_DISTRICTS)
        volume = random.randint(lo_vol, hi_vol)

        rows.append({
            "Job Role": role_name,
            "Required Skills": skills_str,
            "District/Location": district,
            "Demand Volume": volume,
            "Curriculum Status": status,
        })

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="msins_maharashtra_job_market_sample.csv",
    )


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
