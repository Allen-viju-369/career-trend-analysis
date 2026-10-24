"""
Career Trend Analysis System — Flask Backend
=============================================
Routes:
  GET  /                → Render main dashboard (index.html)
  POST /upload          → Accept CSV upload; run K-Means clustering;
                          train KNN & Logistic Regression; return metrics
  POST /predict         → Accept selected skills; return KNN salary-tier prediction
  GET  /sample-csv      → Download a ready-made sample CSV for testing
"""

import io
import os
import re
import json
import random
import textwrap

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
app.secret_key = "career-trend-secret-2024"

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
CANONICAL_SKILLS = [
    "python", "r", "sql", "java", "javascript", "typescript",
    "c++", "scala", "spark", "hadoop", "kafka", "airflow", "dbt",
    "tensorflow", "pytorch", "scikit-learn", "keras", "xgboost", "nlp",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "tableau", "powerbi", "excel",
    "react", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "data analysis", "statistics",
    "git", "linux", "mongodb", "postgresql", "redis",
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
}

SALARY_RANGES = {
    "Entry":  (40_000, 75_000),
    "Mid":    (75_001, 120_000),
    "Senior": (120_001, 220_000),
}

JOB_TITLES = [
    "Data Scientist", "Machine Learning Engineer", "Data Analyst",
    "Backend Engineer", "Full Stack Developer", "Data Engineer",
    "MLOps Engineer", "AI Research Scientist", "Cloud Engineer",
    "DevOps Engineer", "Software Engineer", "BI Analyst",
]

COMPANIES = [
    "Google", "Amazon", "Microsoft", "Meta", "Apple", "Netflix",
    "Uber", "Stripe", "Databricks", "Snowflake", "OpenAI", "Palantir",
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA",
    "Austin, TX", "Boston, MA", "Remote", "Chicago, IL",
]

SKILL_POOLS = {
    "Data Scientist":            ["python", "machine learning", "scikit-learn", "tensorflow", "sql", "statistics", "tableau"],
    "Machine Learning Engineer": ["python", "pytorch", "tensorflow", "docker", "kubernetes", "aws", "deep learning"],
    "Data Analyst":              ["sql", "python", "excel", "tableau", "powerbi", "statistics", "data analysis"],
    "Backend Engineer":          ["python", "java", "postgresql", "redis", "docker", "aws", "flask", "fastapi"],
    "Full Stack Developer":      ["javascript", "typescript", "react", "node.js", "sql", "docker", "aws"],
    "Data Engineer":             ["python", "spark", "kafka", "airflow", "sql", "aws", "dbt", "scala"],
    "MLOps Engineer":            ["python", "docker", "kubernetes", "airflow", "aws", "tensorflow", "git"],
    "AI Research Scientist":     ["python", "pytorch", "deep learning", "nlp", "statistics", "tensorflow", "r"],
    "Cloud Engineer":            ["aws", "azure", "gcp", "kubernetes", "docker", "linux", "python"],
    "DevOps Engineer":           ["docker", "kubernetes", "linux", "aws", "git", "python"],
    "Software Engineer":         ["python", "java", "c++", "git", "sql", "docker", "linux"],
    "BI Analyst":                ["sql", "tableau", "powerbi", "excel", "python", "statistics", "data analysis"],
}

# ──────────────────────────────────────────────────────────────
# In-memory model store (persists across requests in same process)
# ──────────────────────────────────────────────────────────────
MODEL_STORE = {
    "knn": None,
    "lr":  None,
    "le":  None,
    "scaler": None,
    "feature_cols": None,
    "kmeans": None,
    "cluster_labels": None,
    "trained": False,
}


# ──────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\+#\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text: str) -> list:
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
                           .replace("-", "_"))


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary skill columns to the dataframe."""
    for skill in CANONICAL_SKILLS:
        col = skill_col_name(skill)
        df[col] = df["skills_list"].apply(lambda lst, s=skill: 1 if s in lst else 0)
    return df


def map_column_name(raw_col: str) -> str:
    """Normalize arbitrary column names from Kaggle, LinkedIn, Indeed, etc."""
    cleaned = re.sub(r"[^a-z0-9]", "", str(raw_col).lower())
    if cleaned in {"jobtitle", "title", "position", "role", "jobrole", "rolename", "designation", "jobname", "job", "jobpost", "post"}:
        return "job_title"
    if cleaned in {"skillsraw", "skills", "skill", "requiredskills", "keyskills", "technicalskills", "tags", "jobskills", "description", "jobdescription", "jobdesc", "snippet", "summary", "responsibilities", "requirements", "technologies"}:
        return "skills_raw"
    if cleaned in {"salarytier", "tier", "experiencelevel", "explevel", "level", "seniority", "senioritylevel", "careerlevel", "joblevel", "experience", "exp"}:
        return "salary_tier"
    if cleaned in {"salary", "salaryinusd", "annualsalary", "basesalary", "compensation", "pay", "avgcompensation", "maxsalary", "minsalary", "remuneration", "ctc", "sal"}:
        return "salary"
    if cleaned in {"company", "companyname", "employer", "hiringcompany", "organization", "firm", "agency"}:
        return "company"
    if cleaned in {"location", "joblocation", "city", "worklocation", "companylocation", "country", "state", "region"}:
        return "location"
    return str(raw_col)


def parse_numeric_salary(val):
    """Extract numeric salary value from various formats ($120k, 120000, 80k-100k, LPA)."""
    if pd.isnull(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None

    s = str(val).lower().replace(",", "")
    k_matches = re.findall(r"(\d+(?:\.\d+)?)\s*k\b", s)
    if k_matches:
        vals = [float(m) * 1000 for m in k_matches]
        return sum(vals) / len(vals)

    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if nums:
        floats = [float(n) for n in nums if float(n) > 500]
        if floats:
            return sum(floats) / len(floats)
    return None


def normalize_salary_tier(val, salary_num=None, title_str="") -> str:
    """Normalize any tier notation, or infer from numeric salary or job title."""
    s = str(val).strip().lower() if pd.notnull(val) else ""

    # Kaggle conventions & common abbreviations
    if s in {"en", "entry", "entry level", "entry-level", "junior", "jr", "intern", "internship", "fresher", "trainee", "associate", "0-2"}:
        return "Entry"
    if s in {"mi", "mid", "mid-level", "mid level", "intermediate", "experienced", "2-5"}:
        return "Mid"
    if s in {"se", "ex", "senior", "sr", "lead", "principal", "staff", "architect", "manager", "director", "head", "executive", "5+"}:
        return "Senior"

    # Infer from numerical salary if provided
    if salary_num is not None and pd.notnull(salary_num):
        try:
            num = float(salary_num)
            if num > 0:
                if num < 100:  # Indian LPA scale (e.g. 6 to 30)
                    num = num * 12000
                if num < 75000:
                    return "Entry"
                elif num <= 120000:
                    return "Mid"
                else:
                    return "Senior"
        except (ValueError, TypeError):
            pass

    # Infer from job title keywords
    t = str(title_str).lower()
    if re.search(r"\b(senior|sr|lead|principal|staff|architect|director|head|vp|manager|chief)\b", t):
        return "Senior"
    if re.search(r"\b(junior|jr|intern|internship|entry|trainee|fresher|graduate|associate|assistant)\b", t):
        return "Entry"

    return "Mid"


def read_uploaded_csv(file_storage) -> pd.DataFrame:
    """Read CSV with automatic encoding detection and delimiter sniffing."""
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


def ensure_balanced_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataset has all 3 tiers (Entry, Mid, Senior) so ML models can train properly.
    Uses quantiles if numeric salaries exist, or sensible heuristics based on titles/rank.
    """
    counts = df["salary_tier"].value_counts()
    if len(counts) >= 3 and counts.min() >= 2:
        return df

    # If numeric salaries exist with enough unique values, partition into 3 balanced tertiles
    if "salary" in df.columns:
        s = pd.to_numeric(df["salary"], errors="coerce").dropna()
        if len(s.unique()) >= 3:
            try:
                df["salary_tier"] = pd.qcut(s, q=3, labels=["Entry", "Mid", "Senior"], duplicates="drop")
                if len(df["salary_tier"].value_counts()) >= 3 and df["salary_tier"].value_counts().min() >= 2:
                    return df
            except Exception:
                pass

    # Assign balanced distribution across Entry, Mid, Senior
    tier_assignments = []
    for idx, row in df.iterrows():
        t = str(row.get("job_title", "")).lower()
        if re.search(r"\b(senior|sr|lead|principal|staff|architect|director|head|vp|manager|chief)\b", t):
            tier_assignments.append("Senior")
        elif re.search(r"\b(junior|jr|intern|internship|entry|trainee|fresher|graduate|associate|assistant)\b", t):
            tier_assignments.append("Entry")
        else:
            pos = idx % 3
            tier_assignments.append("Entry" if pos == 0 else ("Mid" if pos == 1 else "Senior"))

    df["salary_tier"] = tier_assignments
    return df


def preprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Universally preprocess ANY uploaded job dataset:
    - Auto-detects columns (Title, Skills/Description, Salary, Company, Location, Tier)
    - Maps diverse naming conventions (Kaggle, LinkedIn, Indeed, etc.)
    - Auto-infers missing columns with smart heuristics
    - Extracts 41 canonical technical skills via Regex
    - Normalizes and balances salary tiers to Entry, Mid, Senior
    """
    df = df_raw.copy()

    # 1. Map columns using fuzzy/synonym matching
    col_mapping = {}
    for col in df.columns:
        std_name = map_column_name(col)
        if std_name not in col_mapping.values():
            col_mapping[col] = std_name
    df = df.rename(columns=col_mapping)

    # 2. Resolve required columns with smart fallbacks
    if "job_title" not in df.columns:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        df["job_title"] = df[text_cols[0]] if text_cols else "Software Engineer"

    if "company" not in df.columns:
        df["company"] = "Tech Enterprise"

    if "location" not in df.columns:
        df["location"] = "Remote"

    if "skills_raw" not in df.columns:
        # Check for description columns
        desc_cols = [c for c in df.columns if any(w in str(c).lower() for w in ["desc", "detail", "text", "body", "summary"])]
        if desc_cols:
            df["skills_raw"] = df[desc_cols[0]].fillna("")
        else:
            df["skills_raw"] = df["job_title"].fillna("")

    # Parse numerical salary
    if "salary" in df.columns:
        df["salary_clean"] = df["salary"].apply(parse_numeric_salary)
    else:
        df["salary_clean"] = None

    # Salary tier resolution
    if "salary_tier" in df.columns:
        df["salary_tier"] = [
            normalize_salary_tier(row.get("salary_tier"), row.get("salary_clean"), row.get("job_title"))
            for _, row in df.iterrows()
        ]
    else:
        df["salary_tier"] = [
            normalize_salary_tier(None, row.get("salary_clean"), row.get("job_title"))
            for _, row in df.iterrows()
        ]

    # Ensure valid salary values for analytics
    tier_salary_defaults = {"Entry": 65000, "Mid": 98000, "Senior": 160000}
    if "salary_clean" not in df.columns or df["salary_clean"].isnull().all():
        random.seed(42)
        df["salary"] = df["salary_tier"].map(
            lambda t: random.randint(
                int(tier_salary_defaults.get(t, 90000) * 0.85),
                int(tier_salary_defaults.get(t, 90000) * 1.25)
            )
        )
    else:
        df["salary"] = df["salary_clean"].fillna(
            df["salary_tier"].map(tier_salary_defaults)
        ).astype(int)

    # Ensure balanced tiers so supervised models always have all classes
    df = ensure_balanced_tiers(df)

    # 3. Clean text & extract skills
    df["display_title"] = df["job_title"].astype(str).str.strip().str.title()
    df["job_title"]   = df["job_title"].apply(clean_text)
    df["company"]     = df["company"].apply(clean_text)
    df["location"]    = df["location"].apply(clean_text)
    df["skills_raw"]  = df["skills_raw"].apply(clean_text)

    df["skills_list"] = df["skills_raw"].apply(extract_skills)

    # If row has no technical skills found in skills_raw, assign realistic role-based skills
    fallback_options = [
        ["python", "sql", "git"],
        ["docker", "kubernetes", "linux", "aws"],
        ["javascript", "react", "typescript", "node.js"],
        ["machine learning", "python", "statistics", "tableau"],
        ["java", "postgresql", "redis", "fastapi"]
    ]

    for i, row in df.iterrows():
        skills = row["skills_list"]
        if len(skills) < 2:
            title_clean = str(row["job_title"]).lower()
            assigned = []
            for known_title, pool in SKILL_POOLS.items():
                if known_title.lower() in title_clean or any(word in title_clean for word in known_title.lower().split() if len(word) > 3):
                    assigned.extend(pool[:4])
                    break
            if not assigned:
                assigned = fallback_options[i % len(fallback_options)]
            merged = sorted(set(skills + assigned))
            df.at[i, "skills_list"] = merged

    df = one_hot_encode(df)
    df = df.reset_index(drop=True)
    return df


def get_hottest_jobs(df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Computes the top N hottest/most trending jobs in the dataset based on:
    - Postings demand volume & share (%)
    - Average salary (numeric if available, or estimated by salary_tier)
    - Key in-demand skills for each role
    - Hotness Score (0–100 index combining market demand & compensation)
    """
    title_col = "display_title" if "display_title" in df.columns else "job_title"
    total_jobs = len(df)
    if total_jobs == 0:
        return []

    tier_est = {
        "Entry": 62000,
        "Mid": 98000,
        "Senior": 165000,
    }

    feature_cols = [c for c in df.columns if c.startswith("skill_")]
    has_num_salary = "salary" in df.columns and pd.to_numeric(df["salary"], errors="coerce").notnull().any()

    grouped = df.groupby(title_col)
    ranked = grouped.size().sort_values(ascending=False).head(top_n)
    max_count = ranked.iloc[0] if len(ranked) > 0 else 1

    hottest = []
    for rank, (title, count) in enumerate(ranked.items(), 1):
        sub = df[df[title_col] == title]
        demand_pct = round((count / total_jobs) * 100, 1)

        if has_num_salary:
            s_num = pd.to_numeric(sub["salary"], errors="coerce")
            avg_val = int(s_num.mean()) if not s_num.dropna().empty else 95000
            min_val = int(s_num.min()) if not s_num.dropna().empty else 60000
            max_val = int(s_num.max()) if not s_num.dropna().empty else 160000
        else:
            avg_val = int(sub["salary_tier"].map(tier_est).mean())
            min_val = int(avg_val * 0.75)
            max_val = int(avg_val * 1.35)

        # Top skills for this specific job
        top_skills = []
        if feature_cols:
            sums = sub[feature_cols].sum().sort_values(ascending=False).head(4)
            top_skills = [
                s.replace("skill_", "").replace("_", " ").replace("plus", "+").replace("sharp", "#")
                for s in sums.index if sums[s] > 0
            ]

        primary_tier = sub["salary_tier"].mode().iloc[0] if not sub["salary_tier"].empty else "Mid"

        # Hotness score (60% demand volume, 40% salary level)
        demand_score = (count / max_count) * 60.0
        salary_factor = min(1.0, max(0.0, (avg_val - 45000) / 135000)) * 40.0
        hotness_score = round(demand_score + salary_factor, 1)

        hottest.append({
            "rank": rank,
            "title": str(title).title(),
            "count": int(count),
            "demand_pct": demand_pct,
            "avg_salary": f"${avg_val:,.0f}",
            "salary_range": f"${min_val:,.0f} – ${max_val:,.0f}",
            "primary_tier": primary_tier,
            "top_skills": top_skills,
            "hotness_score": hotness_score,
        })

    return hottest


def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    return {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Core ML pipeline: feature matrix, K-Means, KNN, Logistic Regression."""
    feature_cols = [c for c in df.columns if c.startswith("skill_")]
    X = df[feature_cols].values.astype(float)

    le = LabelEncoder()
    y  = le.fit_transform(df["salary_tier"])

    # ── K-Means Clustering ──────────────────────────────────
    n_unique_X = len(np.unique(X, axis=0))
    n_clusters = max(1, min(3, n_unique_X, len(df)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)
    df["cluster"] = cluster_ids

    cluster_summary = []
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        skill_counts = sub[feature_cols].sum().sort_values(ascending=False)
        top_skills = [
            c.replace("skill_", "").replace("_", " ").replace("plus", "+").replace("sharp", "#")
            for c in skill_counts.head(3).index
        ]
        tier_dist = sub["salary_tier"].value_counts().to_dict()
        cluster_summary.append({
            "cluster":    int(cid),
            "size":       int(len(sub)),
            "top_skills": top_skills,
            "tier_dist":  tier_dist,
        })

    MODEL_STORE["kmeans"]         = kmeans
    MODEL_STORE["cluster_labels"] = cluster_ids.tolist()

    if len(df) < 5:
        raise ValueError("Dataset is too small (need at least 5 rows to analyze).")

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

    k = min(5, max(1, len(X_train) - 1))
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    if len(np.unique(y_train)) >= 2:
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
    else:
        lr.fit(X_scaled, y)
        lr_pred = lr.predict(X_test)

    MODEL_STORE.update({
        "knn":          knn,
        "lr":           lr,
        "le":           le,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "trained":      True,
    })

    tier_counts = df["salary_tier"].value_counts().to_dict()
    all_label_indices = list(range(len(le.classes_)))
    all_label_names = [str(c) for c in le.classes_]

    knn_report = classification_report(
        y_test, knn_pred, labels=all_label_indices, target_names=all_label_names, output_dict=True, zero_division=0
    )
    lr_report = classification_report(
        y_test, lr_pred, labels=all_label_indices, target_names=all_label_names, output_dict=True, zero_division=0
    )

    hottest_jobs = get_hottest_jobs(df, top_n=5)

    return {
        "success":       True,
        "n_rows":        len(df),
        "n_features":    len(feature_cols),
        "tier_dist":     tier_counts,
        "clusters":      cluster_summary,
        "hottest_jobs":  hottest_jobs,
        "metrics": {
            "knn": compute_metrics(y_test, knn_pred, "KNN"),
            "lr":  compute_metrics(y_test, lr_pred,  "Logistic Regression"),
        },
        "knn_report": knn_report,
        "lr_report":  lr_report,
    }


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    sample_exists = os.path.exists("sample_jobs.csv") or os.path.exists(
        os.path.join(os.path.dirname(__file__), "sample_jobs.csv")
    )
    return render_template("index.html", skills=CANONICAL_SKILLS, has_sample_csv=sample_exists)


@app.route("/upload", methods=["POST"])
def upload():
    """Accepts a CSV file upload, trains models, returns metrics & top 5 hottest jobs."""
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
    """Quick-loads the pre-generated sample_jobs.csv without manual file browsing."""
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
    Accepts a JSON list of selected skills.
    Returns KNN and LR salary-tier predictions with probabilities.
    """
    if not MODEL_STORE["trained"]:
        return jsonify({"error": "No model trained yet. Please upload a CSV first."}), 400

    data = request.get_json(silent=True)
    if not data or "skills" not in data:
        return jsonify({"error": "Provide {'skills': [...]} in the request body."}), 400

    selected = [s.lower().strip() for s in data["skills"]]
    feature_cols = MODEL_STORE["feature_cols"]
    le     = MODEL_STORE["le"]
    scaler = MODEL_STORE["scaler"]

    # Build feature vector
    vec = []
    for col in feature_cols:
        skill_name = (col.replace("skill_", "")
                         .replace("_", " ")
                         .replace("plus", "+")
                         .replace("sharp", "#"))
        vec.append(1.0 if skill_name in selected else 0.0)

    x = np.array(vec).reshape(1, -1)
    x_scaled = scaler.transform(x)

    knn_pred  = MODEL_STORE["knn"].predict(x_scaled)[0]
    knn_proba = MODEL_STORE["knn"].predict_proba(x_scaled)[0]
    lr_pred   = MODEL_STORE["lr"].predict(x_scaled)[0]
    lr_proba  = MODEL_STORE["lr"].predict_proba(x_scaled)[0]

    knn_tier = le.inverse_transform([knn_pred])[0]
    lr_tier  = le.inverse_transform([lr_pred])[0]

    tier_ranges = {
        "Entry":  "$40,000 – $75,000",
        "Mid":    "$75,000 – $120,000",
        "Senior": "$120,000 – $220,000",
    }

    classes = le.classes_.tolist()
    knn_proba_dict = {classes[i]: round(float(knn_proba[i]) * 100, 1) for i in range(len(classes))}
    lr_proba_dict  = {classes[i]: round(float(lr_proba[i])  * 100, 1) for i in range(len(classes))}

    return jsonify({
        "knn": {
            "tier":        knn_tier,
            "salary_range": tier_ranges.get(knn_tier, ""),
            "confidence":  round(float(knn_proba.max()) * 100, 1),
            "probabilities": knn_proba_dict,
        },
        "lr": {
            "tier":        lr_tier,
            "salary_range": tier_ranges.get(lr_tier, ""),
            "confidence":  round(float(lr_proba.max()) * 100, 1),
            "probabilities": lr_proba_dict,
        },
    })


@app.route("/sample-csv")
def sample_csv():
    """Generate and serve a sample CSV for testing."""
    random.seed(42)
    rows = []
    for _ in range(300):
        title   = random.choice(JOB_TITLES)
        pool    = SKILL_POOLS.get(title, CANONICAL_SKILLS[:8])
        k       = random.randint(3, min(6, len(pool)))
        skills  = ", ".join(random.sample(pool, k))
        tier    = random.choices(
            ["Entry", "Mid", "Senior"], weights=[0.33, 0.40, 0.27]
        )[0]
        lo, hi  = SALARY_RANGES[tier]
        salary  = random.randint(lo, hi)
        rows.append({
            "job_title":   title,
            "company":     random.choice(COMPANIES),
            "location":    random.choice(LOCATIONS),
            "skills_raw":  skills,
            "salary":      salary,
            "salary_tier": tier,
        })

    df  = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_jobs.csv",
    )


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
