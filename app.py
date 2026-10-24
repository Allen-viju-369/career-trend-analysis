"""
Career Trend Analysis System
==============================
A single-file Streamlit application that:
  1. Scrapes real-time job listings with BeautifulSoup + requests
  2. Cleans & preprocesses text; extracts skills via Regex; one-hot encodes
  3. Produces EDA visualisations rendered with st.pyplot()
  4. Trains & compares RandomForestClassifier vs. Multinomial Naive Bayes
  5. Shows feature importances and a full model-comparison metrics table
  6. Provides a sidebar for user skill input -> live salary-tier prediction
"""

# ==============================================================
# Imports
# ==============================================================
import re
import time
import random
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ==============================================================
# Page Config & Global Styling
# ==============================================================
st.set_page_config(
    page_title="Career Trend Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG  = "#0e1117"
CARD_BG  = "#1a1f2e"
ACCENT   = "#6c63ff"
ACCENT2  = "#ff6584"
TEXT_MAIN= "#e0e0e0"
TEXT_SUB = "#9e9e9e"
BORDER   = "#2d3250"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_MAIN};
  }}

  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #12172a 0%, #1a1f2e 100%);
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
  }}

  .hero-banner {{
    background: linear-gradient(135deg, #1a1f2e 0%, #12172a 60%, #0e1117 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }}
  .hero-banner::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(108,99,255,.25), transparent 70%);
    border-radius: 50%;
  }}
  .hero-title {{
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }}
  .hero-sub {{
    color: {TEXT_SUB};
    font-size: .95rem;
    margin-top: .4rem;
  }}

  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }}
  .metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
  }}
  .metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(108,99,255,.15);
  }}
  .metric-card .accent-bar {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
  }}
  .metric-label {{ color: {TEXT_SUB}; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}
  .metric-value {{ color: {TEXT_MAIN}; font-size: 1.9rem; font-weight: 700; margin-top: .3rem; }}
  .metric-delta {{ font-size: .78rem; margin-top: .2rem; }}

  .section-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXT_MAIN};
    margin: 1.5rem 0 .8rem;
    display: flex;
    align-items: center;
    gap: .5rem;
  }}
  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BORDER};
  }}

  .prediction-box {{
    background: linear-gradient(135deg, #1a1f2e, #12172a);
    border: 1px solid {ACCENT};
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(108,99,255,.2);
    margin-top: 1rem;
  }}
  .prediction-tier {{
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .prediction-conf {{
    color: {TEXT_SUB};
    font-size: .85rem;
    margin-top: .3rem;
  }}

  .styled-table {{ width: 100%; border-collapse: collapse; }}
  .styled-table th {{
    background: {ACCENT};
    color: #fff;
    padding: .7rem 1rem;
    text-align: left;
    font-size: .85rem;
    font-weight: 600;
  }}
  .styled-table td {{
    background: {CARD_BG};
    color: {TEXT_MAIN};
    padding: .65rem 1rem;
    font-size: .88rem;
    border-bottom: 1px solid {BORDER};
  }}
  .styled-table tr:hover td {{ background: #222840; }}

  .stButton > button {{
    background: linear-gradient(90deg, {ACCENT}, #8b7cf6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: .55rem 1.4rem;
    font-weight: 600;
    width: 100%;
    transition: opacity .2s;
  }}
  .stButton > button:hover {{ opacity: .85; }}

  div[data-testid="stStatusWidget"] {{ display: none; }}
  .stSpinner > div {{ border-top-color: {ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# Constants
# ==============================================================
CANONICAL_SKILLS = [
    "python", "r", "sql", "java", "javascript", "typescript", "c++", "scala",
    "spark", "hadoop", "kafka", "airflow", "dbt",
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
    "power-bi": "powerbi",
    "node": "node.js",
    "nodejs": "node.js",
    "ml": "machine learning",
    "dl": "deep learning",
    "natural language processing": "nlp",
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
    "DevOps Engineer", "Software Engineer", "Business Intelligence Analyst",
]

COMPANIES = [
    "Google", "Amazon", "Microsoft", "Meta", "Apple", "Netflix",
    "Uber", "Stripe", "Databricks", "Snowflake", "OpenAI", "Palantir",
    "Airbnb", "Shopify", "Cloudflare", "Twilio", "HubSpot", "Salesforce",
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Remote", "Chicago, IL", "Los Angeles, CA",
    "Atlanta, GA", "Denver, CO",
]

SKILL_POOLS = {
    "Data Scientist":            ["python", "machine learning", "scikit-learn", "tensorflow", "sql", "statistics", "tableau", "docker"],
    "Machine Learning Engineer": ["python", "pytorch", "tensorflow", "docker", "kubernetes", "aws", "spark", "deep learning"],
    "Data Analyst":              ["sql", "python", "excel", "tableau", "powerbi", "statistics", "r", "data analysis"],
    "Backend Engineer":          ["python", "java", "postgresql", "redis", "docker", "aws", "flask", "fastapi"],
    "Full Stack Developer":      ["javascript", "typescript", "react", "node.js", "sql", "docker", "aws", "git"],
    "Data Engineer":             ["python", "spark", "kafka", "airflow", "sql", "aws", "dbt", "scala"],
    "MLOps Engineer":            ["python", "docker", "kubernetes", "airflow", "aws", "tensorflow", "git", "machine learning"],
    "AI Research Scientist":     ["python", "pytorch", "deep learning", "nlp", "statistics", "tensorflow", "r", "gcp"],
    "Cloud Engineer":            ["aws", "azure", "gcp", "kubernetes", "docker", "linux", "python", "git"],
    "DevOps Engineer":           ["docker", "kubernetes", "linux", "aws", "git", "python", "gcp", "azure"],
    "Software Engineer":         ["python", "java", "c++", "git", "sql", "docker", "linux", "javascript"],
    "Business Intelligence Analyst": ["sql", "tableau", "powerbi", "excel", "python", "statistics", "data analysis", "r"],
}


# ==============================================================
# 1. Web Scraping
# ==============================================================
@st.cache_data(show_spinner=False)
def scrape_jobs(n_synthetic: int = 500) -> pd.DataFrame:
    """
    Attempt to scrape job listings from RemoteOK public JSON API.
    Also tries a secondary HTML scrape from TimesJobs.
    Falls back gracefully to synthetic data when scraping is blocked.
    Returns DataFrame with columns:
        [job_title, company, location, skills_raw, salary, salary_tier]
    """
    records = []

    # --- Primary: RemoteOK JSON API ---
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
        resp = requests.get("https://remoteok.com/api", headers=headers, timeout=12)
        if resp.status_code == 200:
            jobs_json = resp.json()
            for job in jobs_json[1:]:   # element 0 is a legal notice dict
                title    = job.get("position", "")
                company  = job.get("company", "")
                location = job.get("location", "Remote") or "Remote"
                tags     = job.get("tags", [])
                skills_r = ", ".join(tags) if tags else ""
                salary_r = str(job.get("salary", "") or "")
                records.append({
                    "job_title":  title,
                    "company":    company,
                    "location":   location,
                    "skills_raw": skills_r,
                    "salary_raw": salary_r,
                })
                if len(records) >= 300:
                    break
    except Exception:
        pass

    # --- Secondary: TimesJobs HTML scrape ---
    if len(records) < 30:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            url = (
                "https://www.timesjobs.com/candidate/job-search.html"
                "?searchType=personalizedSearch&from=submit"
                "&txtKeywords=data+science&txtLocation="
            )
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            postings = soup.find_all("li", class_="clearfix job-bx wht-shd-bx")
            for post in postings[:200]:
                try:
                    title    = post.find("h2").text.strip()
                    company  = post.find("h3", class_="joblist-comp-name").text.strip()
                    skills_t = post.find("span", class_="srp-skills")
                    skills_r = skills_t.text.strip() if skills_t else ""
                    records.append({
                        "job_title":  title,
                        "company":    company,
                        "location":   "India",
                        "skills_raw": skills_r,
                        "salary_raw": "",
                    })
                except Exception:
                    continue
        except Exception:
            pass

    # --- Parse salary from raw text ---
    def _parse_salary(raw: str):
        nums = re.findall(r"\d[\d,]*", raw.replace("$", "").replace("k", "000"))
        nums = [int(n.replace(",", "")) for n in nums]
        return float(np.mean(nums)) if nums else None

    parsed = []
    for r in records:
        sal = _parse_salary(r.get("salary_raw", ""))
        if sal is None or sal < 1_000:
            tier = random.choices(
                ["Entry", "Mid", "Senior"], weights=[0.35, 0.40, 0.25]
            )[0]
            lo, hi = SALARY_RANGES[tier]
            sal = random.randint(lo, hi)
        elif sal <= 75_000:
            tier = "Entry"
        elif sal <= 120_000:
            tier = "Mid"
        else:
            tier = "Senior"
        r["salary"]      = sal
        r["salary_tier"] = tier
        parsed.append(r)

    df_live = pd.DataFrame(parsed) if parsed else pd.DataFrame()

    # --- Synthetic data (top-up to n_synthetic rows) ---
    random.seed(42)
    np.random.seed(42)
    synth = []
    for _ in range(max(0, n_synthetic - len(df_live))):
        title    = random.choice(JOB_TITLES)
        pool     = SKILL_POOLS.get(title, CANONICAL_SKILLS[:10])
        k        = random.randint(3, min(7, len(pool)))
        skills_r = ", ".join(random.sample(pool, k))
        tier     = random.choices(
            ["Entry", "Mid", "Senior"], weights=[0.33, 0.40, 0.27]
        )[0]
        lo, hi   = SALARY_RANGES[tier]
        salary   = random.randint(lo, hi)
        synth.append({
            "job_title":   title,
            "company":     random.choice(COMPANIES),
            "location":    random.choice(LOCATIONS),
            "skills_raw":  skills_r,
            "salary":      salary,
            "salary_tier": tier,
        })
    df_synth = pd.DataFrame(synth)

    df = pd.concat([df_live, df_synth], ignore_index=True)
    df = df[["job_title", "company", "location", "skills_raw", "salary", "salary_tier"]]
    return df


# ==============================================================
# 2. Data Preprocessing
# ==============================================================
def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, remove non-essential characters."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\+#\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> list:
    """Return sorted list of canonical skill names found in text."""
    text = clean_text(text)
    found = set()
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            found.add(canonical)
    for skill in CANONICAL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found.add(skill)
    return sorted(found)


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

    if s in {"en", "entry", "entry level", "entry-level", "junior", "jr", "intern", "internship", "fresher", "trainee", "associate", "0-2"}:
        return "Entry"
    if s in {"mi", "mid", "mid-level", "mid level", "intermediate", "experienced", "2-5"}:
        return "Mid"
    if s in {"se", "ex", "senior", "sr", "lead", "principal", "staff", "architect", "manager", "director", "head", "executive", "5+"}:
        return "Senior"

    if salary_num is not None and pd.notnull(salary_num):
        try:
            num = float(salary_num)
            if num > 0:
                if num < 100:
                    num = num * 12000
                if num < 75000:
                    return "Entry"
                elif num <= 120000:
                    return "Mid"
                else:
                    return "Senior"
        except (ValueError, TypeError):
            pass

    t = str(title_str).lower()
    if re.search(r"\b(senior|sr|lead|principal|staff|architect|director|head|vp|manager|chief)\b", t):
        return "Senior"
    if re.search(r"\b(junior|jr|intern|internship|entry|trainee|fresher|graduate|associate|assistant)\b", t):
        return "Entry"

    return "Mid"


def ensure_balanced_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataset has all 3 tiers so ML models can train smoothly."""
    counts = df["salary_tier"].value_counts()
    if len(counts) >= 3 and counts.min() >= 2:
        return df

    if "salary" in df.columns:
        s = pd.to_numeric(df["salary"], errors="coerce").dropna()
        if len(s.unique()) >= 3:
            try:
                df["salary_tier"] = pd.qcut(s, q=3, labels=["Entry", "Mid", "Senior"], duplicates="drop")
                if len(df["salary_tier"].value_counts()) >= 3 and df["salary_tier"].value_counts().min() >= 2:
                    return df
            except Exception:
                pass

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


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text fields, extract skills via Regex, one-hot encode skill columns.
    Universally supports arbitrary column names and schemas from Kaggle/LinkedIn.
    """
    df = df_raw.copy()

    # 1. Map columns using fuzzy/synonym matching
    col_mapping = {}
    for col in df.columns:
        std_name = map_column_name(col)
        if std_name not in col_mapping.values():
            col_mapping[col] = std_name
    df = df.rename(columns=col_mapping)

    if "job_title" not in df.columns:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        df["job_title"] = df[text_cols[0]] if text_cols else "Software Engineer"

    if "company" not in df.columns:
        df["company"] = "Tech Enterprise"

    if "location" not in df.columns:
        df["location"] = "Remote"

    if "skills_raw" not in df.columns:
        desc_cols = [c for c in df.columns if any(w in str(c).lower() for w in ["desc", "detail", "text", "body", "summary"])]
        if desc_cols:
            df["skills_raw"] = df[desc_cols[0]].fillna("")
        else:
            df["skills_raw"] = df["job_title"].fillna("")

    if "salary" in df.columns:
        df["salary_clean"] = df["salary"].apply(parse_numeric_salary)
    else:
        df["salary_clean"] = None

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

    df = ensure_balanced_tiers(df)

    df["job_title"]  = df["job_title"].apply(clean_text)
    df["company"]    = df["company"].apply(clean_text)
    df["location"]   = df["location"].apply(clean_text)
    df["skills_raw"] = df["skills_raw"].apply(clean_text)
    df["skills_list"]= df["skills_raw"].apply(extract_skills)

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

    for skill in CANONICAL_SKILLS:
        safe = (skill.replace(" ", "_")
                     .replace(".", "_")
                     .replace("+", "plus")
                     .replace("#", "sharp")
                     .replace("-", "_"))
        col = "skill_" + safe
        df[col] = df["skills_list"].apply(lambda lst, s=skill: 1 if s in lst else 0)

    df = df.dropna(subset=["salary_tier"]).reset_index(drop=True)
    return df


# ==============================================================
# Helper – dark-themed matplotlib figure
# ==============================================================
def _dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT_SUB, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SUB)
    ax.yaxis.label.set_color(TEXT_SUB)
    ax.title.set_color(TEXT_MAIN)
    return fig, ax


# ==============================================================
# 3. Exploratory Data Analysis – Plot Functions
# ==============================================================
def plot_top_skills(df: pd.DataFrame, top_n: int = 15):
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    counts = df[skill_cols].sum().sort_values(ascending=False).head(top_n)
    labels = [c.replace("skill_", "").replace("_", " ") for c in counts.index]
    palette = sns.color_palette("coolwarm", len(labels))

    fig, ax = _dark_fig(10, 5)
    bars = ax.barh(labels[::-1], counts.values[::-1],
                   color=palette[::-1], edgecolor="none", height=0.65)
    for bar in bars:
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center", color=TEXT_SUB, fontsize=8,
        )
    ax.set_xlabel("Job Count", color=TEXT_SUB)
    ax.set_title(f"Top {top_n} Most In-Demand Skills",
                 color=TEXT_MAIN, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_salary_distribution(df: pd.DataFrame):
    palette_map = {"Entry": "#6c63ff", "Mid": "#ff6584", "Senior": "#43e97b"}
    fig, ax = _dark_fig(10, 4)
    for tier, grp in df.groupby("salary_tier"):
        ax.hist(grp["salary"], bins=30, alpha=0.72,
                label=tier, color=palette_map.get(tier, "#aaa"), edgecolor="none")
    ax.set_xlabel("Annual Salary (USD)", color=TEXT_SUB)
    ax.set_ylabel("Frequency", color=TEXT_SUB)
    ax.set_title("Salary Distribution by Tier",
                 color=TEXT_MAIN, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k")
    )
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT_MAIN, framealpha=0.6, fontsize=9)
    ax.grid(axis="y", color=BORDER, alpha=0.5, linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_tier_pie(df: pd.DataFrame):
    counts = df["salary_tier"].value_counts()
    colors = ["#6c63ff", "#ff6584", "#43e97b"]
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=CARD_BG)
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor=CARD_BG, linewidth=2),
    )
    for t in texts:
        t.set_color(TEXT_MAIN)
        t.set_fontsize(10)
    for a in autotexts:
        a.set_color("#fff")
        a.set_fontsize(9)
        a.set_fontweight("bold")
    ax.set_title("Salary Tier Distribution",
                 color=TEXT_MAIN, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def plot_skills_by_tier(df: pd.DataFrame, top_n: int = 10):
    skill_cols  = [c for c in df.columns if c.startswith("skill_")]
    tier_order  = ["Entry", "Mid", "Senior"]
    tier_colors = ["#6c63ff", "#ff6584", "#43e97b"]
    top_skills  = df[skill_cols].sum().sort_values(ascending=False).head(top_n).index.tolist()
    labels      = [c.replace("skill_", "").replace("_", " ") for c in top_skills]

    tier_data = {}
    for tier in tier_order:
        sub = df[df["salary_tier"] == tier]
        tier_data[tier] = sub[top_skills].sum().values

    fig, ax = _dark_fig(11, 5)
    x = np.arange(len(labels))
    bar_w = 0.26
    for i, (tier, color) in enumerate(zip(tier_order, tier_colors)):
        ax.bar(x + i * bar_w, tier_data[tier], bar_w,
               label=tier, color=color, alpha=0.85, edgecolor="none")
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Count", color=TEXT_SUB)
    ax.set_title(f"Top {top_n} Skills by Salary Tier",
                 color=TEXT_MAIN, fontsize=12, fontweight="bold", pad=10)
    ax.legend(facecolor=CARD_BG, labelcolor=TEXT_MAIN, framealpha=0.6, fontsize=9)
    ax.grid(axis="y", color=BORDER, alpha=0.5, linewidth=0.6)
    fig.tight_layout()
    return fig


# ==============================================================
# 4 & 5. Machine Learning – Train & Compare Models
# ==============================================================
def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("skill_")]


def train_models(df: pd.DataFrame):
    """
    Train RandomForestClassifier and MultinomialNB on one-hot skill features.
    Returns: rf_model, nb_model, X_test, y_test, feature_cols, label_encoder, metrics_df
    """
    feature_cols = get_feature_cols(df)
    le = LabelEncoder()
    y  = le.fit_transform(df["salary_tier"])
    X  = df[feature_cols].values

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
        X, y, test_size=test_size, random_state=42,
        stratify=y if can_stratify else None
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Naive Bayes (Multinomial) – works with non-negative integer/count features
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    def _metrics(y_true, y_pred, name):
        return {
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "F1-Score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }

    metrics_df = pd.DataFrame([
        _metrics(y_test, y_pred_rf, "Random Forest"),
        _metrics(y_test, y_pred_nb, "Naive Bayes"),
    ])

    return rf, nb, X_test, y_test, feature_cols, le, metrics_df


def plot_feature_importances(rf_model, feature_cols: list, top_n: int = 15):
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(top_n)
    labels = [c.replace("skill_", "").replace("_", " ") for c in importances.index]
    gradient = sns.color_palette("plasma", top_n)

    fig, ax = _dark_fig(10, 5)
    ax.barh(labels[::-1], importances.values[::-1],
            color=gradient[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Feature Importance", color=TEXT_SUB)
    ax.set_title(f"Top {top_n} Skills Driving Salary (RF Importance)",
                 color=TEXT_MAIN, fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", color=BORDER, alpha=0.5, linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(rf_model, X_test, y_test, le):
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor=BORDER,
        ax=ax, cbar=False,
        annot_kws={"size": 12, "weight": "bold", "color": TEXT_MAIN},
    )
    ax.set_xlabel("Predicted", color=TEXT_SUB)
    ax.set_ylabel("Actual", color=TEXT_SUB)
    ax.set_title("RF Confusion Matrix",
                 color=TEXT_MAIN, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_SUB, labelsize=9)
    fig.tight_layout()
    return fig


# ==============================================================
# Utility – build skill vector for user input
# ==============================================================
def build_user_vector(selected_skills: list, feature_cols: list) -> "np.ndarray":
    vec = []
    for col in feature_cols:
        skill_name = (col.replace("skill_", "")
                         .replace("_", " ")
                         .replace("plus", "+")
                         .replace("sharp", "#"))
        vec.append(1 if skill_name in selected_skills else 0)
    return np.array(vec).reshape(1, -1)


# ==============================================================
# UI Helpers
# ==============================================================
def render_metric_cards(df: pd.DataFrame, metrics_df: pd.DataFrame):
    rf_row = metrics_df[metrics_df["Model"] == "Random Forest"].iloc[0]
    nb_row = metrics_df[metrics_df["Model"] == "Naive Bayes"].iloc[0]
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="accent-bar" style="background:linear-gradient(90deg,#6c63ff,#a78bfa);"></div>
        <div class="metric-label">Total Job Listings</div>
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-delta" style="color:#43e97b;">📋 Scraped + Synthetic</div>
      </div>
      <div class="metric-card">
        <div class="accent-bar" style="background:linear-gradient(90deg,#ff6584,#f59e0b);"></div>
        <div class="metric-label">Skills Tracked</div>
        <div class="metric-value">{len(CANONICAL_SKILLS)}</div>
        <div class="metric-delta" style="color:#43e97b;">🛠 Canonical Skills</div>
      </div>
      <div class="metric-card">
        <div class="accent-bar" style="background:linear-gradient(90deg,#43e97b,#38f9d7);"></div>
        <div class="metric-label">RF Accuracy</div>
        <div class="metric-value">{rf_row['Accuracy']:.1%}</div>
        <div class="metric-delta" style="color:#6c63ff;">🌲 Random Forest</div>
      </div>
      <div class="metric-card">
        <div class="accent-bar" style="background:linear-gradient(90deg,#f59e0b,#ef4444);"></div>
        <div class="metric-label">NB Accuracy</div>
        <div class="metric-value">{nb_row['Accuracy']:.1%}</div>
        <div class="metric-delta" style="color:#ff6584;">📊 Naive Bayes</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def metrics_table_html(metrics_df: pd.DataFrame) -> str:
    colors = {"Random Forest": "#6c63ff", "Naive Bayes": "#ff6584"}
    rows = ""
    for _, row in metrics_df.iterrows():
        c = colors.get(row["Model"], "#aaa")
        rows += (
            f"<tr>"
            f"<td><span style='color:{c};font-weight:700;'>{row['Model']}</span></td>"
            f"<td>{row['Accuracy']:.4f}</td>"
            f"<td>{row['Precision']:.4f}</td>"
            f"<td>{row['Recall']:.4f}</td>"
            f"<td>{row['F1-Score']:.4f}</td>"
            f"</tr>"
        )
    return f"""
    <table class="styled-table">
      <thead>
        <tr>
          <th>Model</th><th>Accuracy</th><th>Precision</th>
          <th>Recall</th><th>F1-Score</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def get_hottest_jobs_summary(df: pd.DataFrame, top_n: int = 5) -> list:
    total_jobs = len(df)
    title_col = "job_title"
    feature_cols = [c for c in df.columns if c.startswith("skill_")]
    has_salary = "salary" in df.columns and pd.to_numeric(df["salary"], errors="coerce").notnull().any()

    tier_est = {"Entry": 62000, "Mid": 98000, "Senior": 165000}
    grouped = df.groupby(title_col)
    ranked = grouped.size().sort_values(ascending=False).head(top_n)
    max_count = ranked.iloc[0] if len(ranked) > 0 else 1

    results = []
    for rank, (title, count) in enumerate(ranked.items(), 1):
        sub = df[df[title_col] == title]
        demand_pct = round((count / total_jobs) * 100, 1)
        if has_salary:
            numeric_s = pd.to_numeric(sub["salary"], errors="coerce")
            avg_sal = int(numeric_s.mean()) if not numeric_s.dropna().empty else 95000
        else:
            avg_sal = int(sub["salary_tier"].map(tier_est).mean())

        top_skills = []
        if feature_cols:
            sums = sub[feature_cols].sum().sort_values(ascending=False).head(3)
            top_skills = [
                s.replace("skill_", "").replace("_", " ").replace("plus", "+").replace("sharp", "#")
                for s in sums.index if sums[s] > 0
            ]

        primary_tier = sub["salary_tier"].mode().iloc[0] if not sub["salary_tier"].empty else "Mid"
        demand_score = (count / max_count) * 60.0
        salary_factor = min(1.0, max(0.0, (avg_sal - 45000) / 135000)) * 40.0
        hotness_score = round(demand_score + salary_factor, 1)

        results.append({
            "rank": rank,
            "title": str(title).title(),
            "count": int(count),
            "demand_pct": demand_pct,
            "avg_salary": f"${avg_sal:,.0f}",
            "primary_tier": primary_tier,
            "top_skills": ", ".join(top_skills),
            "hotness_score": hotness_score,
        })
    return results


def render_hottest_jobs(df: pd.DataFrame):
    hottest = get_hottest_jobs_summary(df, top_n=5)
    st.markdown(
        '<div class="section-title">🔥 Top 5 Hottest Jobs in Market</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, job in enumerate(hottest):
        with cols[i]:
            st.markdown(f"""
            <div style="background:#1a1f2e;border:1px solid #2d3250;border-radius:12px;padding:1rem;height:100%;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 4px 15px rgba(0,0,0,.25);">
              <div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem;">
                  <span style="font-size:1.05rem;font-weight:700;color:#c7d2fe;">{medals[i]} #{job['rank']}</span>
                  <span style="background:rgba(245,158,11,.15);color:#fbbf24;font-size:.72rem;font-weight:700;padding:.2rem .45rem;border-radius:6px;border:1px solid rgba(245,158,11,.3);">🔥 {job['hotness_score']}</span>
                </div>
                <div style="font-size:.92rem;font-weight:700;color:#e0e0e0;margin-bottom:.3rem;line-height:1.3;">{job['title']}</div>
                <div style="font-size:.82rem;color:#43e97b;font-weight:700;">{job['avg_salary']} <span style="color:#9e9e9e;font-size:.7rem;font-weight:400;">avg</span></div>
                <div style="font-size:.72rem;color:#9e9e9e;margin-top:.2rem;">{job['count']} postings ({job['demand_pct']}%)</div>
              </div>
              <div style="margin-top:.8rem;padding-top:.5rem;border-top:1px solid rgba(255,255,255,.07);font-size:.68rem;color:#c7d2fe;line-height:1.4;">
                <b style="color:#9e9e9e;">Key Skills:</b><br>{job['top_skills']}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================
# 6. Main Streamlit App
# ==============================================================
def main():
    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;margin-bottom:1.2rem;">
          <div style="font-size:2.2rem;">🚀</div>
          <div style="font-size:1.1rem;font-weight:700;color:#e0e0e0;">Salary Predictor</div>
          <div style="font-size:.78rem;color:#9e9e9e;margin-top:.2rem;">
            Select your skills to predict your salary tier
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🛠 Your Technical Skills")
        selected_skills = st.multiselect(
            "Choose skills you know:",
            options=sorted(CANONICAL_SKILLS),
            default=["python", "machine learning", "sql"],
            help="Select all skills you are comfortable working with.",
        )

        st.markdown("---")
        st.markdown("#### 📂 Custom Dataset (Optional)")
        uploaded_csv = st.file_uploader(
            "Upload any CSV dataset:",
            type=["csv"],
            help="Upload your own Kaggle, LinkedIn, or custom tech job dataset.",
        )

        st.markdown("---")
        st.markdown("#### ⚙️ App Settings")
        top_n_skills = st.slider("Skills to display in charts", 8, 20, 15)
        show_raw     = st.toggle("Show raw DataFrame", value=False)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict My Salary Tier")

        st.markdown("""
        <div style="margin-top:1.5rem;padding:.8rem;background:#12172a;border-radius:10px;
                    border:1px solid #2d3250;font-size:.75rem;color:#9e9e9e;line-height:1.6;">
          <b style="color:#e0e0e0;">ℹ️ Universal Compatibility</b><br>
          Accepts any CSV dataset! Automatically detects Title, Skills, Description, Salary, and Experience.
        </div>
        """, unsafe_allow_html=True)

    # ── Hero Banner ───────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
      <h1 class="hero-title">📊 Career Trend Analysis System</h1>
      <p class="hero-sub">
        Real-time job market intelligence · ML-powered salary prediction · Skill demand analytics
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load & Preprocess Data ────────────────────────────────
    with st.spinner("⏳ Loading and preprocessing job data…"):
        if uploaded_csv is not None:
            df_raw = pd.read_csv(uploaded_csv)
            st.sidebar.success(f"✅ Loaded: {len(df_raw)} rows")
        else:
            df_raw = scrape_jobs(n_synthetic=500)
        df = preprocess(df_raw)

    # ── Train Models ──────────────────────────────────────────
    with st.spinner("🤖 Training ML models…"):
        rf_model, nb_model, X_test, y_test, feature_cols, le, metrics_df = train_models(df)

    # ── Metric Cards ──────────────────────────────────────────
    render_metric_cards(df, metrics_df)

    # ── Prediction (triggered by button or any skill selection) ──
    if predict_btn or selected_skills:
        user_vec = build_user_vector(selected_skills, feature_cols)
        rf_pred  = le.inverse_transform(rf_model.predict(user_vec))[0]
        nb_pred  = le.inverse_transform(nb_model.predict(user_vec))[0]
        rf_prob  = rf_model.predict_proba(user_vec).max() * 100

        tier_emojis = {"Entry": "🟢", "Mid": "🟡", "Senior": "🔴"}
        tier_desc   = {
            "Entry":  "$40k – $75k / year",
            "Mid":    "$75k – $120k / year",
            "Senior": "$120k – $220k / year",
        }

        with st.sidebar:
            st.markdown(f"""
            <div class="prediction-box">
              <div style="font-size:.8rem;color:#9e9e9e;text-transform:uppercase;letter-spacing:.08em;">
                RF Prediction
              </div>
              <div class="prediction-tier">
                {tier_emojis.get(rf_pred, "")} {rf_pred}
              </div>
              <div class="prediction-conf">{tier_desc.get(rf_pred, "")}</div>
              <div style="margin-top:.6rem;padding:.4rem;background:#12172a;
                          border-radius:8px;font-size:.78rem;color:#9e9e9e;">
                Confidence: <b style="color:#6c63ff;">{rf_prob:.1f}%</b>
              </div>
              <div style="margin-top:.6rem;font-size:.75rem;color:#9e9e9e;">
                Naive Bayes also predicts:
                <b style="color:#ff6584;">{nb_pred}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Top 5 Hottest Jobs ────────────────────────────────────
    render_hottest_jobs(df)

    # ── EDA Section ───────────────────────────────────────────
    st.markdown(
        '<div class="section-title">📈 Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Most In-Demand Skills**")
        fig = plot_top_skills(df, top_n=top_n_skills)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        st.markdown("**Salary Tier Breakdown**")
        fig = plot_tier_pie(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**Salary Distribution by Tier**")
    fig = plot_salary_distribution(df)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**Skills by Salary Tier (Grouped Bar)**")
    fig = plot_skills_by_tier(df, top_n=top_n_skills)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── ML Results Section ────────────────────────────────────
    st.markdown(
        '<div class="section-title">🤖 Machine Learning Results</div>',
        unsafe_allow_html=True,
    )

    col3, col4 = st.columns([3, 2])
    with col3:
        st.markdown("**Model Comparison**")
        st.markdown(metrics_table_html(metrics_df), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Full Classification Report (Random Forest)"):
            y_pred_rf = rf_model.predict(X_test)
            report = classification_report(
                y_test, y_pred_rf,
                target_names=le.classes_,
                output_dict=False,
            )
            st.code(report, language="text")
    with col4:
        st.markdown("**RF Confusion Matrix**")
        fig = plot_confusion_matrix(rf_model, X_test, y_test, le)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**Feature Importances – Which Skills Drive Salary?**")
    fig = plot_feature_importances(rf_model, feature_cols, top_n=top_n_skills)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Raw Data ──────────────────────────────────────────────
    if show_raw:
        st.markdown(
            '<div class="section-title">🗃 Raw Data Preview</div>',
            unsafe_allow_html=True,
        )
        display_cols = ["job_title", "company", "location", "skills_raw", "salary", "salary_tier"]
        st.dataframe(
            df[display_cols].head(200),
            use_container_width=True,
            height=380,
        )

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:3rem;padding:1rem;border-top:1px solid #2d3250;
                text-align:center;font-size:.78rem;color:#4a5080;">
      Career Trend Analysis System &nbsp;·&nbsp;
      Built with Streamlit, Scikit-learn, BeautifulSoup &amp; Seaborn &nbsp;·&nbsp;
      Data: RemoteOK API + Synthetic
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
