# 📊 Career Trend Analysis & Salary Intelligence System

An end-to-end Machine Learning and Data Analytics system designed to decode tech industry hiring trends, extract in-demand technical competencies from real job listings, uncover natural talent clusters, compare machine learning models, rank the **Top 5 Hottest Jobs**, and predict salary tiers based on an individual's technical skill profile.

The repository includes **two production-ready implementations**:
1. 🎈 **Streamlit Dashboard (`app.py`)** — Interactive data science application featuring web scraping, Matplotlib/Seaborn visualization suite, Random Forest vs. Multinomial Naive Bayes comparison, and live feature importance analysis.
2. ⚡ **Flask Full-Stack Web App (`flask_app/`)** — Production Flask application with a custom dark-mode frontend (Vanilla HTML/CSS/JS), CSV dataset uploader with quick-load capability, K-Means clustering, KNN vs. Logistic Regression model comparison, and interactive probability breakdown.

---

## 🌟 Key Features Developed

| Feature | Implementation Details |
|---|---|
| **🌐 Web Scraping & Ingestion** | Live scraping from RemoteOK REST API and TimesJobs via `BeautifulSoup` & `requests`, with an automatic realistic synthetic data generator fallback. |
| **🧹 Data Preprocessing & NLP** | Regex-based token cleaning, tech alias normalization (e.g., `k8s` → `kubernetes`, `sklearn` → `scikit-learn`), and binary One-Hot Encoding across 41 canonical skills. |
| **🔥 Top 5 Hottest Jobs Ranking** | Dynamic calculation of the 5 hottest roles in the market based on hiring volume, market share, average salary, and hotness index (0–100 score). |
| **📈 Exploratory Data Analysis** | Rich visualizations for top demanded skills, salary distributions by tier, donut charts, and grouped multi-tier skill comparisons. |
| **🔵 Unsupervised Clustering** | K-Means clustering ($k=3$) to partition job listings into distinct market profiles based on high-dimensional skill vectors. |
| **🤖 Supervised ML Comparisons** | **Random Forest vs. Naive Bayes** (Streamlit) and **KNN vs. Logistic Regression** (Flask), evaluated on Accuracy, Precision, Recall, and Weighted F1-score. |
| **🔮 Live Salary Tier Predictor** | Interactive skill selection yielding instant salary tier predictions (`Entry`, `Mid`, `Senior`), confidence levels, and compensation ranges. |

---

## 🏗️ System Architecture & Directory Structure

```
career_trend_analysis/
│
├── README.md                      ← Comprehensive documentation (this file)
├── app.py                         ← Complete Streamlit Web Application (Port 8501)
│
└── flask_app/                     ← Complete Flask Web Application (Port 5000)
    ├── app.py                     ← Flask backend & ML pipeline routes
    ├── sample_jobs.csv            ← 500-row realistic job market dataset
    ├── static/
    │   └── style.css              ← Custom dark-mode CSS (animations, cards, glassmorphism)
    └── templates/
        └── index.html             ← Semantic HTML5 frontend with Vanilla JS engine
```

---

## ⚙️ How It Works (End-to-End Workflow)

### 1. Data Collection & Fallback Strategy
- The scraper attempts to query remote tech listings from public endpoints.
- If network policies, rate limits, or firewalls block external scraping requests, the system falls back to a deterministic synthetic generator that produces realistic job postings with authentic distributions across titles, companies, locations, skill combinations, and salary ranges.

### 2. NLP & Regex Technical Skill Extraction
Every job description or raw skill string is cleaned and scanned using regular expression word boundaries:
$$\text{Pattern} = \text{re.compile}(r'\b' + \text{skill} + r'\b', \text{re.IGNORECASE})$$
- Aliases are resolved first (e.g., `postgres` $\to$ `postgresql`, `ml` $\to$ `machine learning`).
- 41 canonical skills across languages, cloud infrastructure, AI/ML libraries, databases, and DevOps are mapped into a binary one-hot encoded matrix:
$$X_{ij} = \begin{cases} 1 & \text{if candidate } i \text{ possesses skill } j \\ 0 & \text{otherwise} \end{cases}$$

### 3. Top 5 Hottest Jobs Intelligence Engine
The system groups listings by job title and computes:
- **Posting Frequency & Market Share**: Percentage of total market listings for that role.
- **Average Compensation & Range**: Mean, minimum, and maximum salary for the title.
- **Top Demanded Skills**: The most frequent technical requirements specific to that role.
- **Hotness Index ($0 - 100$)**:
$$\text{Hotness} = 60 \times \left(\frac{\text{Count}}{\text{Max Count}}\right) + 40 \times \min\left(1, \max\left(0, \frac{\text{Avg Salary} - \$45,000}{\$135,000}\right)\right)$$
This blends current hiring velocity with compensation strength to identify which jobs offer the highest career leverage.

### 4. Unsupervised K-Means Clustering ($k=3$)
To detect natural skill clusters in the job market without labeled supervision, the one-hot skill vectors are clustered using Euclidean distance:
$$\arg\min_S \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$
This segments listings into natural industry groupings:
- **Cluster 1**: Data Science, AI & Machine Learning (`python`, `pytorch`, `sql`, `statistics`)
- **Cluster 2**: Cloud, DevOps & Infrastructure (`docker`, `kubernetes`, `aws`, `linux`)
- **Cluster 3**: Full Stack & Enterprise Engineering (`javascript`, `react`, `java`, `node.js`)

### 5. Supervised Machine Learning & Model Comparison

#### Implementation A (Streamlit): Random Forest vs. Multinomial Naive Bayes
- **Random Forest Classifier**: Ensemble of 200 decision trees trained with bootstrap aggregation and Gini impurity. Handles non-linear skill interactions and outputs feature importances.
- **Multinomial Naive Bayes**: Probabilistic classifier applying Bayes' Theorem under feature conditional independence assumptions:
$$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y)$$

#### Implementation B (Flask): K-Nearest Neighbors (KNN) vs. Logistic Regression
- **KNN Classifier ($k=5$)**: Measures Euclidean distance in standardized feature space to classify salary tier based on the majority vote of the nearest 5 job profiles.
- **Multinomial Logistic Regression**: Uses a softmax link function with balanced class weights to compute calibrated posterior probabilities for each salary tier.

#### Evaluation Metrics Computed:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 6. Live Salary Tier Prediction
When a user selects technical skills:
1. A 41-dimensional binary vector $\mathbf{x}_{\text{user}}$ is constructed.
2. The vector is transformed using the fitted `StandardScaler`.
3. The models compute predictions and class probabilities for `Entry` (\$40k–\$75k), `Mid` (\$75k–\$120k), and `Senior` (\$120k–\$220k).

---

## 📊 Dataset Specification (`sample_jobs.csv`)

The system accepts any CSV file with the following standard columns:

The system includes an **Intelligent Schema Normalizer** with **Universal Dataset Compatibility**. You can upload **any CSV dataset from Kaggle, LinkedIn, Indeed, Glassdoor, or custom scrapers**, and the system will automatically detect, map, and clean the data.

### Supported Flexible Column Mappings:

| Canonical Field | Recognized Aliases (Case-Insensitive) | Fallback if Missing |
|---|---|---|
| **Job Title** | `job_title`, `title`, `position`, `role`, `jobrole`, `designation`, `jobpost` | Auto-detected from any text column or `"Software Engineer"` |
| **Technical Skills** | `skills_raw`, `skills`, `skill`, `required_skills`, `key_skills`, `description`, `job_description`, `snippet`, `summary` | Regex-extracted from `job_title` or description |
| **Salary Tier / Exp** | `salary_tier`, `tier`, `experience_level`, `level`, `seniority`, `career_level`, `experience` (e.g. `EN`, `MI`, `SE`, `Junior`, `Senior`) | Auto-binned from `salary` or inferred from `job_title` keywords |
| **Salary Amount** | `salary`, `salary_in_usd`, `annual_salary`, `compensation`, `pay`, `ctc`, `remuneration` (e.g. `120000`, `$140k`, `80k-100k`) | Auto-synthesized based on role and tier |
| **Company** | `company`, `company_name`, `employer`, `hiring_company`, `organization`, `firm` | `"Tech Enterprise"` |
| **Location** | `location`, `job_location`, `city`, `work_location`, `country`, `state` | `"Remote"` |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` package manager

### 1. Install Dependencies
```bash
pip install streamlit flask scikit-learn pandas numpy matplotlib seaborn beautifulsoup4 requests
```

---

### 2. Option A: Run the Flask Web Application (Recommended)
```bash
# Navigate to flask_app directory
cd flask_app

# Start the Flask development server
python app.py
```
Open your browser at:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

**How to use the Flask App:**
1. Click **🚀 Quick Load Sample Data** (loads and trains instantly on `sample_jobs.csv`), or drag-and-drop your custom CSV.
2. View the **🔥 Top 5 Hottest Jobs in Market** cards with salary ranges, hiring demand, and skill requirements.
3. Review the **K-Means Clusters** and **Model Comparison Table** (KNN vs. Logistic Regression).
4. Scroll down to **Step 2 — Predict Your Salary Tier**, tick your skills, and click **Predict My Salary Tier** to see your confidence and compensation estimate!

---

### 3. Option B: Run the Streamlit Application
```bash
# From workspace root
streamlit run app.py --server.port 8501
```
Open your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

**Features in Streamlit:**
- Top 5 Hottest Jobs cards in demand
- Interactive skill selection in the sidebar with live instant prediction
- Feature importances bar chart showing which skills drive salary the most
- Confusion matrix and full classification report

---

## 📡 Flask REST API Reference

| Endpoint | Method | Payload / Params | Response | Description |
|---|---|---|---|---|
| `/` | `GET` | None | HTML | Renders the main dashboard. |
| `/upload` | `POST` | `multipart/form-data` (`csv_file`) | JSON | Preprocesses dataset, trains K-Means, KNN & Logistic Regression, returns metrics & top 5 hottest jobs. |
| `/load-sample` | `POST` / `GET` | None | JSON | Instantly trains on `sample_jobs.csv` without file browsing. |
| `/predict` | `POST` | `{"skills": ["python", "aws", ...]}` | JSON | Computes salary tier and probabilities for KNN & LR. |
| `/sample-csv` | `GET` | None | CSV File | Generates and downloads a fresh 300-row sample dataset. |

---

## 🏆 Summary of Technologies Used

- **Backend & Serving**: Python 3.12, Flask, Streamlit
- **Machine Learning**: Scikit-Learn (`KMeans`, `KNeighborsClassifier`, `LogisticRegression`, `RandomForestClassifier`, `MultinomialNB`, `StandardScaler`, `LabelEncoder`)
- **Data Manipulation**: Pandas, NumPy
- **Data Scraping**: BeautifulSoup4, Requests, Regex (`re`)
- **Visualizations**: Matplotlib, Seaborn
- **Frontend & Styling**: Semantic HTML5, Vanilla JavaScript (Fetch API), Vanilla CSS (CSS Variables, Grid & Flexbox layouts, Glassmorphism, Responsive Design)
