"""
generate_maharashtra_data.py
============================
Generates mock labor-market job postings for Maharashtra Skill Align Platform (MSINS ID26134).
Introduces 15% synthetic noise into 'Curriculum Status' to prevent 100% overfitting.
"""

import os
import random
import pandas as pd

JOB_PROFILES = [
    # High Demand
    {"title": "Cloud & DevOps Infrastructure Engineer", "skills": "aws, docker, kubernetes, linux, terraform, ci/cd", "status": "High Demand", "volume_range": (150, 350)},
    {"title": "Big Data & Pipeline Engineer", "skills": "airflow, aws, kafka, python, spark, sql", "status": "High Demand", "volume_range": (140, 320)},
    {"title": "MLOps Systems Specialist", "skills": "airflow, aws, ci/cd, docker, git, kubernetes, machine learning, python", "status": "High Demand", "volume_range": (120, 280)},
    {"title": "Full Stack React & Node Developer", "skills": "docker, fastapi, git, javascript, mongodb, node.js, react, typescript", "status": "High Demand", "volume_range": (160, 380)},
    {"title": "AI & Deep Learning Researcher", "skills": "deep learning, machine learning, nlp, python, pytorch, tensorflow", "status": "High Demand", "volume_range": (90, 220)},
    {"title": "Cloud Security & DevSecOps Engineer", "skills": "aws, azure, cybersecurity, docker, kubernetes, linux", "status": "High Demand", "volume_range": (100, 260)},
    {"title": "Microservices Backend Architect", "skills": "docker, git, java, kubernetes, microservices, redis, sql", "status": "High Demand", "volume_range": (130, 290)},

    # Needs Update
    {"title": "Backend Java Developer", "skills": "git, java, postgresql, sql", "status": "Needs Update", "volume_range": (80, 240)},
    {"title": "Business Data Analyst", "skills": "data analysis, excel, powerbi, sql, statistics, tableau", "status": "Needs Update", "volume_range": (100, 250)},
    {"title": "BI Reporting Specialist", "skills": "excel, powerbi, sql, statistics, tableau", "status": "Needs Update", "volume_range": (70, 200)},
    {"title": "Junior Web Developer", "skills": "git, javascript, react, sql", "status": "Needs Update", "volume_range": (90, 210)},
    {"title": "Python Scripting Assistant", "skills": "data analysis, excel, python, sql", "status": "Needs Update", "volume_range": (60, 180)},
    {"title": "Statistical Associate", "skills": "excel, r, statistics, sql", "status": "Needs Update", "volume_range": (50, 160)},
    {"title": "Database Support Executive", "skills": "mysql, postgresql, sql", "status": "Needs Update", "volume_range": (70, 190)},

    # Obsolete / Oversupplied
    {"title": "Legacy Desktop Programmer", "skills": "c++, linux", "status": "Obsolete/Oversupplied", "volume_range": (15, 65)},
    {"title": "Static Web Maintenance Technician", "skills": "html, css, basic javascript", "status": "Obsolete/Oversupplied", "volume_range": (20, 75)},
    {"title": "Legacy Mainframe Operator", "skills": "cobol, sql", "status": "Obsolete/Oversupplied", "volume_range": (10, 45)},
    {"title": "Standalone Visual Basic Developer", "skills": "visual basic, access, sql", "status": "Obsolete/Oversupplied", "volume_range": (10, 50)},
]

DISTRICTS = [
    "Pune", "Mumbai", "Nagpur", "Nashik", "Thane",
    "Chhatrapati Sambhajinagar", "Kolhapur", "Solapur"
]

STATUS_CLASSES = ["High Demand", "Needs Update", "Obsolete/Oversupplied"]


def generate_maharashtra_dataset(num_records: int = 450, noise_rate: float = 0.15, seed: int = 42):
    random.seed(seed)
    records = []
    swapped_count = 0

    for _ in range(num_records):
        profile = random.choice(JOB_PROFILES)
        title = profile["title"]
        base_skills = [s.strip() for s in profile["skills"].split(",")]

        # Add minor natural variation to skills
        if len(base_skills) > 3 and random.random() < 0.35:
            # Drop 1 skill occasionally or add a cross-discipline skill
            sample_k = random.randint(3, len(base_skills))
            skills_subset = random.sample(base_skills, sample_k)
        else:
            skills_subset = list(base_skills)

        skills_str = ", ".join(sorted(skills_subset))
        district = random.choice(DISTRICTS)
        volume = random.randint(profile["volume_range"][0], profile["volume_range"][1])

        base_status = profile["status"]

        # 1. 15% probability of randomly swapping 'Curriculum Status' to an incorrect class
        if random.random() < noise_rate:
            other_classes = [c for c in STATUS_CLASSES if c != base_status]
            final_status = random.choice(other_classes)
            swapped_count += 1
        else:
            final_status = base_status

        records.append({
            "Job Title": title,
            "Job Role": title,
            "Required Skills": skills_str,
            "District/Location": district,
            "Demand Volume": volume,
            "Curriculum Status": final_status,
        })

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} records. Injected label noise into {swapped_count} rows ({swapped_count/len(df)*100:.1f}%).")
    print("Class Distribution:")
    print(df["Curriculum Status"].value_counts())

    # Save to targets
    out_paths = [
        "flask_app/scraped_maharashtra_jobs.csv",
        "flask_app/sample_jobs.csv",
        "scraped_maharashtra_jobs.csv"
    ]
    for p in out_paths:
        dir_name = os.path.dirname(p)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"Saved to: {p}")

    return df


if __name__ == "__main__":
    generate_maharashtra_dataset()
