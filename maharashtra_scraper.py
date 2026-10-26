import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_maharashtra_jobs():
    # Example URL for TimesJobs searching for IT jobs in Pune/Mumbai
    url = "https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=IT+Software&txtLocation=Pune%2C+Mumbai"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    jobs = soup.find_all('li', class_='clearfix job-bx wht-shd-bx')
    scraped_data = []
    
    for job in jobs:
        title = job.find('h2').text.strip()
        location = job.find('ul', class_='top-jd-dtl clearfix').find('span').text.strip()
        skills = job.find('span', class_='srp-skills').text.strip()
        
        # Format the data to match your MSINS columns
        scraped_data.append({
            "Job Role": title,
            "Required Skills": skills,
            "District/Location": location,
            "Demand Volume": 1, # Live scraping usually represents 1 vacancy per post
            "Curriculum Status": "High Demand" # Default status for active postings
        })
        
    df = pd.DataFrame(scraped_data)
    df.to_csv("flask_app/scraped_maharashtra_jobs.csv", index=False)
    print("Scraping complete. Saved to scraped_maharashtra_jobs.csv")

scrape_maharashtra_jobs()