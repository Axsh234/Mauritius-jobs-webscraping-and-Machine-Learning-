import requests
from bs4 import BeautifulSoup
import csv
import time
import psycopg2

# PostgreSQL config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'scrapnalyze',
    'user': 'postgres',
    'password': '1234'  # Replace with your actual password
}

BASE_URL = "https://www.myjob.mu/ShowResults.aspx?Keywords=&Location=&Category="
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

def get_total_pages(soup):
    pagination = soup.find("ul", id="pagination")
    if not pagination:
        return 1

    page_links = pagination.find_all("a", href=True)
    pages = []
    for link in page_links:
        href = link['href']
        if "Page=" in href:
            try:
                page_num = int(href.split("Page=")[1].split("&")[0])
                pages.append(page_num)
            except ValueError:
                continue

    return max(pages) if pages else 1


def scrape_jobs_from_page(url):
    print(f"\n🔎 Scraping: {url}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    jobs = []
    job_modules = soup.find_all("div", class_="module job-result")
    for job in job_modules:
        module_content = job.find("div", class_="module-content")
        if not module_content:
            continue

        # Extract title and link
        title_div = module_content.find("div", class_="job-result-logo-title")
        title = title_div.get_text(strip=True) if title_div else ""

        link_tag = title_div.find("a", href=True) if title_div else None
        link = link_tag['href'] if link_tag else ""

        # Optional: if links are relative, prefix with base URL
        if link and link.startswith("/"):
            link = "https://www.myjob.mu" + link

        overview_ul = module_content.find("ul", class_="job-overview")
        salary = date_posted = location = closing_date = ""

        if overview_ul:
            salary_li = overview_ul.find("li", class_="salary")
            salary = salary_li.get_text(strip=True) if salary_li else ""

            date_posted_li = overview_ul.find("li", class_="updated-time")
            date_posted = date_posted_li.get_text(strip=True).replace("Added", "").strip() if date_posted_li else ""

            location_li = overview_ul.find("li", class_="location")
            location = location_li.get_text(strip=True) if location_li else ""

            closing_li = overview_ul.find("li", class_="closed-time")
            closing_date = closing_li.get_text(strip=True).replace("Closing", "").strip() if closing_li else ""

        jobs.append({
            "Title": title,
            "Salary": salary,
            "Date Posted": date_posted,
            "Location": location,
            "Closing Date": closing_date,
            "Link": link,
        })

    print(f"✅ Found {len(jobs)} jobs on this page.")
    return jobs, soup




def save_to_csv(jobs, filename="myjob.csv"):
    if not jobs:
        print("No jobs to save.")
        return

    keys = jobs[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(jobs)

    print(f"📝 Saved {len(jobs)} jobs to {filename}")


def insert_into_db(jobs):
    print("🛠 Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ Connected to PostgreSQL.")

        insert_sql = """
            INSERT INTO myjob (title, salary, date_posted, location, closing_date, link)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """

        for job in jobs:
            cursor.execute(insert_sql, (
                job["Title"],
                job["Salary"],
                job["Date Posted"],
                job["Location"],
                job["Closing Date"],
                job["Link"]
            ))

        conn.commit()
        print(f"📥 Inserted {len(jobs)} records into the 'myjob' table.")
        cursor.close()
        conn.close()
        print("🔌 Connection closed.")
    except Exception as e:
        print(f"❌ Failed to insert into DB: {e}")

######################################


def main():
    first_page_url = BASE_URL + "&Page=1"
    response = requests.get(first_page_url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    total_pages = get_total_pages(soup)
    print(f"\n🌐 Total pages found: {total_pages}")

    all_jobs = []

    for page in range(1, total_pages + 1):
        page_url = BASE_URL + f"&Page={page}"
        jobs, _ = scrape_jobs_from_page(page_url)
        all_jobs.extend(jobs)
        time.sleep(1)  # be respectful

    save_to_csv(all_jobs)
    insert_into_db(all_jobs)

if __name__ == "__main__":
    main()
