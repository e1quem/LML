from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from utils import force_ipv4, get_driver
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

force_ipv4()

def scrape_letterboxd_user(pseudo):
    driver = get_driver()
    base_url = "https://letterboxd.com"
    current_url = f"{base_url}/{pseudo}/watchlist/"
    films_urls = []

    try:
        while current_url:
            driver.get(current_url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.griditem")))
            
            soup = BeautifulSoup(driver.page_source, "html.parser")
            items = soup.select("li.griditem")
            
            for item in items:
                component = item.select_one(".react-component")
                if not component: continue

                frame = item.select_one("a.frame")
                title, year = "", ""

                if frame:
                    raw = frame.get("data-original-title") or frame.get_text(strip=True)
                    if raw and "(" in raw and ")" in raw:
                        title = raw.rsplit("(", 1)[0].strip()
                        year = raw.rsplit("(", 1)[1].replace(")", "").strip()
                    else:
                        title = raw
                
                films_urls.append({
                    'title': title,
                    'year': year,   
                    'url': base_url + component.get("data-item-link", "")
                })

            next_btn = soup.select_one(".paginate-nextprev a.next")
            current_url = base_url + next_btn.get("href") if next_btn else None
            time.sleep(random.uniform(0.5, 1.5))

    finally:
        driver.quit()
    return films_urls

if __name__ == "__main__":
    username = input("Letterboxd profile: ")
    data = scrape_letterboxd_user(username)
    pd.DataFrame(data).to_csv(f"out/watchlist_{username}", index=False)