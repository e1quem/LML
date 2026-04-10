from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from scripts.utils import force_ipv4, get_driver
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

force_ipv4()

def scrape_letterboxd_user(driver, pseudo):
    base_url = "https://letterboxd.com"
    current_url = f"{base_url}/{pseudo}/films/rated/.5-5/"
    films_urls = []

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
                
            note = ""
            rating_span = item.select_one(".rating")
            if rating_span:
                for cls in rating_span.get("class", []):
                    if cls.startswith("rated-"):
                        note = str(int(cls.split("-")[1]) / 2)
                        break
            
            films_urls.append({
                'title': title,
                'year': year,   
                'url': base_url + component.get("data-item-link", ""),
                'user_rating': note,
                'user_like': 1 if item.select_one(".like") else 0
            })

        next_btn = soup.select_one(".paginate-nextprev a.next")
        current_url = base_url + next_btn.get("href") if next_btn else None
        time.sleep(random.uniform(0.5, 1.5))

    return films_urls

def scrape_user_wl(driver, pseudo):
    base_url = "https://letterboxd.com"
    current_url = f"{base_url}/{pseudo}/watchlist/"
    films_urls = []

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

    return films_urls

if __name__ == "__main__":
    username = input("Letterboxd profile: ")
    driver = get_driver()

    try:
        movies = scrape_letterboxd_user(driver, username)
        pd.DataFrame(movies).to_csv(f"out/movies_{username}.csv", index=False)
        print(f"{username}'s rated movies successfully exported to out/movies_{username}.csv")
    
        wl = scrape_user_wl(driver, username)
        pd.DataFrame(wl).to_csv(f"out/watchlist_{username}.csv", index=False)
        print(f"{username}'s watchlist successfully exported to out/watchlist_{username}.csv")

    finally: 
        driver.quit()