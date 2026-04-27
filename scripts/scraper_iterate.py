from scripts.utils import force_ipv4, get_driver, parse_number, compute_stats, kill_chrome
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
import urllib.parse
import pandas as pd
import re
import random
import time

# Network configuration, ipv6 functions poorly in China
force_ipv4()

def scrape_letterboxd_user(driver, pseudo):
    base_url = "https://letterboxd.com"
    current_url = f"{base_url}/{pseudo}/films/rated/.5-5/"
    films_urls = []

    while current_url:
        driver.get(current_url)
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.griditem")))
        except TimeoutException:
            return []
        
        soup = BeautifulSoup(driver.page_source, "html.parser") # tested with lxml: doesn't work
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
        time.sleep(random.uniform(0.1, 0.5))

    return films_urls


def extract_movie_details(driver, movie_data):
    movie_data.pop('user_rating', None)
    movie_data.pop('user_like', None)
    driver.get(movie_data['url'])
    
    try:
        WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".rating-histogram .barcolumn"))
        )
    except TimeoutException:
        pass

    # Selenium extraction for rating histogram
    ratings_keys = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
    for k in ratings_keys:
        movie_data[f'rating_{k}_count'] = 0

    bars = driver.find_elements(By.CSS_SELECTOR, ".rating-histogram .barcolumn")
    for bar in bars:
        try:
            title = bar.get_attribute('data-original-title')
            href = urllib.parse.unquote(bar.get_attribute('href'))
            
            m_count = re.match(r'^([\d,]+)\s', title)
            if m_count:
                count = int(m_count.group(1).replace(',', ''))
                if '/rated/½/' in href: movie_data['rating_0.5_count'] = count
                elif '/rated/1/' in href: movie_data['rating_1.0_count'] = count
                elif '/rated/1½/' in href: movie_data['rating_1.5_count'] = count
                elif '/rated/2/' in href: movie_data['rating_2.0_count'] = count
                elif '/rated/2½/' in href: movie_data['rating_2.5_count'] = count
                elif '/rated/3/' in href: movie_data['rating_3.0_count'] = count
                elif '/rated/3½/' in href: movie_data['rating_3.5_count'] = count
                elif '/rated/4/' in href: movie_data['rating_4.0_count'] = count
                elif '/rated/4½/' in href: movie_data['rating_4.5_count'] = count
                elif '/rated/5/' in href: movie_data['rating_5.0_count'] = count
        except:
            continue

    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    stats_list = soup.select_one('.production-statistic-list')
    if stats_list:
        views_el = stats_list.select_one('.-watches')
        if views_el and views_el.has_attr('aria-label'):
            m_views = re.search(r'([\d,]+)', views_el['aria-label'])
            movie_data['views'] = int(m_views.group(1).replace(',', '')) if m_views else 0
            
        likes_el = stats_list.select_one('.-likes')
        if likes_el and likes_el.has_attr('aria-label'):
            m_likes = re.search(r'([\d,]+)', likes_el['aria-label'])
            movie_data['likes'] = int(m_likes.group(1).replace(',', '')) if m_likes else 0

    fans_el = soup.select_one('a[href$="/fans/"]')
    movie_data['fans'] = parse_number(fans_el.text.replace('fans', '').strip()) if fans_el else 0

    rating_el = soup.select_one('a.display-rating')
    if rating_el:
        movie_data['avg_rating'] = float(rating_el.text.strip())
        title_attr = rating_el.get('data-original-title', '')
        m = re.search(r'based on ([\d,]+) ratings', title_attr)
        if m:
            movie_data['total_ratings'] = parse_number(m.group(1))

    # Manual computatioon of avg_rating (not present for smaller movies)
    if not movie_data.get('avg_rating'):
        total_score = sum(movie_data.get(f'rating_{k}_count', 0) * float(k) for k in ratings_keys)
        total_votes = sum(movie_data.get(f'rating_{k}_count', 0) for k in ratings_keys)
        if total_votes > 0:
            movie_data['avg_rating'] = round(total_score / total_votes, 2)
            if not movie_data.get('total_ratings'):
                movie_data['total_ratings'] = total_votes

    footer_p = soup.select_one('.text-footer')
    if footer_p:
        m_mins = re.search(r'(\d+)\s*mins?', footer_p.text)
        if m_mins:
            movie_data['duration_mins'] = int(m_mins.group(1))

    cast_links = soup.select('#tab-cast .cast-list a.text-slug')
    for i in range(5):
        movie_data[f'actor_{i+1}'] = cast_links[i].text.strip() if i < len(cast_links) else ""

    studios = soup.select('a[href^="/studio/"]')
    for i in range(2): movie_data[f'studio_{i+1}'] = studios[i].text.strip() if i < len(studios) else ""
    
    countries = soup.select('a[href^="/films/country/"]')
    movie_data.pop('country', None)
    #for i in range(1): movie_data[f'country_{i+1}'] = countries[i].text.strip() if i < len(countries) else ""
    movie_data['country_1'] = countries[0].text.strip() if countries else ""
    
    languages = soup.select('a[href^="/films/language/"]')
    movie_data['primary_language'] = languages[0].text.strip() if languages else ""

    genres = soup.select('a[href^="/films/genre/"]')
    #for i in range(1): movie_data[f'genre_{i+1}'] = genres[i].text.strip() if i < len(genres) else ""
    movie_data['genre_1'] = genres[0].text.strip() if genres else ""

    # Don't need theme
    #themes = soup.select('a[href^="/films/theme/"]')
    #for i in range(2): movie_data[f'theme_{i+1}'] = themes[i].text.strip() if i < len(themes) else ""

    crew_tab = soup.select_one('#tab-crew')
    if crew_tab:
        def get_crew_from_soup(target_soup, role_name, limit):
            headers = target_soup.select('h3')
            for h in headers:
                if role_name.lower() in h.text.lower():
                    sluglist = h.find_next_sibling('div', class_='text-sluglist')
                    if sluglist:
                        return [a.text.strip() for a in sluglist.select('a.text-slug')][:limit]
            return []

        directors = get_crew_from_soup(crew_tab, 'Director', 3)
        for i in range(3): movie_data[f'director_{i+1}'] = directors[i] if i < len(directors) else ""
        producers = get_crew_from_soup(crew_tab, 'Producers', 3)
        for i in range(3): movie_data[f'producer_{i+1}'] = producers[i] if i < len(producers) else ""
        writers = get_crew_from_soup(crew_tab, 'Writers', 1)
        for i in range(1): movie_data[f'writer_{i+1}'] = writers[i] if i < len(writers) else ""

    return compute_stats(movie_data)


# For values that are availables but that we missed
def validate_and_retry(driver, df, max_retries=3):
    """Vérifie et réessaie les films avec valeurs manquantes"""
    for attempt in range(max_retries):
        missing = df[
            (df['views'].isna() | (df['views'] == 0)) |
            (df['avg_rating'].isna() | (df['avg_rating'] == 0)) |
            (df['year'].isna() | (df['year'] == 0))
        ]
        if missing.empty:
            print(f"All valid after {attempt} retries")
            return df
        
        print(f"Retry {attempt + 1}: {len(missing)} movies missing views")
        
        retry_results = []
        for _, movie in missing.iterrows():
            movie_dict = movie.to_dict()
            movie_dict['views'] = None
            retry_results.append(extract_movie_details(driver, movie_dict))
        
        retry_df = pd.DataFrame(retry_results)
        
        for idx, row in retry_df.iterrows():
            mask = df['url'] == row['url']
            if mask.any():
                df.loc[mask, 'views'] = row['views']
                if 'likes' in row:
                    df.loc[mask, 'likes'] = row['likes']
    
    return df

if __name__ == "__main__":
    kill_chrome()
    pseudos_input = input("Letterboxd pseudos list (separated by commas): ")
    usernames = [u.strip() for u in pseudos_input.split(",") if u.strip()]
    
    driver = get_driver()

    try:
        for username in usernames:
            print(f"\n{username}")
            try:
                movies = scrape_letterboxd_user(driver, username)
                
                if not movies:
                    print(f"No movies o_O?")
                    continue
                    
                df_rated = pd.DataFrame(movies)
                df_rated.to_csv(f"out/movies_{username}.csv", index=False)

                print(f"{len(df_rated)} rated movies exported to out/movies_{username}.csv")

                df = df_rated.drop_duplicates(subset=['url'], keep='first')
                
                db_path = "out/movies.csv"
                if Path(db_path).exists():
                    movie_db = pd.read_csv(db_path, sep=None, engine='python')
                    existing_urls = set(movie_db['url'].dropna().values)
                else:
                    movie_db = pd.DataFrame()
                    existing_urls = set()
                
                movies_to_scrape = []
                already_in_base = 0
                
                for idx, row in df.iterrows():
                    if row['url'] not in existing_urls:
                        movies_to_scrape.append(row.to_dict())
                    else:
                        already_in_base += 1
                
                print(f"{len(df)} total movies ({already_in_base} in DB, {len(movies_to_scrape)} to scrape)")
                
                if not movies_to_scrape:
                    print("No new movies to scrape")
                else:
                    enriched_results = []
                    for movie_data in tqdm(movies_to_scrape, total=len(movies_to_scrape)):
                        enriched_results.append(extract_movie_details(driver, movie_data))
                    
                    new_df = pd.DataFrame(enriched_results)
                    new_df = validate_and_retry(driver, new_df)
                    
                    if movie_db.empty:
                        updated_db = new_df
                    else:
                        updated_db = pd.concat([movie_db, new_df], ignore_index=True)
                        updated_db = updated_db.drop_duplicates(subset=['url'], keep='last')
                    
                    updated_db.to_csv(db_path, index=False)
                    print(f"Updated DB: {len(updated_db)} movies")
                    
            except Exception as e:
                print(f"Error on {username}: {e}")

    finally:
        driver.quit()