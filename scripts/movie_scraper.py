from scripts.utils import force_ipv4, get_driver, parse_number, compute_stats
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
import urllib.parse
import pandas as pd
import re

force_ipv4()

def extract_movie_details(driver, movie_data):
    url = movie_data['url']
    driver.get(url)
    
    try:
        WebDriverWait(driver, 5).until(
            # Presence of element located is not sufficient for smaller movies
            # Other possible fix: EC.visibility_of_element_located
            # EC.presence_of_element_located((By.CSS_SELECTOR, ".production-statistic-list, #film-page-wrapper")) 
            lambda d: d.find_element(By.CSS_SELECTOR, ".-watches .label").text.strip() != ""
        )
    except:
        pass

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

    ratings_dict = {
        '0.5': '1/2', '1.0': '1', '1.5': '1 1/2', '2.0': '2', '2.5': '2 1/2', '3.0': '3', '3.5': '3 1/2', '4.0': '4', '4.5': '4 1/2', '5.0': '5'
    }
    for k in ratings_dict.keys():
        movie_data[f'rating_{k}_count'] = 0

    hist_bars = soup.select('.rating-histogram-bar a')
    for bar in hist_bars:
        title = bar.get('data-original-title', '')
        href = urllib.parse.unquote(bar.get('href', ''))
        
        m_count = re.match(r'^([\d,]+)\s', title)
        if not m_count: continue
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
    for i in range(3): movie_data[f'country_{i+1}'] = countries[i].text.strip() if i < len(countries) else ""
    
    languages = soup.select('a[href^="/films/language/"]')
    movie_data['primary_language'] = languages[0].text.strip() if languages else ""

    genres = soup.select('a[href^="/films/genre/"]')
    for i in range(3): movie_data[f'genre_{i+1}'] = genres[i].text.strip() if i < len(genres) else ""

    themes = soup.select('a[href^="/films/theme/"]')
    for i in range(2): movie_data[f'theme_{i+1}'] = themes[i].text.strip() if i < len(themes) else ""

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

if __name__ == "__main__":
    filename_input = input("CSV file: ")
    path = Path(filename_input)
    df = pd.read_csv('out/' + filename_input + '.csv', sep=None, engine='python')
    driver = get_driver()
    
    enriched_results = []
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            enriched_results.append(extract_movie_details(driver, row.to_dict()))
    finally:
        driver.quit()

    output_path = f"out/enriched_{path.name}"
    pd.DataFrame(enriched_results).to_csv(output_path, index=False)
    print(f"Enriched data saved: {output_path}")