import socket
import math
import re
import undetected_chromedriver as uc

def force_ipv4():
    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(*args, **kwargs):
        responses = old_getaddrinfo(*args, **kwargs)
        return [r for r in responses if r[0] == socket.AF_INET]
    socket.getaddrinfo = new_getaddrinfo

def get_driver():
    options = uc.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--start-maximized')
    options.add_argument('--blink-settings=imagesEnabled=false')
    return uc.Chrome(options=options, version_main=146, browser_executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

def parse_number(num_str):
    if not num_str: return 0
    num_str = num_str.upper().replace(',', '')
    if 'K' in num_str: return int(float(num_str.replace('K', '')) * 1000)
    if 'M' in num_str: return int(float(num_str.replace('M', '')) * 1000000)
    try:
        return int(num_str)
    except:
        return 0

def compute_stats(movie_data):
    ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    counts = [movie_data.get(f'rating_{r}_count', 0) for r in ratings]
    total_n = sum(counts)
    
    if total_n > 0:
        mean = sum(r * c for r, c in zip(ratings, counts)) / total_n
        variance = sum(c * (r - mean)**2 for r, c in zip(ratings, counts)) / total_n
        movie_data['rating_std_dev'] = round(math.sqrt(variance), 4)

    views = movie_data.get('views', 0)
    likes = movie_data.get('likes', 0)
    if views > 0:
        movie_data['rating_ratio'] = round(total_n / views, 4)
        movie_data['like_view_ratio'] = round(likes / views, 4)
    return movie_data