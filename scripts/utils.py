import undetected_chromedriver as uc
import socket
import math
import subprocess
import time

def kill_chrome():
    try:
        subprocess.run(["pkill", "-9", "Google Chrome"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "chromedriver"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

# Network
def force_ipv4():
    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(*args, **kwargs):
        responses = old_getaddrinfo(*args, **kwargs)
        return [r for r in responses if r[0] == socket.AF_INET]
    socket.getaddrinfo = new_getaddrinfo

# UC settings. option --headless is caught by Cloudflare
def get_driver():
    options = uc.ChromeOptions()

    #options.page_load_strategy = 'eager'
    options.page_load_strategy = 'none' # test
    options.add_argument('--disable-gpu') # test (ça à l'air de fonctionner)
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    #options.add_argument('--start-maximized')
    options.add_argument('--blink-settings=imagesEnabled=false')
    #options.add_argument('--disable-javascript')

    # Speed settings
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    options.add_argument('--silent')

    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.cookies": 2,
        "profile.default_content_setting_values.media_stream": 2,
        "profile.default_content_setting_values.geolocation": 2,
        "profile.default_content_setting_values.midi_sysex": 2, 
    }
    options.add_experimental_option("prefs", prefs)

    driver = uc.Chrome(
        options=options,
        #version_main=148,
        browser_executable_path="/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"
    )

    driver.set_window_size(840, 512)
    return driver

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

    if views is None:
        views = 0

    if views > 0:
        movie_data['rating_ratio'] = round(total_n / views, 4)
        movie_data['like_view_ratio'] = round(likes / views, 4)
    return movie_data