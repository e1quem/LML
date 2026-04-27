from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
from scipy.stats import skew, kurtosis, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def set_stata_like_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "grid.linestyle": "-",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


name = input("Pseudo: ")
user_file = f'out/movies_{name}.csv'
metadata_file = 'out/movies.csv'

if not os.path.exists(user_file):
    raise FileNotFoundError(f"File {user_file} not found o_O?")

user_ratings = pd.read_csv(user_file, sep=None, engine='python')
metadata = pd.read_csv(metadata_file, sep=None, engine='python')
df = user_ratings.merge(metadata, on='url', how='left', suffixes=('_user', '_meta'))
df.rename(columns={'title_user': 'title', 'year_meta': 'year', 'year_user': 'year_user'}, inplace=True)
df['title'] = df['title'].fillna(df.get('title_meta'))
df['year'] = pd.to_numeric(df['year'], errors='coerce')
watchtime = df['duration_mins'].sum() / 60
print(f"Watchtime: {watchtime:.1f} hours")

# DATA PRESENTATION
set_stata_like_style()
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 2, figure=fig)

# 1. Movies & likes per year
ax_tl = fig.add_subplot(gs[0, 0])
yearly_stats = df.groupby('year').agg(
    movie_count=('title', 'count'),
    likes_count=('user_like', 'sum')
)
ax_tl.bar(yearly_stats.index, yearly_stats['movie_count'], label='Movies count', color='0.70', edgecolor='0.15', linewidth=0.4, width=1.0, zorder=2)
ax_tl.bar(yearly_stats.index, yearly_stats['likes_count'], label='Likes count', color='0.30', edgecolor='0.15', linewidth=0.4, width=1.0, zorder=2)
ax_tl.set_title("Count of Movies and Likes per Year", loc="left", pad=10, weight="bold")
ax_tl.grid(axis='y')
ax_tl.grid(axis='x', visible=False)
ax_tl.legend(frameon=False)
sns.despine(ax=ax_tl)

# 2. Decade Average 
ax_tr = fig.add_subplot(gs[0, 1])
df['decade'] = (df['year'] // 10) * 10
decade_avg = df.groupby('decade')['user_rating'].mean().reset_index()
sns.barplot(data=decade_avg, x='decade', y='user_rating', zorder=2, color='0.55', edgecolor='0.15', width=1.0, linewidth=0.4, ax=ax_tr, saturation=1)
ax_tr.set_title("Average Rating per Decade", loc="left", pad=10, weight="bold")
ax_tr.set_ylim(0, 5)
ax_tr.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value * 10)}"))
ax_tr.grid(axis='y')
ax_tr.grid(axis='x', visible=False)
sns.despine(ax=ax_tr)

# 3. Top per category
inner_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0])
categories = ['country_1', 'genre_1']
titles = ['Top Countries', 'Top Genres']
for i, cat in enumerate(categories):
    ax_pie = fig.add_subplot(inner_gs[i])
    counts = df[cat].value_counts()
    
    n_colors = len(counts)
    gray_range = np.linspace(0.15, 0.55, n_colors)
    colors = plt.cm.Greys(gray_range)
    labels = [n if j < 3 else "" for j, n in enumerate(counts.index)]
    
    ax_pie.pie(
        counts, 
        labels=labels, 
        colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p) if p > counts.iloc[2]/counts.sum()*100 else '',
        textprops={'fontsize': 7, 'color': '0.2'},
        startangle=140,
        wedgeprops={'edgecolor': '0.15', 'linewidth': 0.4, 'antialiased': True}
    )
    ax_pie.set_title(titles[i], loc="left", pad=10, weight="bold")

# 4. Rating repartition 
ax_br = fig.add_subplot(gs[1, 1])
ratings = df['user_rating'].dropna()
n = len(ratings)
sk = skew(ratings)
kt = kurtosis(ratings)
stat, p = normaltest(ratings)
sns.histplot(df['user_rating'], bins=10, alpha=1, zorder=2, color='0.55', edgecolor='0.15', linewidth=0.4, ax=ax_br)
ax_br.set_title("Ratings Distribution", loc="left", pad=10, weight="bold")
ax_br.text(
    0.0,
    0.96,
    f"n={n}  |  skew={sk:.2f}  |  kurtosis={kt:.2f} |  normal p={p:.3f}",
    transform=ax_br.transAxes,
    ha="left",
    va="bottom",
    fontsize=8.5,
    color="0.25",
)
ax_br.grid(axis='y')
ax_br.grid(axis='x', visible=False)
sns.despine(ax=ax_br)

plt.tight_layout()
plt.savefig(f'out/art/overview_{name}.svg')
