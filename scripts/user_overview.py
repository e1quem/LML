from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import skew, kurtosis, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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
df = pd.read_csv(f'out/enriched_movies_' + name +'.csv', sep=None, engine='python')
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
    gray_range = np.linspace(0.25, 0.85, n_colors)
    colors = plt.cm.Greys(gray_range[::-1])
    labels = [n if j < 3 else "" for j, n in enumerate(counts.index)]
    
    ax_pie.pie(
        counts, 
        labels=labels, 
        colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p) if p > counts.iloc[2]/counts.sum()*100 else '',
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
ax_br.set_title(
    f"Ratings Distribution\n"
    f"n={n}  |  skew={sk:.2f}  |  kurtosis={kt:.2f} |  normal p={p:.3f}"
    , loc="left", pad=10, weight="bold"
)
ax_br.grid(axis='y')
ax_br.grid(axis='x', visible=False)
sns.despine(ax=ax_br)

plt.tight_layout()
plt.savefig(f'out/art/overview_{name}.svg')
