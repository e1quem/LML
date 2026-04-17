
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import skew, kurtosis, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

name = input("Pseudo: ")
df = pd.read_csv(f'out/enriched_movies_' + name +'.csv', sep=None, engine='python')
watchtime = df['duration_mins'].sum() / 60
print(f"Watchtime: {watchtime:.1f} hours")

# DATA PRESENTATION
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 2, figure=fig)

# 1. Movies & likes per year
ax_tl = fig.add_subplot(gs[0, 0])
yearly_stats = df.groupby('year').agg(
    movie_count=('title', 'count'),
    likes_count=('user_like', 'sum')
)
ax_tl.bar(yearly_stats.index, yearly_stats['movie_count'], label='Movies count', color='#4C86B8', edgecolor='black', linewidth=0.1, width=1.0, zorder=2)
ax_tl.bar(yearly_stats.index, yearly_stats['likes_count'], label='Likes count', color='#D05851', edgecolor='black', linewidth=0.1, width=1.0, zorder=2)
ax_tl.set_title(f"Count of Movies and Likes per Year")
ax_tl.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
ax_tl.legend(frameon=False)
sns.despine(ax=ax_tl)

# 2. Decade Average 
ax_tr = fig.add_subplot(gs[0, 1])
df['decade'] = (df['year'] // 10) * 10
decade_avg = df.groupby('decade')['user_rating'].mean().reset_index()
sns.barplot(data=decade_avg, x='decade', y='user_rating', zorder=2, color='#4C86B8', edgecolor='black', width=1.0, linewidth=0.2, ax=ax_tr, saturation=1)
ax_tr.set_title(f"Average Rating per Decade")
ax_tr.set_ylim(0, 5)
ax_tr.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
sns.despine(ax=ax_tr)

# 3. Top per category
inner_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0])
categories = ['country_1', 'genre_1']
titles = ['Top Countries', 'Top Genres']
for i, cat in enumerate(categories):
    ax_pie = fig.add_subplot(inner_gs[i])
    counts = df[cat].value_counts()
    
    n_colors = len(counts)
    spectral_range = np.linspace(0.1, 0.9, n_colors)
    colors = plt.cm.Spectral(spectral_range[::-1]) 
    labels = [n if j < 3 else "" for j, n in enumerate(counts.index)]
    
    ax_pie.pie(
        counts, 
        labels=labels, 
        colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p) if p > counts.iloc[2]/counts.sum()*100 else '',
        startangle=140,
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.1, 'antialiased': True}
    )
    ax_pie.set_title(titles[i])

# 4. Rating repartition 
ax_br = fig.add_subplot(gs[1, 1])
ratings = df['user_rating'].dropna()
n = len(ratings)
sk = skew(ratings)
kt = kurtosis(ratings)
stat, p = normaltest(ratings)
sns.histplot(df['user_rating'], bins=10, alpha=1, zorder=2, color='#4C86B8', edgecolor='black', linewidth=0.2, ax=ax_br)
ax_br.set_title(
    f"Ratings Distribution\n"
    f"n={n}  |  skew={sk:.2f}  |  kurtosis={kt:.2f} |  normal p={p:.3f}"
)
ax_br.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
sns.despine(ax=ax_br)

plt.tight_layout()
plt.savefig('out/art/overview.svg')