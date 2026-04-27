from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import shap 
import os

warnings.filterwarnings("ignore")

class CatBoostTqdmCallback: # Visual utils (tqdm loading)
    def __init__(self, iterations):
        self.pbar = tqdm(total=iterations)
    def after_iteration(self, info):
        self.pbar.update(1)
        return True
    def __del__(self):
        self.pbar.close()

name = input("Pseudo: ")

# Merging data sources
user_ratings = pd.read_csv(f'out/movies_{name}.csv', sep=None, engine='python')
metadata = pd.read_csv('out/movies.csv', sep=None, engine='python')
df = user_ratings.merge(metadata, on='url', how='left')

# Renaming duplicate columns
df.rename(columns={
    'user_rating_x': 'user_rating',
    'user_like_x': 'user_like',
    'year_x': 'year_user',
}, inplace=True)

# Missing values
initial_count = len(df)
df = df.dropna(subset=['avg_rating', 'total_ratings', 'rating_std_dev'])
if len(df) < initial_count:
    print(f"Warning: {initial_count - len(df)} movies dropped due to missing metadata")

# DEBUG
#print("Merge columns:", df.columns.tolist())
#print("Columns in user_ratings:", user_ratings.columns.tolist())
#print("Columns in metadata:", metadata.columns.tolist())


######      FEATURES ENGINEERING
# Niche, log, controversy, age
df['is_niche'] = np.where(df['total_ratings'] > 500, 0, 1)
for col in ['views', 'likes', 'fans', 'total_ratings']:
    df[col] = np.log1p(df[col])
df['controversy'] = df['rating_std_dev'] * df['total_ratings']
df['year'] = 2026 - df['year_y']

# Ratings skewness
def calculate_skew(row):
    counts = [
        row.get('rating_0.5_count', 0), row.get('rating_1.0_count', 0),
        row.get('rating_1.5_count', 0), row.get('rating_2.0_count', 0),
        row.get('rating_2.5_count', 0), row.get('rating_3.0_count', 0),
        row.get('rating_3.5_count', 0), row.get('rating_4.0_count', 0),
        row.get('rating_4.5_count', 0), row.get('rating_5.0_count', 0)
    ]
    ratings = np.repeat(np.arange(0.5, 5.5, 0.5), counts)
    if len(ratings) < 2: return 0
    return pd.Series(ratings).skew()

df['rating_skew'] = df.apply(calculate_skew, axis=1)

# Visual representation of features correlation 
def heatmap_correlation_movies(df, name='user_rating'):

    all_numeric_vars = [
        'user_rating', 'user_like', 'year_user', 'views', 'likes', 'fans', 
        'avg_rating', 'total_ratings', 'duration_mins', 'rating_std_dev',
        'rating_ratio', 'like_view_ratio', 'rating_0.5_count', 'rating_1.0_count',
        'rating_1.5_count', 'rating_2.0_count', 'rating_2.5_count', 'rating_3.0_count',
        'rating_3.5_count', 'rating_4.0_count', 'rating_4.5_count', 'rating_5.0_count',
        'is_niche', 'controversy', 'rating_skew'
    ]
    
    existing_vars = [var for var in all_numeric_vars if var in df.columns]
    df_numeric = df[existing_vars].apply(pd.to_numeric, errors='coerce')
    corr = df_numeric.corr(method='pearson').round(3)
    
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        ax=ax,
        cmap='RdBu',  
        vmin=-1, vmax=1, center=0,
        annot=True, fmt='.2f', annot_kws={'size': 8},
        linewidths=0.5, linecolor='white',
        cbar=True, cbar_kws={'shrink': 0.8, 'ticks': [-1, -0.5, 0, 0.5, 1]},
        square=True  
    )
    
    ax.set_title(f'Movie Features Correlation Matrix ({name})', pad=20, fontsize=12)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    for text in ax.texts:
        text.set_fontsize(7)

    plt.tight_layout()
    plt.savefig(f'out/art/{name}_correlation_hm.png', dpi=400, bbox_inches='tight')
    plt.close()

    return corr['user_rating'].to_dict()

corr_results = heatmap_correlation_movies(df, name)
corr_dict = {f'corr_{k}': v for k, v in corr_results.items() if k != 'user_rating'}


# Droping columns we don't need
df = df.drop(columns=[
    'rating_0.5_count', 'rating_1.0_count', 'rating_1.5_count', 
    'rating_2.0_count','rating_2.5_count', 'rating_3.0_count', 
    'rating_3.5_count', 'rating_4.0_count', 'rating_4.5_count',
    'rating_5.0_count', 'title_x', 'year_x','year_y', 'url', 
    'title_y'], errors='ignore')

# DEBUG
#print("Columns after dropping:", df.columns.tolist())

# Strict 80/20 split between training and holdout data
# Added stratyfing on user_like
df_trainval, df_holdout = train_test_split(df, test_size=0.2, random_state=42, stratify=df['user_like']) 
# Keep ~60% for base training and 20% for stacking-aware meta-training
df_train, df_stack = train_test_split(df_trainval, test_size=0.25, random_state=42, stratify=df_trainval['user_like'])

# Defining user average per thematics, with a leave-one-out mechanism to avoid overfitting
def target_encode_strict(train_df, test_df, cols, target):
    melted = train_df.melt(id_vars=[target], value_vars=cols).dropna()
    melted = melted[melted['value'] != '']
    stats = melted.groupby('value')[target].agg(['sum', 'count'])
    global_mean = train_df[target].mean()

    def get_loo_train(row):
        entities = [row[c] for c in cols if pd.notna(row[c]) and row[c] != '']
        row_loo = [(stats.loc[e, 'sum'] - row[target]) / (stats.loc[e, 'count'] - 1) 
                   for e in entities if e in stats.index and stats.loc[e, 'count'] > 1]
        return np.mean(row_loo) if row_loo else global_mean

    def get_mean_test(row):
        entities = [row[c] for c in cols if pd.notna(row[c]) and row[c] != '']
        row_means = [stats.loc[e, 'sum'] / stats.loc[e, 'count'] 
                     for e in entities if e in stats.index]
        return np.mean(row_means) if row_means else global_mean

    return train_df.apply(get_loo_train, axis=1), test_df.apply(get_mean_test, axis=1)

for cols in [
    (['director_1', 'director_2', 'director_3'], 'user_dir_avg'),
    (['studio_1', 'studio_2'], 'user_studio_avg'),
    (['writer_1'], 'user_writer_avg'),
    (['genre_1'], 'user_genre_avg')
]:
    column_name = cols[1]
    enc_cols = cols[0]
    df_train[column_name], df_stack[column_name] = target_encode_strict(df_train, df_stack, enc_cols, 'user_rating')
    _, df_holdout[column_name] = target_encode_strict(df_train, df_holdout, enc_cols, 'user_rating')

# Fusioning textual groups, so that actor_1 and actor_5 both count equally as "actors", with no hierarchy
groups = {
    'actors': [f'actor_{i}' for i in range(1, 6)],
    'studios': [f'studio_{i}' for i in range(1, 3)],
    'directors': [f'director_{i}' for i in range(1, 4)],
    'producers': [f'producer_{i}' for i in range(1, 4)]
}

# Cleaning our data and veryfing types
def process_text(dataset):
    temp_df = dataset.copy()
    
    for new_column, old_columns in groups.items():
        existing_cols = [col for col in old_columns if col in temp_df.columns]
        
        for col in existing_cols:
            temp_df[col] = temp_df[col].astype(str).str.replace(' ', '_')
            temp_df[col] = temp_df[col].replace('nan', '').replace('None', '')
        
        def join_non_empty(row):
            non_empty = [str(x) for x in row if str(x).strip() and x not in ['', 'nan', 'None']]
            return ' '.join(non_empty) if non_empty else ''
        
        temp_df[new_column] = temp_df[existing_cols].apply(join_non_empty, axis=1)
        temp_df[new_column] = temp_df[new_column].str.replace(r'\s+', ' ', regex=True).str.strip()
        temp_df = temp_df.drop(columns=existing_cols, errors='ignore')
    
    return temp_df
        
df_train = process_text(df_train)
df_stack = process_text(df_stack)
df_holdout = process_text(df_holdout)


######      MODELS
# Defining Catboost matrix
# We define 3 targets for ou models: absolute rating, delta from the mean rating and binary (like/no like)
y_train_abs = df_train['user_rating']
y_holdout_abs = df_holdout['user_rating']
y_stack_abs = df_stack['user_rating']
y_train_delta = df_train['user_rating'] - df_train['avg_rating']
y_holdout_delta = df_holdout['user_rating'] - df_holdout['avg_rating']
y_stack_delta = df_stack['user_rating'] - df_stack['avg_rating']
y_train_like = df_train['user_like']
y_holdout_like = df_holdout['user_like']
y_stack_like = df_stack['user_like']

X_train = df_train.drop(columns=['title', 'url', 'user_rating', 'user_like'], errors='ignore')
X_stack = df_stack.drop(columns=['title', 'url', 'user_rating', 'user_like'], errors='ignore')
X_holdout = df_holdout.drop(columns=['title', 'url', 'user_rating', 'user_like'], errors='ignore')

# Textual values
for col in ['primary_language', 'country_1', 'genre_1', 'writer_1']:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna('Unknown').astype('category')
        X_stack[col] = X_stack[col].fillna('Unknown').astype('category')
        X_holdout[col] = X_holdout[col].fillna('Unknown').astype('category')
        
cat_features = [col for col in X_train.columns 
                if X_train[col].dtype.name == 'category' or (X_train[col].dtype == 'object' and col not in text_features)]

text_features = list(groups.keys())

X_stack = X_stack.reindex(columns=X_train.columns, fill_value=-1)
X_holdout = X_holdout.reindex(columns=X_train.columns, fill_value=-1)

for dataset in (X_train, X_stack, X_holdout):
    if cat_features:
        dataset[cat_features] = dataset[cat_features].fillna('None')
    dataset[text_features] = dataset[text_features].fillna('')
    
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(-1)

# Respective weighting for each model
# 20 when normally distributed, 40 when not a lot of extreme values
# We tried adaptative weighting based on the distance to the mean, but it didn't perform better than a simple static weighting, we kept it simple
weights_abs = y_train_abs.apply(lambda x: 20.0 if x <= 1.0 or x >= 4.5 else 1.0)
weights_delta = y_train_delta.abs().apply(lambda x: 20.0 if x >= 1.5 else 1.0)

# Adaptive weight for the like model
total_movies = len(y_train_like)
total_likes = (y_train_like == 1).sum()
like_ratio = (total_movies - total_likes) / total_likes if total_likes > 0 else 1.0

pool_train_abs = Pool(
    X_train,
    y_train_abs,
    cat_features=cat_features,
    text_features=text_features,
    weight=weights_abs
)
pool_train_delta = Pool(
    X_train,
    y_train_delta,
    cat_features=cat_features,
    text_features=text_features,
    weight=weights_delta
)
pool_train_like = Pool(
    X_train,
    y_train_like,
    cat_features=cat_features,
    text_features=text_features
) 

pool_stack = Pool(
    X_stack,
    cat_features=cat_features,
    text_features=text_features
)

pool_holdout = Pool(
    X_holdout,
    cat_features=cat_features,
    text_features=text_features
)

# Fitting
print("Training 3 CatBoost models (1. Absolute, 2. Delta, 3. Like)...\n")

cb_abs = CatBoostTqdmCallback(2000)
model_abs = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(pool_train_abs, callbacks=[cb_abs])
cb_delta = CatBoostTqdmCallback(2000)
model_delta = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(pool_train_delta, callbacks=[cb_delta])
cb_like = CatBoostTqdmCallback(2000)
model_like = CatBoostClassifier(iterations=2000, class_weights=[1, like_ratio], learning_rate=0.005, depth=6, l2_leaf_reg=5, eval_metric='AUC', verbose=0).fit(pool_train_like, callbacks=[cb_like])

# Prediction
stack_p_abs = model_abs.predict(pool_stack)
stack_p_delta = model_delta.predict(pool_stack) + X_stack['avg_rating'].values
stack_p_like_prob = model_like.predict_proba(pool_stack)[:, 1]

holdout_p_abs = model_abs.predict(pool_holdout)
holdout_p_delta = model_delta.predict(pool_holdout) + X_holdout['avg_rating'].values
holdout_p_like_prob = model_like.predict_proba(pool_holdout)[:, 1]

# Ridge stacking
stack_meta_X = np.column_stack((stack_p_abs, stack_p_delta, stack_p_like_prob))
holdout_meta_X = np.column_stack((holdout_p_abs, holdout_p_delta, holdout_p_like_prob))
meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_meta_X, y_stack_abs)
final_preds = meta_model.predict(holdout_meta_X)


######      RESULTS
results = pd.DataFrame({
    'Observed': y_holdout_abs.values,
    'Estimated': final_preds
})

print(f"\n\n\n\n")
print(results.describe())
print(f"\n\nMean error: {mean_absolute_error(y_holdout_abs, final_preds):.4f} points")
print(f"R2        : {r2_score(y_holdout_abs, final_preds):.4f}")
print(f"\nModel weights")
print(f"Absolute: {meta_model.coef_[0]:.4f}")
print(f"Relative: {meta_model.coef_[1]:.4f}")
print(f"Like    : {meta_model.coef_[2]:.4f}\n")

# Clipping for hits and misses: hit rate
rounded_preds = np.round(final_preds * 2) / 2
rounded_preds = np.clip(rounded_preds, 0.5, 5.0)

y_true = y_holdout_abs.values
rounded_preds = np.round(final_preds * 2) / 2
rounded_preds = np.clip(rounded_preds, 0.5, 5.0)

hits_exact = np.sum(rounded_preds == y_true)
misses_exact = len(y_true) - hits_exact
hit_rate_exact = (hits_exact / len(y_true)) * 100

within_05 = np.sum(np.abs(final_preds - y_true) <= 0.5)
misses_05 = len(y_true) - within_05
hit_rate_05 = (within_05 / len(y_true)) * 100

within_10 = np.sum(np.abs(final_preds - y_true) <= 1.0)
misses_10 = len(y_true) - within_10
hit_rate_10 = (within_10 / len(y_true)) * 100

# Table
hit_table = pd.DataFrame({
    'Exact': [hits_exact, misses_exact, f"{hit_rate_exact:.2f}%"],
    '±0.5': [within_05, misses_05, f"{hit_rate_05:.2f}%"],
    '±1.0': [within_10, misses_10, f"{hit_rate_10:.2f}%"]
}, index=['Hits', 'Misses', 'Hit rate'])

print(hit_table)

# Hit rate by rating
hit_rates_list = []
hit_rates = {}
bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
for r in bins:
    low, high = (0, 0.75) if r == 0.5 else ((4.75, 5.5) if r == 5.0 else (r-0.25, r+0.25))
    mask = (y_true == r)
    if mask.any():
        hits = np.sum((final_preds[mask] >= low) & (final_preds[mask] <= high))
        hit_rates_list.append(f"{hits/mask.sum()*100:.1f}%")
    else:
        hit_rates_list.append("N/A")

print("Rating  : ", "  ".join([str(r) for r in bins]))
print("Hit rate: ", "  ".join(hit_rates_list))

# Like model F1 score computation
holdout_like_probs = model_like.predict_proba(pool_holdout)[:, 1]
y_true_like = y_holdout_like.values
thresholds = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
f1_scores = []

for threshold in thresholds:
    y_pred = (holdout_like_probs >= threshold).astype(int)
    
    tp = np.sum((y_true_like == 1) & (y_pred == 1))
    fp = np.sum((y_true_like == 0) & (y_pred == 1))
    tn = np.sum((y_true_like == 0) & (y_pred == 0))
    fn = np.sum((y_true_like == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    f1_scores.append(f1)

f1_df = pd.DataFrame([thresholds, f1_scores], index=['Threshold', 'F1-score'])
print(f1_df.to_string(float_format='%.4f', index=False))

#### EXPORTING RESULTS
std_diff = y_holdout_abs.std() - pd.Series(final_preds).std()

f1_dict = {}
for t, f1 in zip(thresholds, f1_scores):
    col_name = f"like_F1_{int(t * 100)}"
    f1_dict[col_name] = f1

for r, hr in zip(bins, hit_rates_list):
    col_name = f"hr_{int(r*10)}"
    hit_rates[col_name] = hr

results_summary = pd.DataFrame({
    'pseudo': [name],
    'followers': [''],
    'rated_movies': [len(user_ratings)],
    'observations': [len(y_holdout_abs)],
    'R2': [r2_score(y_holdout_abs, final_preds)],
    'mean_error': [mean_absolute_error(y_holdout_abs, final_preds)],
    'hit_rate': [hit_rate_exact],
    'hit_rate±05': [hit_rate_05],
    'hit_rate±1': [hit_rate_10],
    **hit_rates,
    'std_diff': [std_diff],
    **f1_dict,
    'absolute_w': [meta_model.coef_[0]],
    'relative_w': [meta_model.coef_[1]],
    'like_w': [meta_model.coef_[2]],
    **corr_dict
})


#######     VISUALISATION OF RESULTS
# Feature importance print
print("\nFeature importance for Absolute model (y):")
feature_importance = model_abs.get_feature_importance()
for score, name in sorted(zip(feature_importance, X_train.columns), reverse=True):
    print(f'{name}: {score:.2f}')

# Feature importance print
print("\nFeature importance for Delta model (y):")
feature_importance = model_delta.get_feature_importance()
for score, name in sorted(zip(feature_importance, X_train.columns), reverse=True):
    print(f'{name}: {score:.2f}')

# Missing data
#fig = plt.figure(figsize=(12, 9))
#sns.heatmap(_clean.isna().T, cbar=False, yticklabels=True, xticklabels=False, cmap='viridis')
#plt.title("Missing Features Heatmap")
#plt.savefig(f'out/art/{name}_missing_features.png', dpi=400, bbox_inches='tight')
#plt.close()

def set_stata_like_style():
    """Configuration du style Stata (à placer en haut du script)"""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.8,
        "axes.grid": False,  # Stata n'a pas de grille sur les heatmaps
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
    })


def plot_missing_data_heatmap(df: pd.DataFrame, name: str = "dataset") -> Path:

    set_stata_like_style()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Palette Stata: gris clair (présent) -> gris foncé (manquant)
    #stata_missing_cmap = sns.light_palette("0.35", as_cmap=True)
    stata_missing_cmap = sns.light_palette("0.35", n_colors=2, as_cmap=True)
    # stata_missing_cmap = 'Greys'
    
    sns.heatmap(
        df.isna().T,
        cbar=False,
        yticklabels=True,
        xticklabels=False,
        cmap=stata_missing_cmap, 
        ax=ax,
        cbar_kws=None
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    
    ax.set_title(
        "Missing features heatmap", 
        loc="left", 
        pad=10, 
        weight="bold",
        fontsize=10
    )
    
    ax.tick_params(axis="y", colors="0.2", labelsize=8)
    
    output_path = Path(f"out/art/{name}_missing_features.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    
    return output_path

plot_missing_data_heatmap(df, name)

# SHAP values absolute & relative
# Config
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # Ou une autre police système

explainer = shap.TreeExplainer(model_abs)
shap_values = explainer.shap_values(pool_train_abs)
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig(f'out/art/{name}_SHAP_absolute.png', dpi=400, bbox_inches='tight')
plt.close()

explainer = shap.TreeExplainer(model_delta)
shap_values = explainer.shap_values(pool_train_delta)
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig(f'out/art/{name}_SHAP_delta.png', dpi=400, bbox_inches='tight')
plt.close()

## Top 3 features shap dependence plot
#models_to_plot = [
#    (model_abs, pool_train_abs, "Absolute"),
#    (model_delta, pool_train_delta, "Delta")]
#
#for model, pool, label in models_to_plot:
#    importances = model.get_feature_importance()
#    feature_names = X_train.columns
#    top_3_features = [name for _, name in sorted(zip(importances, feature_names), reverse=True)[:3]]
#    explainer = shap.TreeExplainer(model)
#    shap_v = explainer.shap_values(pool)
#    
#    for feature in top_3_features:
#        plt.figure(figsize=(8, 5))
#        shap.dependence_plot(feature, shap_v, X_train, show=False)
#        plt.title(f"SHAP Dependence - {label} Model - {feature}")
#        plt.yticks(fontsize=11)
#        plt.xticks(fontsize=11)
#        plt.tight_layout()
#        plt.savefig(f'out/art/{name}_{label}_{feature}_SHAP_dependence.svg')


# Observed vs Estimated plot - Version style Stata
set_stata_like_style()

plot_df = pd.DataFrame({
    'Observed': y_holdout_abs.values,'Estimated': final_preds
}).sort_values(by='Observed').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 6))

y_min = np.where(plot_df['Observed'] == 0.5, 0, plot_df['Observed'] - 0.25)
y_max = plot_df['Observed'] + 0.25
heights = y_max - y_min

ax.bar(x=plot_df.index, height=heights, bottom=y_min, width=1.0,
       color='0.85', alpha=0.5, edgecolor='none', zorder=0, 
       label='Tolerance Zone')

ax.scatter(plot_df.index, plot_df['Observed'],
           label='Observed', color='0.08', s=40,
           edgecolor='white', linewidth=0.5, zorder=3, alpha=1)

ax.scatter(plot_df.index, plot_df['Estimated'],
           label='Estimated', color='0.55', s=40,
           edgecolor='white', linewidth=0.5, zorder=2, alpha=1)

ax.vlines(x=plot_df.index, ymin=plot_df['Observed'],
          ymax=plot_df['Estimated'], color='0.7',
          alpha=0.3, linewidth=1, zorder=1)

# Regression lines
coef_obs = np.polyfit(plot_df.index, plot_df['Observed'], 1)
poly_obs = np.poly1d(coef_obs)
ax.plot(plot_df.index, poly_obs(plot_df.index), color='0.2',
        linestyle='--', linewidth=0.8, label='Regression Observed')

coef_est = np.polyfit(plot_df.index, plot_df['Estimated'], 1)
poly_est = np.poly1d(coef_est)
ax.plot(plot_df.index, poly_est(plot_df.index), color='0.5',
        linestyle='--', linewidth=0.8, label='Regression Estimated')

ax.set_title('Estimated vs Observed User Ratings', loc='left', pad=10, weight='bold')
ax.set_xlabel('Movies')
ax.set_ylabel('User Rating')

ax.grid(True, axis='y', linestyle='-', linewidth=0.5, color='0.88')
ax.grid(False, axis='x')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig(f'out/art/{name}_ObservedEstimated.svg')
plt.close()

# Second plot - style Stata
set_stata_like_style()

plot_df2 = pd.DataFrame({
    'Observed': y_holdout_abs.values,
    'Estimated': final_preds
}).sort_values(by='Estimated').reset_index(drop=True)

fig2, ax2 = plt.subplots(figsize=(12, 6))

ax2.scatter(plot_df2.index, plot_df2['Observed'],
            label='Observed', color='0.08', s=40,
            edgecolor='white', linewidth=0.5, zorder=3, alpha=1)

ax2.scatter(plot_df2.index, plot_df2['Estimated'],
            label='Estimated', color='0.55', s=40,
            edgecolor='white', linewidth=0.5, zorder=2, alpha=1)

ax2.vlines(x=plot_df2.index, ymin=plot_df2['Observed'],
           ymax=plot_df2['Estimated'], color='0.7',
           alpha=0.3, linewidth=1, zorder=1)

# Regression lines
coef_obs2 = np.polyfit(plot_df2.index, plot_df2['Observed'], 1)
poly_obs2 = np.poly1d(coef_obs2)
ax2.plot(plot_df2.index, poly_obs2(plot_df2.index), color='0.2',
         linestyle='--', linewidth=0.8, label='Regression Observed')

coef_est2 = np.polyfit(plot_df2.index, plot_df2['Estimated'], 1)
poly_est2 = np.poly1d(coef_est2)
ax2.plot(plot_df2.index, poly_est2(plot_df2.index), color='0.5',
         linestyle='--', linewidth=0.8, label='Regression Estimated')

ax2.set_title('Observed vs Estimated User Ratings', loc='left', pad=10, weight='bold')
ax2.set_xlabel('Movies')
ax2.set_ylabel('User Rating')

ax2.grid(True, axis='y', linestyle='-', linewidth=0.5, color='0.88')
ax2.grid(False, axis='x')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.legend(frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig(f'out/art/{name}_ObservedEstimated2.svg')
plt.close()