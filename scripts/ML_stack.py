from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import shap 

warnings.filterwarnings("ignore")

# Visual utils (tdqm loading)
class CatBoostTqdmCallback:
    def __init__(self, iterations):
        self.pbar = tqdm(total=iterations)
    def after_iteration(self, info):
        self.pbar.update(1)
        return True
    def __del__(self):
        self.pbar.close()

name = input("Pseudo: ")
df = pd.read_csv(f'out/enriched_movies_' + name +'.csv', sep=None, engine='python')

######      FEATURES ENGINEERING
# Niche, log, controversy, age
df['is_niche'] = np.where(df['total_ratings'] > 500, 0, 1)
for col in ['views', 'likes', 'fans', 'total_ratings']:
    df[col] = np.log1p(df[col])   #### Instead of log, should we use min-max scaling?
df['controversy'] = df['rating_std_dev'] * df['total_ratings']
df['year'] = 2026 - df['year']

# Skewness (instead of D1/D9)
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

# Droping columns we don't need anymore for the model
df = df.drop(columns=[
    'rating_0.5_count', 'rating_1.0_count', 'rating_1.5_count', 
    'rating_2.0_count','rating_2.5_count', 'rating_3.0_count', 
    'rating_3.5_count', 'rating_4.0_count', 'rating_4.5_count',
    'rating_5.0_count', 'country_2', 'country_3', 'genre_2', 
    'genre_3', 'theme_1', 'theme_2', 'D1D9'], errors='ignore')

# Strict 80/20 split between training and holdout data
df_trainval, df_holdout = train_test_split(df, test_size=0.2, random_state=42)
# Keep ~60% for base training and 20% for stacking-aware meta-training
df_train, df_stack = train_test_split(df_trainval, test_size=0.25, random_state=42)
#### Maybe this sampling method isn't good enough 

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

# Formatting datasets
def process_text(dataset):
    temp_df = dataset.copy()
    
    for new_column, old_columns in groups.items():
        existing_cols = [col for col in old_columns if col in temp_df.columns]
        
        for col in existing_cols:
            temp_df[col] = temp_df[col].astype(str).str.replace(' ', '_').replace('nan', '')

        temp_df[new_column] = temp_df[existing_cols].agg(' '.join, axis=1)
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

text_features = list(groups.keys())
cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object' and col not in text_features]

X_stack = X_stack.reindex(columns=X_train.columns, fill_value=-1)
X_holdout = X_holdout.reindex(columns=X_train.columns, fill_value=-1)

for dataset in (X_train, X_stack, X_holdout):
    if cat_features:
        dataset[cat_features] = dataset[cat_features].fillna('None')
    dataset[text_features] = dataset[text_features].fillna('')
    dataset[:] = dataset.fillna(-1)

# Respective weighting for each model
# 20 when normally distributed
# 40 when not a lot of extreme values
# We tried adaptative weighting based on the distance to the mean, but it didn't perform better than a simple static weighting, we kept it simple
weights_abs = y_train_abs.apply(lambda x: 20.0 if x <= 1.0 or x >= 4.5 else 1.0)
weights_delta = y_train_delta.abs().apply(lambda x: 20.0 if x >= 1.5 else 1.0)

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

# Models: settings and fitting
print("Training 3 CatBoost models (1. Absolute, 2. Delta, 3. Like)...\n")
iterations = 2000

# Absolute
cb_abs = CatBoostTqdmCallback(iterations)
model_abs = CatBoostRegressor(
    iterations=iterations, 
    learning_rate=0.005, 
    depth=6, 
    loss_function='MAE', # MAE performs better than RMSE
    verbose=0)
model_abs.fit(pool_train_abs, callbacks=[cb_abs])

# Relative (delta)
cb_delta = CatBoostTqdmCallback(iterations)
model_delta = CatBoostRegressor(
    iterations=iterations,
    learning_rate=0.005,
    depth=6,
    loss_function='MAE',
    verbose=0)
model_delta.fit(pool_train_delta, callbacks=[cb_delta])

# Like
cb_like = CatBoostTqdmCallback(iterations)
model_like = CatBoostClassifier(
    iterations=iterations,
    class_weights=[1,2], # Need to adapt this depending on the user likes / total movies ratio
    learning_rate=0.005,
    depth=6,
    l2_leaf_reg=5,
    #loss_function='Logloss',
    eval_metric='AUC', # F1
    verbose=0)
model_like.fit(pool_train_like, callbacks=[cb_like])


# Prediction
stack_p_abs = model_abs.predict(pool_stack)
stack_p_delta = model_delta.predict(pool_stack) + X_stack['avg_rating'].values
stack_p_like_prob = model_like.predict_proba(pool_stack)[:, 1]

holdout_p_abs = model_abs.predict(pool_holdout)
holdout_p_delta = model_delta.predict(pool_holdout) + X_holdout['avg_rating'].values
holdout_p_like_prob = model_like.predict_proba(pool_holdout)[:, 1]

# Ridge stacking of the 3 models
stack_meta_X = np.column_stack((stack_p_abs, stack_p_delta, stack_p_like_prob))
holdout_meta_X = np.column_stack((holdout_p_abs, holdout_p_delta, holdout_p_like_prob))
meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_meta_X, y_stack_abs)

final_preds = meta_model.predict(holdout_meta_X)


######      RESULTS
# Evaluation
results = pd.DataFrame({
    'Observed': y_holdout_abs.values,
    'Estimated': final_preds
})

print(f"\n\n\n\n")
print(results.describe())
print(f"\n\nMean error: {mean_absolute_error(y_holdout_abs, final_preds):.4f} points.")
print(f"R2 : {r2_score(y_holdout_abs, final_preds):.4f}")
print(f"\nModel weights")
print(f"Absolute: {meta_model.coef_[0]:.4f}")
print(f"Relative: {meta_model.coef_[1]:.4f}")
print(f"Like    : {meta_model.coef_[2]:.4f}\n")

# Clipping for hits and misses
rounded_preds = np.round(final_preds * 2) / 2
rounded_preds = np.clip(rounded_preds, 0.5, 5.0)

hits = np.sum(rounded_preds == y_holdout_abs.values)
total = len(y_holdout_abs)
misses = total - hits
hit_rate = (hits / total) * 100

print(f"Hits: {hits}")
print(f"Misses: {misses}")
print(f"Hit Percentage: {hit_rate:.2f}%")


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
fig = plt.figure(figsize=(12, 9))
sns.heatmap(df.isna().T, cbar=False, yticklabels=True, xticklabels=False, cmap='viridis')
plt.title("Missing Features Heatmap")
plt.savefig('out/art/missing_features.png', dpi=400, bbox_inches='tight')
plt.close()

# SHAP values absolute & relative
# Config
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # Ou une autre police système

explainer = shap.TreeExplainer(model_abs)
shap_values = explainer.shap_values(pool_train_abs)
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig('out/art/SHAPabsolute.png', dpi=400, bbox_inches='tight')
plt.close()

explainer = shap.TreeExplainer(model_delta)
shap_values = explainer.shap_values(pool_train_delta)
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig('out/art/SHAPdelta.png', dpi=400, bbox_inches='tight')
plt.close()

# Top 3 features shap dependence plot
models_to_plot = [
    (model_abs, pool_train_abs, "Absolute"),
    (model_delta, pool_train_delta, "Delta")]

for model, pool, label in models_to_plot:
    importances = model.get_feature_importance()
    feature_names = X_train.columns
    top_3_features = [name for _, name in sorted(zip(importances, feature_names), reverse=True)[:3]]
    explainer = shap.TreeExplainer(model)
    shap_v = explainer.shap_values(pool)
    
    for feature in top_3_features:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feature, shap_v, X_train, show=False)
        plt.title(f"SHAP Dependence - {label} Model - {feature}")
        plt.yticks(fontsize=11)
        plt.xticks(fontsize=11)
        plt.tight_layout()
        plt.savefig(f'out/art/{label}_{feature}_SHAP_dependence.svg')

# Observed vs Estimated plot
plot_df = pd.DataFrame({
    'Observed': y_holdout_abs.values,'Estimated': final_preds
}).sort_values(by='Observed').reset_index(drop=True)
plt.figure(figsize=(12, 6))

y_min = np.where(plot_df['Observed'] == 0.5, 0, plot_df['Observed'] - 0.25)
y_max = plot_df['Observed'] + 0.25

heights = y_max - y_min
plt.bar(x=plot_df.index, height=heights, bottom=y_min, width=1.0,
        color='lightgrey', alpha=0.5, edgecolor='none', zorder=0, 
        label='Tolerance Zone')

sns.scatterplot(
    x=plot_df.index, y=plot_df['Observed'],
    label='Observed', color='grey', s=40,
    edgecolor='dimgrey', zorder=3, alpha=1)

sns.scatterplot(
    x=plot_df.index, y=plot_df['Estimated'],
    label='Estimated', color='lightgrey', s=40,
    edgecolor='grey', zorder=2, alpha=1)

plt.vlines(
    x=plot_df.index, ymin=plot_df['Observed'],
    ymax=plot_df['Estimated'], color='grey',
    alpha=0.3, linewidth=1, zorder=1)

# Regression lines
coef_obs = np.polyfit(plot_df.index, plot_df['Observed'], 1)
poly_obs = np.poly1d(coef_obs)
plt.plot(plot_df.index, poly_obs(plot_df.index), color='black',
    linestyle='--', linewidth=0.8, label='Regression Observed')

coef_est = np.polyfit(plot_df.index, plot_df['Estimated'], 1)
poly_est = np.poly1d(coef_est)
plt.plot(plot_df.index, poly_est(plot_df.index), color='grey',
         linestyle='--', linewidth=0.8, label='Regression Estimated')

plt.title('Estimated vs Observed User Ratings')
plt.xlabel('Movies')
plt.ylabel('User Rating')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
sns.despine()
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('out/art/ObservedEstimated.svg')

plot_df = pd.DataFrame({
    'Observed': y_holdout_abs.values,
    'Estimated': final_preds
}).sort_values(by='Estimated').reset_index(drop=True)
plt.figure(figsize=(12, 6))

sns.scatterplot(
    x=plot_df.index, y=plot_df['Observed'],
    label='Observed', color='grey', s=40,
    edgecolor='dimgrey', zorder=3, alpha=1)

sns.scatterplot(
    x=plot_df.index, y=plot_df['Estimated'],
    label='Estimated', color='lightgrey', s=40,
    edgecolor='grey', zorder=2, alpha=1)

plt.vlines(
    x=plot_df.index, ymin=plot_df['Observed'],
    ymax=plot_df['Estimated'], color='grey',
    alpha=0.3, linewidth=1, zorder=1)

# Regression lines
coef_obs = np.polyfit(plot_df.index, plot_df['Observed'], 1)
poly_obs = np.poly1d(coef_obs)
plt.plot(plot_df.index, poly_obs(plot_df.index), color='black',
    linestyle='--', linewidth=0.8, label='Regression Observed')

coef_est = np.polyfit(plot_df.index, plot_df['Estimated'], 1)
poly_est = np.poly1d(coef_est)
plt.plot(plot_df.index, poly_est(plot_df.index), color='grey',
         linestyle='--', linewidth=0.8, label='Regression Estimated')

plt.title('Observed vs Estimated User Ratings')
plt.xlabel('Movies')
plt.ylabel('User Rating')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
sns.despine()
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('out/art/ObservedEstimated2.svg')

# Residual analysis
residuals = pd.DataFrame({
    'Title': df_holdout['title'].values,
    'Observed': y_holdout_abs.values,
    'Estimated': final_preds})

residuals['Delta'] = residuals['Estimated'] - residuals['Observed']
residuals['Abs_Delta'] = residuals['Delta'].abs()

best = residuals.nsmallest(20, 'Abs_Delta')
worst = residuals.nlargest(20, 'Abs_Delta')

print(f"\n\n\n{'min(Delta)':<45} {'max(Delta)':<45}")
print("-" * 90)

for i in range(20):
    b_row = best.iloc[i]
    b_txt = f"{b_row['Title'][:30]:<30} {b_row['Delta']:+0.3f}"
    w_row = worst.iloc[i]
    w_txt = f"{w_row['Title'][:30]:<30} {w_row['Delta']:+0.3f}"

    print(f"{b_txt:<45} {w_txt:<45}")