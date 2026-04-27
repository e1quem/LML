from catboost import CatBoostRegressor, CatBoostClassifier, Pool, CatBoostError
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import Ridge
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import os
import sys

warnings.filterwarnings("ignore")

class CatBoostTqdmCallback:
    def __init__(self, pbar):
        self.pbar = pbar
    def after_iteration(self, info):
        self.pbar.update(1)
        sys.stdout.flush() # test
        return True
    
def calculate_skew(row):
    counts = [row.get(f'rating_{r}_count', 0) for r in ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']]
    ratings = np.repeat(np.arange(0.5, 5.5, 0.5), counts)
    if len(ratings) < 2: return 0
    return pd.Series(ratings).skew()

def heatmap_correlation_movies(df, name='user_rating'):
    all_numeric_vars = ['user_rating', 'user_like', 'year_user', 'views', 'likes', 'fans', 'avg_rating', 'total_ratings', 'duration_mins', 'rating_std_dev', 'rating_ratio', 'like_view_ratio', 'is_niche', 'controversy', 'rating_skew']
    all_numeric_vars += [f'rating_{r}_count' for r in ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']]
    existing_vars = [var for var in all_numeric_vars if var in df.columns]
    df_numeric = df[existing_vars].apply(pd.to_numeric, errors='coerce')
    corr = df_numeric.corr(method='pearson').round(3)
    return corr['user_rating'].to_dict()

# Defining user average per thematics, with a leave-one-out mechanism to avoid data-leakage
def target_encode_strict(train_df, test_df, cols, target):
    melted = train_df.melt(id_vars=[target], value_vars=cols).dropna()
    melted = melted[melted['value'] != '']
    stats = melted.groupby('value')[target].agg(['sum', 'count'])
    global_mean = train_df[target].mean()

    def get_loo_train(row):
        entities = [row[c] for c in cols if pd.notna(row[c]) and row[c] != '']
        row_loo = [(stats.loc[e, 'sum'] - row[target]) / (stats.loc[e, 'count'] - 1) for e in entities if e in stats.index and stats.loc[e, 'count'] > 1]
        return np.mean(row_loo) if row_loo else global_mean

    def get_mean_test(row):
        entities = [row[c] for c in cols if pd.notna(row[c]) and row[c] != '']
        row_means = [stats.loc[e, 'sum'] / stats.loc[e, 'count'] for e in entities if e in stats.index]
        return np.mean(row_means) if row_means else global_mean

    return train_df.apply(get_loo_train, axis=1), test_df.apply(get_mean_test, axis=1)

# Fusioning textual groups, so that actor_1 and actor_5 both count equally as "actors", with no hierarchy
groups = {
    'actors': [f'actor_{i}' for i in range(1, 6)],
    'studios': [f'studio_{i}' for i in range(1, 3)],
    'directors': [f'director_{i}' for i in range(1, 4)],
    'producers': [f'producer_{i}' for i in range(1, 4)]
}   

# Cleaning data and verifying types
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


# Data loading (loop)
pseudos_input = input("Letterboxd pseudos list (separated by commas): ")
usernames = [u.strip() for u in pseudos_input.split(",") if u.strip()]

for name in usernames:
    print(f"\n  {name}...")
    user_file = f'out/movies_{name}.csv'

    if not os.path.exists(user_file):
        print(f"File {user_file} not found o_O?")
        continue

    user_ratings = pd.read_csv(user_file, sep=None, engine='python')
    metadata = pd.read_csv('out/movies.csv', sep=None, engine='python')
    df = user_ratings.merge(metadata, on='url', how='left')

    # Renaming duplicate columns
    df.rename(columns={'user_rating_x': 'user_rating', 'user_like_x': 'user_like', 'year_x': 'year_user'}, inplace=True)
    df = df.dropna(subset=['avg_rating', 'total_ratings', 'rating_std_dev'])

    # Missing values
    initial_count = len(df)
    df = df.dropna(subset=['avg_rating', 'total_ratings', 'rating_std_dev'])
    if len(df) < initial_count:
        print(f"Warning: {initial_count - len(df)} movies dropped due to missing metadata")

    # Features engineering
    df['is_niche'] = np.where(df['total_ratings'] > 500, 0, 1)
    for col in ['views', 'likes', 'fans', 'total_ratings']:
        df[col] = np.log1p(df[col])
    df['controversy'] = df['rating_std_dev'] * df['total_ratings']
    df['year'] = 2026 - df['year_y']
    df['rating_skew'] = df.apply(calculate_skew, axis=1)

    corr_results = heatmap_correlation_movies(df, name)
    corr_dict = {f'corr_{k}': v for k, v in corr_results.items() if k != 'user_rating'}

    # Droping columns
    df = df.drop(columns=['rating_0.5_count', 'rating_1.0_count', 'rating_1.5_count', 'rating_2.0_count','rating_2.5_count', 'rating_3.0_count', 'rating_3.5_count', 'rating_4.0_count', 'rating_4.5_count', 'rating_5.0_count', 'title_x', 'year_x','year_y', 'url', 'title_y'], errors='ignore')

    # Models
    # Isolation variables initialization per user
    oof_preds = np.zeros(len(df))
    oof_like_probs = np.zeros(len(df))
    meta_coefs = []

    has_likes = df['user_like'].sum() >= 5
    if not has_likes:
        print(f"Not enough likes ({int(df['user_like'].sum())}), stratification on ratings and 2 models stacking")
    
    strat_target = df['user_like'] if has_likes else (df['user_rating'] * 2).astype(int)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    global_pbar = tqdm(total=30000, desc=f"Training")
    
    for fold, (trainval_idx, holdout_idx) in enumerate(kf.split(df, strat_target)):
        df_trainval = df.iloc[trainval_idx].copy()
        df_holdout = df.iloc[holdout_idx].copy()
        local_strat = strat_target.iloc[trainval_idx]
        df_train, df_stack = train_test_split(df_trainval, test_size=0.25, random_state=42, stratify=local_strat)

        # Target Encoding intra-fold
        for enc_cols, col_name in [(['director_1', 'director_2', 'director_3'], 'user_dir_avg'), (['studio_1', 'studio_2'], 'user_studio_avg'), (['writer_1'], 'user_writer_avg'), (['genre_1'], 'user_genre_avg')]:
            df_train[col_name], df_stack[col_name] = target_encode_strict(df_train, df_stack, enc_cols, 'user_rating')
            _, df_holdout[col_name] = target_encode_strict(df_train, df_holdout, enc_cols, 'user_rating')

        df_train_proc = process_text(df_train)
        df_stack_proc = process_text(df_stack)
        df_holdout_proc = process_text(df_holdout)

        drop_cols = ['title', 'user_rating', 'user_like']
        X_train = df_train_proc.drop(columns=drop_cols, errors='ignore')
        X_stack = df_stack_proc.drop(columns=drop_cols, errors='ignore')
        X_holdout = df_holdout_proc.drop(columns=drop_cols, errors='ignore')
        
        y_train_abs = df_train['user_rating']
        y_stack_abs = df_stack['user_rating']
        y_train_delta = y_train_abs - df_train['avg_rating']
        y_train_like = df_train['user_like']

        # Data type detection and cleaning
        explicit_cat = ['primary_language', 'country_1', 'genre_1', 'writer_1', 'user_dir_avg', 'user_studio_avg', 'user_writer_avg', 'user_genre_avg']
        for dataset in (X_train, X_stack, X_holdout):
            for col in explicit_cat:
                if col in dataset.columns:
                    dataset[col] = dataset[col].fillna('None').astype(str).astype('category')

        text_features = list(groups.keys())
        text_features = [col for col in text_features if X_train[col].astype(str).str.strip().ne('').any()]
        cat_features = [col for col in X_train.columns if (X_train[col].dtype.name == 'category' or col in explicit_cat) and col not in text_features]

        # Weights
        weights_abs = y_train_abs.apply(lambda x: 20.0 if x <= 1.0 or x >= 4.5 else 1.0)
        weights_delta = y_train_delta.abs().apply(lambda x: 20.0 if x >= 1.5 else 1.0)
        like_ratio = (len(y_train_like) - y_train_like.sum()) / y_train_like.sum() if y_train_like.sum() > 0 else 1.0

        try:
            p_train_abs = Pool(X_train, y_train_abs, cat_features=cat_features, text_features=text_features, weight=weights_abs)
            p_train_delta = Pool(X_train, y_train_delta, cat_features=cat_features, text_features=text_features, weight=weights_delta)
            p_train_like = Pool(X_train, y_train_like, cat_features=cat_features, text_features=text_features)
            p_stack = Pool(X_stack, cat_features=cat_features, text_features=text_features)
            p_holdout = Pool(X_holdout, cat_features=cat_features, text_features=text_features)

            # Fitting, stacking meta-predictions, holdout-predictions (OOF)
            cb_global = CatBoostTqdmCallback(global_pbar)

            model_abs = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(p_train_abs, callbacks=[cb_global])
            s_p_abs = model_abs.predict(p_stack)
            h_p_abs = model_abs.predict(p_holdout)

            model_delta = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(p_train_delta, callbacks=[cb_global])
            s_p_delta = model_delta.predict(p_stack) + X_stack['avg_rating'].values
            h_p_delta = model_delta.predict(p_holdout) + X_holdout['avg_rating'].values

            # Conditional on sufficient likes
            if has_likes:
                like_ratio = (len(y_train_like) - y_train_like.sum()) / y_train_like.sum() if y_train_like.sum() > 0 else 1.0
                model_like = CatBoostClassifier(iterations=2000, learning_rate=0.005, depth=6, class_weights=[1, like_ratio], verbose=0).fit(p_train_like, callbacks=[cb_global])
                s_p_like = model_like.predict_proba(p_stack)[:, 1]
                h_p_like = model_like.predict_proba(p_holdout)[:, 1]
            else:
                s_p_like = np.zeros(len(df_stack))
                h_p_like = np.zeros(len(df_holdout))
                global_pbar.update(2000)

        except CatBoostError:
            text_features = []
            cat_features_fb = list(dict.fromkeys(cat_features + ['actors', 'studios', 'directors', 'producers']))

            p_train_abs = Pool(X_train, y_train_abs, cat_features=cat_features_fb, text_features=text_features, weight=weights_abs)
            p_train_delta = Pool(X_train, y_train_delta, cat_features=cat_features_fb, text_features=text_features, weight=weights_delta)
            p_train_like = Pool(X_train, y_train_like, cat_features=cat_features_fb, text_features=text_features)
            p_stack = Pool(X_stack, cat_features=cat_features_fb, text_features=text_features)
            p_holdout = Pool(X_holdout, cat_features=cat_features_fb, text_features=text_features)

            cb_global = CatBoostTqdmCallback(global_pbar)

            model_abs = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(p_train_abs, callbacks=[cb_global])
            s_p_abs = model_abs.predict(p_stack)
            h_p_abs = model_abs.predict(p_holdout)

            model_delta = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(p_train_delta, callbacks=[cb_global])
            s_p_delta = model_delta.predict(p_stack) + X_stack['avg_rating'].values
            h_p_delta = model_delta.predict(p_holdout) + X_holdout['avg_rating'].values

            if has_likes:
                like_ratio = (len(y_train_like) - y_train_like.sum()) / y_train_like.sum() if y_train_like.sum() > 0 else 1.0
                model_like = CatBoostClassifier(iterations=2000, learning_rate=0.005, depth=6, class_weights=[1, like_ratio], verbose=0).fit(p_train_like, callbacks=[cb_global])
                s_p_like = model_like.predict_proba(p_stack)[:, 1]
                h_p_like = model_like.predict_proba(p_holdout)[:, 1]
            else:
                s_p_like = np.zeros(len(df_stack))
                h_p_like = np.zeros(len(df_holdout))
                global_pbar.update(2000)
                
        X_meta_stack = np.column_stack((s_p_abs, s_p_delta, s_p_like))
        X_meta_holdout = np.column_stack((h_p_abs, h_p_delta, h_p_like))

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_meta_stack, y_stack_abs)

        oof_preds[holdout_idx] = meta_model.predict(X_meta_holdout)
        oof_like_probs[holdout_idx] = h_p_like
        meta_coefs.append(meta_model.coef_)

    global_pbar.close()

    # Mean weights
    avg_meta_coefs = np.mean(meta_coefs, axis=0)
    final_preds = np.clip(oof_preds, 0.5, 5.0)

    # Results
    y_true = df['user_rating'].values
    rounded_preds = np.round(final_preds * 2) / 2
    hit_rate_exact = (np.sum(rounded_preds == y_true) / len(y_true)) * 100
    hit_rate_05 = (np.sum(np.abs(final_preds - y_true) <= 0.5) / len(y_true)) * 100
    hit_rate_10 = (np.sum(np.abs(final_preds - y_true) <= 1.0) / len(y_true)) * 100

    bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    hr_summary = {}
    for r in bins:
        low, high = (0, 0.75) if r == 0.5 else ((4.75, 5.5) if r == 5.0 else (r-0.25, r+0.25))
        mask = (y_true == r)
        hr_val = f"{np.sum((final_preds[mask] >= low) & (final_preds[mask] <= high))/mask.sum()*100:.1f}%" if mask.any() else "N/A"
        hr_summary[f"hr_{int(r*10)}"] = hr_val

    thresholds = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
    f1_dict = {}
    for t in thresholds:
        y_pred_l = (oof_like_probs >= t).astype(int)
        tp = np.sum((df['user_like'] == 1) & (y_pred_l == 1))
        fp = np.sum((df['user_like'] == 0) & (y_pred_l == 1))
        fn = np.sum((df['user_like'] == 1) & (y_pred_l == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_dict[f"like_F1_{int(t*100)}"] = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    std_diff = y_true.std() - pd.Series(final_preds).std()
    
    # Exports
    results_summary = pd.DataFrame({
        'pseudo': [name], 'has_like': [int(has_likes)], 'followers': [''], 'rated_movies': [len(user_ratings)],
        'observations': [len(y_true)], 'R2': [round(r2_score(y_true, final_preds), 4)],
        'mean_error': [round(mean_absolute_error(y_true, final_preds), 4)],
        'hit_rate': [round(hit_rate_exact, 2)], 'hit_rate±05': [round(hit_rate_05, 2)], 'hit_rate±1': [round(hit_rate_10, 2)],
        **hr_summary, 'std_diff': [round(std_diff, 4)], **f1_dict,
        'absolute_w': [round(avg_meta_coefs[0], 4)], 'relative_w': [round(avg_meta_coefs[1], 4)], 'like_w': [round(avg_meta_coefs[2], 4)],
        **corr_dict
    })


    if not os.path.exists('out/performance.csv') or os.path.getsize('out/performance.csv') == 0: results_summary.head(0).to_csv('out/performance.csv', index=False)
    results_summary.to_csv('out/performance.csv', mode='a', index=False, header=False)

    obs_df = pd.DataFrame({'pseudo': name, 'observed_rating': y_true, 'predicted_rating': final_preds})
    if not os.path.exists('out/observations.csv') or os.path.getsize('out/observations.csv') == 0: obs_df.head(0).to_csv('out/observations.csv', index=False)
    obs_df.to_csv('out/observations.csv', mode='a', index=False, header=False)

## Data loading
#name = input("Pseudo: ")
#user_ratings = pd.read_csv(f'out/movies_{name}.csv', sep=None, engine='python')
#metadata = pd.read_csv('out/movies.csv', sep=None, engine='python')
#df = user_ratings.merge(metadata, on='url', how='left')
#
## Renaming duplicate columns
#df.rename(columns={
#    'user_rating_x': 'user_rating',
#    'user_like_x': 'user_like',
#    'year_x': 'year_user',
#}, inplace=True)
#
## Missing values
#initial_count = len(df)
#df = df.dropna(subset=['avg_rating', 'total_ratings', 'rating_std_dev'])
#if len(df) < initial_count:
#    print(f"Warning: {initial_count - len(df)} movies dropped due to missing metadata")
#
#
####### FEATURES ENGINEERING
#df['is_niche'] = np.where(df['total_ratings'] > 500, 0, 1)
#for col in ['views', 'likes', 'fans', 'total_ratings']:
#    df[col] = np.log1p(df[col])
#df['controversy'] = df['rating_std_dev'] * df['total_ratings']
#df['year'] = 2026 - df['year_y']
#df['rating_skew'] = df.apply(calculate_skew, axis=1)
#
## Calculating correlations for numeric values
#corr_results = heatmap_correlation_movies(df, name)
#corr_dict = {f'corr_{k}': v for k, v in corr_results.items() if k != 'user_rating'}
#
## Droping columns
#df = df.drop(columns=['rating_0.5_count', 'rating_1.0_count', 'rating_1.5_count', 'rating_2.0_count','rating_2.5_count', 'rating_3.0_count', 'rating_3.5_count', 'rating_4.0_count', 'rating_4.5_count', 'rating_5.0_count', 'title_x', 'year_x','year_y', 'url', 'title_y'], errors='ignore')
#
#
#
####### MODELS
#oof_preds = np.zeros(len(df))
#oof_like_probs = np.zeros(len(df))
#meta_coefs = []
#
#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#global_pbar = tqdm(total=30000, desc="\n5-fold training", file=sys.stdout)
#
#for fold, (trainval_idx, holdout_idx) in enumerate(kf.split(df, df['user_like'])):
#
#    df_trainval = df.iloc[trainval_idx].copy()
#    df_holdout = df.iloc[holdout_idx].copy()
#    df_train, df_stack = train_test_split(df_trainval, test_size=0.25, random_state=42, stratify=df_trainval['user_like'])
#
#    # Target Encoding intra-fold
#    for cols_set in [(['director_1', 'director_2', 'director_3'], 'user_dir_avg'),
#                     (['studio_1', 'studio_2'], 'user_studio_avg'),
#                     (['writer_1'], 'user_writer_avg'),
#                     (['genre_1'], 'user_genre_avg')]:
#        col_name = cols_set[1]
#        enc_cols = cols_set[0]
#        df_train[col_name], df_stack[col_name] = target_encode_strict(df_train, df_stack, enc_cols, 'user_rating')
#        _, df_holdout[col_name] = target_encode_strict(df_train, df_holdout, enc_cols, 'user_rating')
#
#    df_train_proc = process_text(df_train)
#    df_stack_proc = process_text(df_stack)
#    df_holdout_proc = process_text(df_holdout)
#
#    drop_cols = ['title', 'url', 'user_rating', 'user_like', 'year_y']
#    X_train = df_train_proc.drop(columns=drop_cols, errors='ignore')
#    X_stack = df_stack_proc.drop(columns=drop_cols, errors='ignore')
#    X_holdout = df_holdout_proc.drop(columns=drop_cols, errors='ignore')
#    
#    y_train_abs = df_train['user_rating']
#    y_stack_abs = df_stack['user_rating']
#    y_train_delta = y_train_abs - df_train['avg_rating']
#    y_train_like = df_train['user_like']
#
#    # Data type detection and cleaning
#    explicit_cat = ['primary_language', 'country_1', 'genre_1', 'writer_1', 
#                    'user_dir_avg', 'user_studio_avg', 'user_writer_avg', 'user_genre_avg']
#    
#    for dataset in (X_train, X_stack, X_holdout):
#        for col in explicit_cat:
#            if col in dataset.columns:
#                dataset[col] = dataset[col].fillna('None').astype(str).astype('category')
#
#    text_features = list(groups.keys())
#    cat_features = [col for col in X_train.columns 
#                    if (X_train[col].dtype.name == 'category' or col in explicit_cat) and col not in text_features]
#
#    # Weights
#    weights_abs = y_train_abs.apply(lambda x: 20.0 if x <= 1.0 or x >= 4.5 else 1.0)
#    weights_delta = y_train_delta.abs().apply(lambda x: 20.0 if x >= 1.5 else 1.0)
#    like_ratio = (len(y_train_like) - y_train_like.sum()) / y_train_like.sum() if y_train_like.sum() > 0 else 1.0
#
#    # Pools
#    p_train_abs = Pool(X_train, y_train_abs, cat_features=cat_features, text_features=text_features, weight=weights_abs)
#    p_train_delta = Pool(X_train, y_train_delta, cat_features=cat_features, text_features=text_features, weight=weights_delta)
#    p_train_like = Pool(X_train, y_train_like, cat_features=cat_features, text_features=text_features)
#    p_stack = Pool(X_stack, cat_features=cat_features, text_features=text_features)
#    p_holdout = Pool(X_holdout, cat_features=cat_features, text_features=text_features)
#
#    # Fitting
#    cb_global = CatBoostTqdmCallback(global_pbar)
#    model_abs = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(p_train_abs, callbacks=[cb_global])
#    model_delta = CatBoostRegressor(iterations=2000, learning_rate=0.005, depth=6, loss_function='MAE', verbose=0).fit(p_train_delta, callbacks=[cb_global])
#    model_like = CatBoostClassifier(iterations=2000, learning_rate=0.005, depth=6, class_weights=[1, like_ratio], verbose=0).fit(p_train_like, callbacks=[cb_global])
#
#    # Meta-predictions (Stacking)
#    s_p_abs = model_abs.predict(p_stack)
#    s_p_delta = model_delta.predict(p_stack) + X_stack['avg_rating'].values
#    s_p_like = model_like.predict_proba(p_stack)[:, 1]
#
#    # Holdout-predictions (OOF)
#    h_p_abs = model_abs.predict(p_holdout)
#    h_p_delta = model_delta.predict(p_holdout) + X_holdout['avg_rating'].values
#    h_p_like = model_like.predict_proba(p_holdout)[:, 1]
#
#    # Meta-Model Ridge
#    meta_model = Ridge(alpha=1.0)
#    meta_model.fit(np.column_stack((s_p_abs, s_p_delta, s_p_like)), y_stack_abs)
#    
#    oof_preds[holdout_idx] = meta_model.predict(np.column_stack((h_p_abs, h_p_delta, h_p_like)))
#    oof_like_probs[holdout_idx] = h_p_like
#    meta_coefs.append(meta_model.coef_)
#
#global_pbar.close()
#
## Mean weights
#avg_meta_coefs = np.mean(meta_coefs, axis=0)
#final_preds = np.clip(oof_preds, 0.5, 5.0)
#
#
####### RESULTS
#y_true = df['user_rating'].values
#
#results = pd.DataFrame({
#    'Observed': y_true,
#    'Estimated': final_preds
#})
#
#print(f"\n  Global results:\n")
#print(results.describe())
#print("\nAverage on 5-folds:")
#print(f"Mean error (MAE): {mean_absolute_error(y_true, final_preds):.4f} points")
#print(f"Mean R2 score  : {r2_score(y_true, final_preds):.4f}")
#print(f"Absolute weight: {avg_meta_coefs[0]:.4f}\nRelative weight: {avg_meta_coefs[1]:.4f}\nLike weight    : {avg_meta_coefs[2]:.4f}\n")
#
## Hit rate logic
#rounded_preds = np.round(final_preds * 2) / 2
#hit_rate_exact = (np.sum(rounded_preds == y_true) / len(y_true)) * 100
#hit_rate_05 = (np.sum(np.abs(final_preds - y_true) <= 0.5) / len(y_true)) * 100
#hit_rate_10 = (np.sum(np.abs(final_preds - y_true) <= 1.0) / len(y_true)) * 100
#
#hit_table = pd.DataFrame({
#    'Exact': [f"{hit_rate_exact:.2f}%"], '±0.5': [f"{hit_rate_05:.2f}%"], '±1.0': [f"{hit_rate_10:.2f}%"]
#}, index=['Hit rate (avg)'])
#print(hit_table)
#
## Hit rate by rating
#bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
#hr_summary = {}
#for r in bins:
#    low, high = (0, 0.75) if r == 0.5 else ((4.75, 5.5) if r == 5.0 else (r-0.25, r+0.25))
#    mask = (y_true == r)
#    hr_val = f"{np.sum((final_preds[mask] >= low) & (final_preds[mask] <= high))/mask.sum()*100:.1f}%" if mask.any() else "N/A"
#    hr_summary[f"hr_{int(r*10)}"] = hr_val
#
#print("\nRating  : ", "  ".join([str(r) for r in bins]))
#print("Hit rate: ", "  ".join([hr_summary.get(f"hr_{int(r*10)}", "N/A") for r in bins]))
#
## Like model F1 (OOF)
#thresholds = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
#f1_dict = {}
#for t in thresholds:
#    y_pred_l = (oof_like_probs >= t).astype(int)
#    tp = np.sum((df['user_like'] == 1) & (y_pred_l == 1))
#    fp = np.sum((df['user_like'] == 0) & (y_pred_l == 1))
#    fn = np.sum((df['user_like'] == 1) & (y_pred_l == 0))
#    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
#    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
#    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
#    f1_dict[f"like_F1_{int(t*100)}"] = f1
#
#f1_df = pd.DataFrame([thresholds, list(f1_dict.values())], index=['Threshold', 'F1-score'])
#print(f"\nLike Model F1 Scores:\n{f1_df.to_string(float_format='%.4f', index=True)}")
#
##### EXPORTING RESULTS
#std_diff = y_true.std() - pd.Series(final_preds).std()
#
#results_summary = pd.DataFrame({
#    'pseudo': [name], 'followers': [''],
#    'rated_movies': [len(user_ratings)],
#    'observations': [len(y_true)],
#    'R2': [r2_score(y_true, final_preds)],
#    'mean_error': [mean_absolute_error(y_true, final_preds)],
#    'hit_rate': [hit_rate_exact], 'hit_rate±05': [hit_rate_05], 'hit_rate±1': [hit_rate_10],
#    **hr_summary, 'std_diff': [std_diff], **f1_dict,
#    'absolute_w': [avg_meta_coefs[0]], 'relative_w': [avg_meta_coefs[1]], 'like_w': [avg_meta_coefs[2]],
#    **corr_dict
#})
#
#results_summary.to_csv('out/performance.csv', mode='a', index=False, header=(not os.path.exists('out/performance.csv') or os.path.getsize('out/performance.csv') == 0))
#
## Saving OOF Observations
#obs_df = pd.DataFrame({
#    'pseudo': name,
#    'observed_rating': y_true,
#    'predicted_rating': final_preds
#})
#obs_df.to_csv('out/observations.csv', mode='a', header=not os.path.exists('out/observations.csv'), index=False)