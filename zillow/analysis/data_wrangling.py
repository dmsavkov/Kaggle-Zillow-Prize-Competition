import os
from tqdm.auto import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .adv_cv import init_adv_cv
from ..features.cleaning import handle_cat_and_missing_quick

# default lgbm because it is more robust
def find_adv_missing_per_col(initial_df, res_path, model_name='lgbm', seed=None):
    path = res_path / f'{model_name}_classify_missingness_per_column.csv'
    if os.path.exists(path):
        print(f"File already exists. Loading it...")
        return pd.read_csv(path, index_col='feature')

    adv_cv = init_adv_cv(model_name, seed=seed, verbose=0,
                         max_iter=300, n_jobs=-1)
    missing_cols = initial_df.columns[initial_df.isna().any()].tolist()
    results = {}

    if model_name == 'logistic':
        df = handle_cat_and_missing_quick(initial_df)
    else:
        df = initial_df

    for col in tqdm(missing_cols):
        df1 = df[initial_df[col].notna()].drop(columns=[col])
        df2 = df[initial_df[col].isna()].drop(columns=[col])

        assert df1.shape[0] + df2.shape[0] == df.shape[0], "Data split by missing values doesn't match original data shape."

        try:
            fold_res = adv_cv.make_cv(df1, df2)
            results[col] = fold_res
        except ValueError as e:  # can be one sample in a set
            print(e)
            results[col] = np.tile([np.nan], 5)
    
    return handle_adv_missing_results(results, path)

def handle_adv_missing_results(results, path):
    res_df = pd.DataFrame(results.values(), columns=[f'fold{i}' for i in range(1, 6)], 
                    index=results.keys())
    res_df['mean'] = res_df.mean(axis=1)
    res_df['std'] = res_df.std(axis=1)
    res_df = np.round(res_df, 2)
    res_df.to_csv(path, index_label='feature')
    return res_df

def find_missing_corrs_per_col(df, res_path):
    df = df.copy()
    res_path = res_path / 'missingness_correlations.csv'
    if os.path.exists(res_path):
        print(f"File already exists. Loading it...")
        return pd.read_csv(res_path, index_col='feature')
    
    imputed_train = handle_cat_and_missing_quick(df)
    missing_cols = df.columns[df.copy().isna().any(axis=0)]
    corr = defaultdict(list)
    pvalues = defaultdict(list)

    for col in tqdm(missing_cols):
        # qq Is it a good strategy to flag and then find corr or I could just use df.corr method? How nans impact correlation?
        # Only the current missing column is casted to boolean!
        cur_train = imputed_train.drop(columns=col).copy()
        col_is_missing = df[col].notna()

        # qq What is the best way to find correlation between a series and a dataframe?
        # corr = cur_train.corrwith(col_is_missing, method='pearson', )

        for original_col in cur_train.columns:
            if imputed_train[original_col].nunique(dropna=True) == 1: 
                pass
            
            if col == original_col:
                r, p = np.nan, np.nan
            else:
                try:
                    r, p = stats.pointbiserialr(col_is_missing, imputed_train[original_col])
                except ValueError as e:
                    print(f"Error calculating point biserial correlation for {original_col}: {e}")
                    r, p = np.nan, np.nan
                
            corr[f"{original_col}_r"].append(r)
            pvalues[f"{original_col}_p"].append(p)
        
    data = {
        **corr, **pvalues
    }
    res_df = pd.DataFrame(data.values(), columns=missing_cols, index=data.keys())

    res_r, res_p = split_rp_from_miss_corr_df(res_df)
    res_r = res_r.abs()

    #res_df.loc['mean_r', :] = res_r.mean(axis=0)
    #res_df.loc['mean_p', :] = res_p.mean(axis=0)
    #res_df.loc['median_r', :] = res_r.median(axis=0)
    #res_df.loc['median_p', :] = res_p.median(axis=0)

    res_df = np.round(res_df, 3)
    res_df.to_csv(res_path, index_label='feature')
    return res_df

def split_rp_from_miss_corr_df(miss_corr:pd.DataFrame):
    r = miss_corr.loc[miss_corr.index.str.endswith('_r'), :]
    p = miss_corr.loc[miss_corr.index.str.endswith('_p'), :]
    return r, p

def plot_miss_corrs(miss_corrs):
    miss_corrs_r, miss_corrs_p = split_rp_from_miss_corr_df(miss_corrs)
    fig, axes = plt.subplots(1, 2, figsize=(28, 16))
    for i, (ax, (lbl, df)) in enumerate(zip(axes, [('r', miss_corrs_r), ('p', miss_corrs_p)])):
        sns.heatmap(df, ax=ax, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title(f"Missingness Correlation - {lbl.upper()}", fontsize=25)
        ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.show()

