import os
import gc
from tqdm.auto import tqdm
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from zillow.config.feature_engineering import FeaturesDtypeConversionConfig_v1
from zillow.utils.common import read_data, find_shared_cols, find_unshared_cols
from zillow.config.config import load_config_no_wrap, create_config_from_dict, merge_configs
from zillow.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, REPORTS_DIR, ANALYSIS_RESULTS_DIR
from zillow.analysis.adv_cv import AdversarialCV
from zillow.features.cleaning import initial_X_clean

cfg = load_config_no_wrap('default')

np.random.seed(cfg.RSEED)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

zillow_dictionary = pd.read_csv(RAW_DATA_DIR / "zillow_data_dictionary.csv")

properties_2016 = read_data(path=INTERIM_DATA_DIR / "cleaned_properties_2016_v1.0.parquet", dtype='default')
properties_2017 = read_data(path=INTERIM_DATA_DIR / "cleaned_properties_2017_v1.0.parquet", dtype='default')
train_2016 = read_data(path=INTERIM_DATA_DIR / "cleaned_train_2016_v1.0.parquet", dtype='default')
train_2017 = read_data(path=INTERIM_DATA_DIR / "cleaned_train_2017_v1.0.parquet", dtype='default')
train = read_data(path=INTERIM_DATA_DIR / "cleaned_train_v1.0.parquet", dtype='default')

features_dtype_cfg = FeaturesDtypeConversionConfig_v1()

# %% [markdown]
# ### Datasets similarity
# 
# Major datsets:
# 1. train 2016, train 2017
# 2. quarter4 train2016, train without q4
# 3. months quarter4 train2016 
# 4. train, properties 2016, 2017 without train
# 5. properties 2016, 2017
# 6. train 2016, properties 2016 without train

# %%
# TODO: creating a class?
# Drag kwargs to adv.

def handle_adv_computing(datasets_dict: dict, model_name: str = "lgbm"):
    path = f"{ANALYSIS_RESULTS_DIR}/advcv_{model_name}_datasets_pairwise_similarity_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    
    all_dfs = []
    for title, dfs in tqdm(datasets_dict.items()):
        res = handle_adv_dfs(dfs)
        pairs = list(combinations(range(len(dfs)), 2))
        col_names = [f"{title}_{i}{j}" for i, j in pairs]
        df_temp = pd.DataFrame(np.round(res, 4)).T
        df_temp.columns = col_names
        all_dfs.append(df_temp)

    adv_results_df = pd.concat(all_dfs, axis=1)
    adv_results_df.to_csv(path)
    return adv_results_df

def handle_adv_dfs(datasets):
    prepared_datasets = []
    leaking_cols = ['assessmentyear', 'transactiondate', 
                    *features_dtype_cfg.dtype_break_down_mapping['date'].keys()]

    for d in datasets:
        cur_leaking_cols = find_shared_cols(leaking_cols, d.columns)
        if cur_leaking_cols:
            d = d.drop(columns=cur_leaking_cols)
        d = initial_X_clean(d)
        prepared_datasets.append(d)

    return compute_verbose_skf_adv_dfs(
        prepared_datasets
    )

def compute_verbose_skf_adv_dfs(datasets):
    res = []

    for df1, df2 in tqdm(combinations(datasets, 2)):
        res.append(
            adv_cv.make_cv(df1, df2, cv=5, shuffle=True, fold_method='skf')
        )
    return res

# LGBMClassifier(random_state=cfg.RSEED, verbose=-1) - results into near perfect score for all comparisons!
# LogisticRegression(random_state=cfg.RSEED)
adv_cv = AdversarialCV(
    LGBMClassifier(random_state=cfg.RSEED, verbose=-1),
    cfg.RSEED
)

# %%

# Lambdas are expenisve, but better designed and lazy because of calling. 
# It is still not a bottleneck. Caching or prepopulated dictionary can be used. 
get_train_sets = lambda: [train_2016, train_2017]
get_properties_sets = lambda: [properties_2016, properties_2017]
get_train_properties_sets = lambda: [
    train,
    properties_2016.drop(train.index),
    properties_2017.drop(train.index)
]
get_train_properties_2016_sets = lambda: [
    train_2016,
    properties_2016.drop(train_2016.index)
]

# Q4 is only in the train 2016.
# TODO: create a config FEATURES to store breakdown columns. Accessing from cleaning is wrong.
get_no_date_train_2016 = lambda: train_2016.drop(
    columns=features_dtype_cfg.dtype_break_down_mapping['date'].keys()
)
get_train_q4 = lambda: get_no_date_train_2016().loc[train_2016['is_trans_q4']]
get_train_q4_train_2016_sets = lambda: [
    get_no_date_train_2016()[~get_no_date_train_2016().index.isin(get_train_q4().index)],
    get_train_q4()
]
get_train_months_q4_sets = lambda: [
    get_train_q4().loc[train_2016['trans_month'] == m_num] for m_num in [10, 11, 12]
]

# %%
def make_datasets_dict_to_test(datasets_dict: dict):
    """
    Creates a dictionary with datasets to test adversarial computing.
    """
    return {title: [d.sample(2000, replace=True, random_state=cfg.RSEED) for d in dfs] 
                                                    for title, dfs in datasets_dict.items()}

major_datasets = {
    "train": get_train_sets(),
    "properties": get_properties_sets(),
    "train_properties": get_train_properties_sets(),
    "train_properties_2016": get_train_properties_2016_sets(),
    "train_months_q4": get_train_months_q4_sets(),
    "train_q4_train_2016": get_train_q4_train_2016_sets()
}

if cfg.dev_run:
    adv_results_df = handle_adv_computing(make_datasets_dict_to_test(major_datasets), 'lgbm')
else:
    adv_results_df = handle_adv_computing(major_datasets, 'lgbm')
