from ..config.feature_engineering import FeaturesDtypeConversionConfig_v1
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from dataclasses import replace

default_load_dtype_map = {}
for col in FeaturesDtypeConversionConfig_v1().all_cat_cols:
    default_load_dtype_map[col] = 'category'

def print_separation_line():
    print("\n", "-" * 30, "\n")

def find_shared_dfs_cols(df1, *args):
    return find_shared_cols(df1.columns, *[df.columns for df in args])

# NOTE: using sets for these methods changes the order of cols.
def find_all_cols(cols1, *args):
    cols = set(cols1)
    for more_cols in args:
        cols |= set(more_cols)
    return list(cols)

def find_shared_cols(cols1, *args):
    cols = set(cols1)
    for more_cols in args:
        cols &= set(more_cols)
    return list(cols)

def find_unshared_cols(cols1, *args):
    return list(
        set(find_all_cols(cols1, *args)) ^ set(find_shared_cols(cols1, *args))
    )

def find_shared_dtype_map(df, dtype_map):
    existing_cols = set(
        find_shared_cols(df.columns, dtype_map.keys())
    )
    return {
        col: dtype for col, dtype in dtype_map.items() if col in existing_cols
    }


def read_data(path, dtype='default', **kwargs):
    if isinstance(path, str):
        path = Path(path)
    suffix = path.suffix

    match suffix:
        case '.csv':
            df = pd.read_csv(path, **kwargs)
        case '.parquet':
            df = pd.read_parquet(path, **kwargs)
        case '.feather':
            df = pd.read_feather(path, **kwargs)
        case _:
            raise ValueError(f"Unsupported file extension: {suffix}")
        
    if dtype is not None:
        if dtype == 'default':
            dtype = default_load_dtype_map
        
        df = df.astype(
            find_shared_dtype_map(df, dtype)
        )
    
    return df

def save_df_by_extension(df, path, extension):
    match extension:
        case 'csv':
            df.to_csv(path, index=None)
        case 'parquet':
            df.to_parquet(path, index=None)
        case _:
            raise ValueError(f"Unsupported file extension: {extension}")
        

def check_for_not_existing_cols(df, col_names):
    if isinstance(col_names, str):
        col_names = [col_names]

    # Set for O(1) performance.
    existing_cols = set(df.columns)
    sign = False
    for col_name in col_names:
        if col_name not in existing_cols:
            throw_col_not_exist_warning(col_name)
            sign = True

    return sign

def throw_col_not_exist_warning(col_name):
    warnings.warn(f"Column '{col_name}' not found in DataFrame.", UserWarning)


def is_bool_false(x):
    return pd.isna(x) or x == 0

def get_feat_nature_types(df):
    d = {}

    d['numeric_features'] = df.select_dtypes([np.number]).columns
    d['categorical_features'] = df.select_dtypes([np.object_, 'category']).columns
    d['boolean_features'] = df.select_dtypes([np.bool_]).columns
    return d


def modify_dataclass(dataclass, **kwargs):
    return replace(dataclass, **kwargs)

def get_obj_features(df: pd.DataFrame):
    return [c for c in df.columns if (
        pd.api.types.is_string_dtype(df[c])
    )]


