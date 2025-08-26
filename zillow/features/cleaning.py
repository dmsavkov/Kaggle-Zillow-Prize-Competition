import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .break_down import break_down_censuses, break_down_date
from ..utils.common import find_unshared_cols, is_bool_false, find_shared_cols, throw_col_not_exist_warning

# TODO: probably should create class for initial preprocessing.
# TODO: update the function.
# TODO: create common global config and replcae target and index cols.
def initial_X_clean(df):
    df = df.copy()
    df_cols = set(df.columns)
    cols_to_dtype_bool = [c for c in ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag'] if c in df_cols]
    cols_to_dtype_cat = [c for c in ['propertycountylandusecode', 'propertyzoningdesc'] if c in df_cols]
    cols_to_drop = [c for c in ['transactiondate', 'logerror', 'parcelid'] if c in df_cols]

    df.drop(cols_to_drop, axis=1, inplace=True)
    for c in cols_to_dtype_bool:
        df[c] = df[c].apply(lambda x: False if pd.isna(x) else True)
    for c in cols_to_dtype_cat:
        # TODO why setting this doesn't work: df.loc[:, c] = df[c].astype('category').cat.codes
        df[c] = df[c].astype('category').cat.codes

    return df


def handle_cat_and_missing_quick(df):
    # I don't consider (and handle) boolean columns as missing.
    df = df.copy()
    cat_cols = df.select_dtypes(include='category').columns
    int_cols = df.select_dtypes(include='integer').columns
    float_cols = df.select_dtypes(include='float').columns

    df[cat_cols] = df[cat_cols].apply(lambda s: s.cat.codes).fillna(-1)
    df[int_cols] = df[int_cols].fillna(df[int_cols].mode().iloc[0])
    df[float_cols] = df[float_cols].fillna(df[float_cols].mean())

    # If you sample, it may be possible because some columns 
    # have very large missing ratios. 
    assert df.isna().sum().sum() == 0, "There are still NaN values in the DataFrame after filling missing values."

    return df

def clip_values_based_cfg(df, cfg):
    df = df.copy()
    tp = cfg.bound_type
    cols_matrix = [cfg.small, cfg.moderate, cfg.extreme]
    df_cols = df.columns

    for q, cols in zip(cfg.group_quantiles, cols_matrix):
        # TODO: additional function?
        existing_cols = find_shared_cols(df_cols, cols) 
        
        if len(existing_cols) == 0: continue
        if len(cols) > len(existing_cols):
            # TODO: Useless repeated looping. 
            for col in find_unshared_cols(cols, existing_cols):
                throw_col_not_exist_warning(col)

        lower = df[existing_cols].quantile((1 - q) if tp in ['both', 'lower'] else 0)
        upper = df[existing_cols].quantile(q if tp in ['both', 'upper'] else 1)
        df[existing_cols] = df[existing_cols].clip(lower=lower, upper=upper, axis=1)
    
    return df


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, config, verbose=1):
        self.config = config
        self.verbose = verbose

        self.initial_memory_usage_mb = None
    
    def fit(self, X, y=None):
        """Initialise the DataFrame to process."""
        self.df = X.copy()
        if self.verbose:
            print("Original data info:")
            print(self.df.info(memory_usage='deep'))
            self.initial_memory_usage_mb = self._calc_memory_usage_mb()
        return self
        
    def transform(self, X):
        """
        Apply full cleaning pipeline to DataFrame.
        If something is wrong change config!
        """

        # Order matters!
        self._apply_feature_engineering()
        self._apply_dtypes()

        # Including broken down columns. 
        self._drop_existing_columns(
            self.config.cols_redundant + self.config.cols_to_drop
        )

        if self.verbose:
            print("\nCleaned data info:")
            print(self.df.info(memory_usage='deep'))
            self._get_memory_usage_report()
        return self.df
    
    def _apply_feature_engineering(self):
        groups = set(self.config.groups_to_break_down)
        if 'date' in groups: 
            self.df = break_down_date(self.df)
        if 'censuses' in groups:
            self.df = break_down_censuses(self.df)
    
    def _apply_dtypes(self):
        map_ = self._get_prepared_dtype_map()
        self._check_dtype_mapping(map_)
        self.df = self.df.astype(map_, copy=False, errors='raise')

    def _get_prepared_dtype_map(self):
        df_cols = set(self.df.columns)
        map_ = self.config.dtype_mapping.copy()
        is_na_dict = self.df.isna().any()

        for group in self.config.groups_to_break_down:
            map_.update(
                self.config.dtype_break_down_mapping[group]
            )

        # NOTE: should filter after groups because we get broken features,
        # however breaking down of a certain group isn't guaranteed.
        map_ = {col: dtype for col, dtype in self.config.dtype_mapping.items() 
                                                            if col in df_cols}
        
        for col, dtype in map_.items():
            # Use map_[col] instead of dtype because we may already changed it!
            if dtype.startswith('int') and is_na_dict[col]:
                map_[col] = self._handle_int_nan(map_[col])
            
            # qq What dtypes are acceptable and wrong for pandas processing and indexing?

        self._prepare_boolean_cols(map_)

        return map_

    def _prepare_boolean_cols(self, map_):
        cols = [c for c, dt in map_.items() if dt == 'bool']
        self.df[cols] = self.df[cols].map(is_bool_false)
    
    def _handle_int_nan(self, dtype: str) -> str:
        # Using nullable int.
        if self.verbose:
            print('Column to int contains NaN. Using nullable int. ')
        return dtype.capitalize()

    def _check_dtype_mapping(self, map_):
        pass
        '''for col, dtype in map_.items():
            match dtype[:3].lower():
                case 'int':
                    if not (self.)'''
    
    def _drop_existing_columns(self, cols_to_drop):
        cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]
        self.df.drop(columns=cols_to_drop, inplace=True)
    
    def _get_memory_usage_report(self):
        """Generate memory usage comparison report"""
        # Memory before cannot be none because of self.verbose (or there 
        # must be a prior error which will be raised).
        memory_before = self.initial_memory_usage_mb 
        memory_after = self._calc_memory_usage_mb()
        reduction = ((memory_before - memory_after) / memory_before) * 100
        
        report = {
            'memory_before_mb': round(memory_before, 2),
            'memory_after_mb': round(memory_after, 2),
            'reduction_percent': round(reduction, 2),
            'reduction_mb': round(memory_before - memory_after, 2)
        }
        
        print(f"Memory Usage Report:")
        print(f"Before: {report['memory_before_mb']} MB")
        print(f"After: {report['memory_after_mb']} MB")
        print(f"Reduction: {report['reduction_mb']} MB ({report['reduction_percent']}%)")
        
        return report

    def _calc_memory_usage_mb(self):
        return self.df.memory_usage(deep=True).sum() / 1024**2


