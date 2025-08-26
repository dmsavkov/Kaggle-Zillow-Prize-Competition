from sklearn import clone
import pandas as pd
from pathlib import Path

from zillow.features.cleaning import DataCleaner
from zillow.config.feature_engineering import FeaturesDtypeConversionConfig_v1
from zillow.utils.common import read_data, save_df_by_extension
from zillow.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR


def handle_main_datasets_cleaning(config):
    datasets = [
        RAW_DATA_DIR / 'train.parquet',
        RAW_DATA_DIR / 'train_2016.parquet',
        RAW_DATA_DIR / 'train_2017.parquet',
        RAW_DATA_DIR / 'properties_2016.parquet',
        RAW_DATA_DIR / 'properties_2017.parquet',
    ]
    handle_dfs_cleaning(datasets, config)
    
def handle_dfs_cleaning(dfs_paths: list[Path], config):
    '''
    dfs: list of dataset file names
    '''
    cleaner = DataCleaner(config=config, verbose=0)
    transformer = TransformDatasets(dfs_paths, cleaner, 
                                    template_title="{}_cleaned_v1", extension='parquet')
    transformer.transform()

class TransformDatasets:
    def __init__(self, dataset_paths, estimator, template_title="{}", extension='parquet') -> None:
        self.dataset_paths = dataset_paths
        self.estimator = estimator
        self.template_title = template_title
        self.extension = extension

    def _save_dataset(self, df, file_name):
        file_name = self.template_title.format(file_name)
        res_path = INTERIM_DATA_DIR / f'{file_name}.{self.extension}'
        save_df_by_extension(df, res_path, self.extension)

    def transform(self):
        for path in self.dataset_paths:
            df = read_data(path, dtype=None)
            est = clone(self.estimator)
            df = est.fit_transform(df)
            self._save_dataset(df, path.stem)

def min_cleaner_test():
    df = read_data(INTERIM_DATA_DIR / "train_2016.parquet")
    df.drop(columns='assessmentyear', inplace=True, errors='raise')
    cleaner = DataCleaner(config=FeaturesDtypeConversionConfig_v1(), verbose=True)
    df = cleaner.fit_transform(df)

def test_data_cleaner_basic_mapping():
    # prepare a small DataFrame
    df = pd.DataFrame({
        'transactiondate': ['2016-01-01', None],
        'hashottuborspa': [1, None],     # bool mapping
        'bedroomcnt': [3, None],         # int mapping with NaN
    })

    cleaner = DataCleaner(config=FeaturesDtypeConversionConfig_v1(), verbose=0)
    df_cleaned = cleaner.fit_transform(df)

    # transactiondate should be dropped
    assert 'transactiondate' not in df_cleaned.columns

    # hashottuborspa should be bool
    assert df_cleaned['hashottuborspa'].dtype == bool

    # bedroomcnt had NaN → should become a nullable Int (e.g. Int8, Int16, etc.)
    assert pd.api.types.is_integer_dtype(df_cleaned['bedroomcnt'])
    assert df_cleaned['bedroomcnt'].dtype.name.startswith('Int')

    # values are preserved
    assert df_cleaned.loc[0, 'bedroomcnt'] == 3
    assert pd.isna(df_cleaned.loc[1, 'bedroomcnt'])

handle_dfs_cleaning([
    INTERIM_DATA_DIR / 'train_train_2016.parquet',
    INTERIM_DATA_DIR / 'val_train_2016.parquet',
], FeaturesDtypeConversionConfig_v1())
