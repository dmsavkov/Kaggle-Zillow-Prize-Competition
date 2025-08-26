import pandas as pd
import numpy as np
from ..utils.common import check_for_not_existing_cols

# TODO common features class creator.
# TODO add more time series (as planned!!!!)
def break_down_date(df):
    cdate = 'transactiondate'
    if check_for_not_existing_cols(df, cdate): return df

    q4_months = [10, 11, 12]
    new_cols = ['trans_month', 'trans_day', 'is_trans_q4']

    date_col = pd.to_datetime(df[cdate], format="%Y-%m-%d").dt.date
    new_date_features = pd.DataFrame(
        date_col.apply(lambda x: [x.month, x.day, x.month in q4_months]).to_list(),
        columns=new_cols,
        index=df.index    # ensure same row index
    )

    prev_shape = df.shape
    df[new_cols] = new_date_features
    assert df.shape == (prev_shape[0], prev_shape[1] + len(new_cols)), "Shape mismatch after breaking down date"
    
    return df

# TODO should I really duplicate functions like this: scalability or duplicating? What's more important? How to design this code properly?
# Should I add external check conditions functions?
def break_down_censuses(df: pd.DataFrame):
    """Extract census components from rawcensustractandblock and censustractandblock."""
    df = df.copy()  
    rawc_col, c_col = 'rawcensustractandblock', 'censustractandblock'
    if check_for_not_existing_cols(df, [rawc_col, c_col]): return df

    result_columns = {
        'censustractandblock': [
            'census_fips',
            'census_main_id', 
            'census_suffix_id',
            'census_group_id',
            'census_group_number_id'
        ],
        'rawcensustractandblock': [
            'raw_census_fips',
            'raw_census_main_id',
            'raw_census_suffix_id',
            'raw_census_group_id',
            'raw_census_group_number_id'
        ]
    }
    

    if rawc_col in df.columns:
        rawc_results = _break_down_rawcensus(df[rawc_col])

        rawc_df = pd.DataFrame(rawc_results.tolist(), dtype=df[rawc_col].dtype,
                             columns=result_columns[rawc_col])

        for col in result_columns[rawc_col]:
            df[col] = rawc_df[col]
            

    if c_col in df.columns:
        cen_results = _break_down_census(df[c_col])

        cen_df = pd.DataFrame(cen_results.tolist(), dtype=df[c_col].dtype,
                                columns=result_columns[c_col])

        for col in result_columns[c_col]:
            df[col] = cen_df[col]
    
    return df

def _break_down_rawcensus(array):
    # Lengths of rawcensus vary. So we have to apply switch.

    def unit_break(n: float): 
        s = str(n)
        if not (10 <= len(s) <= 15):
            return [np.nan] * 5

        first_part, second_part = s.split(".")
        fips, census_main_id = [first_part[:4], first_part[4:]]

        census_suffix_id = block_group_id = block_id = np.nan
        match len(second_part):
            case 1:
                block_group_id = second_part
            case 2:
                census_suffix_id = second_part
            case 3:
                block_id = second_part
            case 4: 
                block_group_id = second_part[0]
                block_id = second_part[1:]
            case 5:
                census_suffix_id = second_part[:2]
                block_id = second_part[2:]
            case 6:
                census_suffix_id = second_part[:2]
                block_group_id = second_part[2]
                block_id = second_part[3:]
        
        return [fips, census_main_id, census_suffix_id, block_group_id, block_id]
    
    # Note: the objects and floats (NaN) are returned from unit.
    return array.apply(unit_break)

def _break_down_census(array):

    def unit_break(n: float): 
        n = float(n)
        s = str(n)

        # if not 16: 14 digits + float's .0, then it's NaN
        if len(s) != 16: 
            return [np.nan] * 5

        fips = s[:4]
        census_main_id = s[4:8]
        census_suffix_id = s[8:10]
        block_group_id = s[10]
        block_id = s[11:14]

        return [fips, census_main_id, census_suffix_id, block_group_id, block_id]
    
    return array.apply(unit_break)

# TESTS
# temp = cur_train[census_columns]
# res_temp = break_down_censuses(temp).head(10)