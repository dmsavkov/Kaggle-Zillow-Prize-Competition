# Using config with initialisation to allow chnages.
# Maybe, you'd like to use version one with 1 col different. 
class FeaturesDtypeConversionConfig_v1:
    def __init__(self) -> None:
        self.groups_to_break_down = [
            'date',
            'censuses'
        ]
        
        self.dtype_mapping = {
            # NaN is False. 
            'hashottuborspa': 'bool', 
            'fireplaceflag': 'bool', 
            'taxdelinquencyflag': 'bool', 
            'poolcnt': 'bool', 
            'storytypeid': 'bool', 
            'pooltypeid10': 'bool', 
            'pooltypeid2': 'bool', 
            'pooltypeid7': 'bool', 
            'decktypeid': 'bool', 
            'buildingclasstypeid': 'bool',
            'typeconstructiontypeid': 'bool', # 2 unique values, one of them contains only 1 sample.

            'propertycountylandusecode': 'category',
            'propertyzoningdesc': 'category',
            'fips': 'category',
            'regionidcounty': 'category',
            'regionidcity': 'category',
            'regionidzip': 'category',
            'regionidneighborhood': 'category',

            'parcelid': 'int32',
            'bedroomcnt': 'int8', 
            'fullbathcnt': 'int8',
            'roomcnt': 'int8',
            'fireplacecnt': 'int8',
            'garagecarcnt': 'int8',
            'numberofstories': 'int8',
            'airconditioningtypeid': 'int8',
            'architecturalstyletypeid': 'int8',
            'buildingqualitytypeid': 'int8',
            'heatingorsystemtypeid': 'int8',
            
            'yearbuilt': 'int16',

            # NOTE: won't drop assessmentyear because it still variats in the 
            # unified datasets.
            'assessmentyear': 'int16',
            'taxdelinquencyyear': 'int16',
            
            'basementsqft': 'float32',
            'finishedfloor1squarefeet': 'float32',
            'calculatedfinishedsquarefeet': 'float32',
            'finishedsquarefeet6': 'float32',
            'finishedsquarefeet12': 'float32', 
            'finishedsquarefeet13': 'float32',
            'finishedsquarefeet15': 'float32',
            'finishedsquarefeet50': 'float32',
            'lotsizesquarefeet': 'float32',
            'garagetotalsqft': 'float32',
            'poolsizesum': 'float32',
            'yardbuildingsqft17': 'float32',
            'yardbuildingsqft26': 'float32',
            'taxvaluedollarcnt': 'float32',
            'structuretaxvaluedollarcnt': 'float32',
            'landtaxvaluedollarcnt': 'float32',
            'taxamount': 'float32',
            'latitude': 'float32',
            'longitude': 'float32',

            # Contain couple of floats. 
            'bathroomcnt': 'float32',
            'calculatedbathnbr': 'float32',
            'unitcnt': 'float32',
            'propertylandusetypeid': 'float32',
        }

        self.dtype_break_down_mapping = {
            'censuses': {
                'raw_census_fips': 'category',             
                'raw_census_main_id': 'category',
                'raw_census_suffix_id': 'category',        
                'raw_census_group_id': 'category',         
                'raw_census_group_number_id': 'category',
                'census_fips': 'category',                 
                'census_main_id': 'category',
                'census_suffix_id': 'category',            
                'census_group_id': 'category',             
                'census_group_number_id': 'category',
            },
            
            'date': {
                'trans_month': 'int8',
                'trans_day': 'int8', 
                'is_trans_q4': 'bool',
            } 
        }
        
        
        self.cols_high_missingness = [
            'threequarterbathnbr',
            'yardbuildingsqft17',
            'yardbuildingsqft26',   
            'basementsqft',
            'storytypeid'
        ]
        
        # Duplicate or redundant. 
        self.cols_redundant = [
            'finishedsquarefeet50',  
            'finishedsquarefeet12'
        ]
        
        # Will be broken down or redundant
        self.cols_to_drop = [
            'transactiondate', 
            'rawcensustractandblock',  
            'censustractandblock'
        ]

        self.all_cat_cols = [
            col for col, dtype in self.dtype_mapping.items() if dtype == 'category'
        ] + [
            col for group in self.dtype_break_down_mapping.values() 
            for col, dtype in group.items() if dtype == 'category'
        ]