class SemanticFeaturesManager:
    # TODO should I add more complexity on cemantic class? Feature groups based on cemantics. 
    room = [
        'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
        'threequarterbathnbr', 'fullbathcnt', 'roomcnt'
    ]

    area = [
        'basementsqft', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
        'finishedsquarefeet6', 'finishedsquarefeet12', 'finishedsquarefeet13',
        'finishedsquarefeet15', 'finishedsquarefeet50', 'lotsizesquarefeet',
        'garagetotalsqft', 'poolsizesum', 'yardbuildingsqft17', 'yardbuildingsqft26'
    ]

    property_amenities = [
        'fireplacecnt', 'fireplaceflag', 'garagecarcnt', 'hashottuborspa',
        'poolcnt', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'decktypeid'
    ]

    location = [
        'fips', 'latitude', 'longitude', 'rawcensustractandblock',
        'censustractandblock', 'regionidcounty', 'regionidcity',
        'regionidzip', 'regionidneighborhood'
    ]

    property_type = [
        'airconditioningtypeid', 'architecturalstyletypeid', 'buildingqualitytypeid',
        'buildingclasstypeid', 'heatingorsystemtypeid', 'propertylandusetypeid',
        'storytypeid', 'typeconstructiontypeid'
    ]

    structural = [
        'numberofstories', 'unitcnt', 'yearbuilt'
    ]

    tax = [
        'taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt',
        'taxamount', 'assessmentyear', 'taxdelinquencyflag', 'taxdelinquencyyear'
    ]
    
    timely = ['transactiondate']

    zoning_landuse = ['propertycountylandusecode', 'propertyzoningdesc']

    def get_all_groups(self):
        return [
            self.room, self.area, self.property_amenities,
            self.location, self.property_type, self.structural,
            self.tax, self.timely, self.zoning_landuse
        ]