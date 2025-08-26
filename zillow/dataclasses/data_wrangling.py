from dataclasses import dataclass, field
from typing import List

@dataclass
class MissingColumns:
    small: List[str] = field(default_factory=lambda: [])
    moderate: List[str] = field(default_factory=lambda: [])
    extreme: List[str] = field(default_factory=lambda: [])

@dataclass
class DefaultMissionColumns(MissingColumns):
    moderate: List[str] = field(default_factory=lambda: [
        'yardbuildingsqft17',
        'basementsqft',
        'poolsizesum',
        'garagetotalsqft'
    ])
    extreme: List[str] = field(default_factory=lambda: [
        'yardbuildingsqft26',
        'threequarterbathnbr',
        'taxdelinquencyyear',
        'storytypeid'
    ])

@dataclass
class ClipColumns:
    small: List[str] = field(default_factory=lambda: [])
    moderate: List[str] = field(default_factory=lambda: [])
    extreme: List[str] = field(default_factory=lambda: [])

    group_quantiles: List[float] = field(default_factory=lambda: [0.99, 0.95, 0.9])
    bound_type: str = 'both'

@dataclass
class DefaultClipColumns(ClipColumns):  
    small: List[str] = field(default_factory=lambda: [
        'bathroomcnt',
        'bedroomcnt',
        'fullbathcnt',
        'calculatedbathnbr',
        'garagecarcnt',
        'garagetotalsqft',
        'finishedsquarefeet6',
        'finishedsquarefeet12',
        'heatingorsystemtypeid',
        'poolsizesum',
        'unitcnt',
        'yardbuildingsqft17',
        'taxdelinquencyyear'
    ])
    moderate: List[str] = field(default_factory=lambda: [
        'structuretaxvaluedollarcnt',
        'taxvaluedollarcnt',
        'landtaxvaluedollarcnt',
        'taxamount',
        'calculatedfinishedsquarefeet',
        'finishedfloor1squarefeet',
        'finishedsquarefeet15',
        'finishedsquarefeet50',
    ])
    extreme: List[str] = field(default_factory=lambda: [
        'lotsizesquarefeet',
    ])
    bound_type: str = 'upper'
    
@dataclass
class WranglingColumnsConfig:
    clip_columns: ClipColumns = field(default_factory=DefaultClipColumns)
    missing_columns: MissingColumns = field(default_factory=DefaultMissionColumns)

    outliers: List[str] = field(default_factory=lambda: [
        'roomcnt',
        'unitcnt',
        'regionidzip',
        'taxdelinquencyyear',
        'poolsizesum',
        'garagecarcnt',
        'bathroomcnt',
        'bedroomcnt',
        'fullbathcnt',
        'calculatedbathnbr',
        'logerror'
    ])