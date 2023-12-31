import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

numerical_features = [
    'WERT_BEST',
    'ANZ_BEST',
    'SESSION_TIME',
    'ANZ_BEST_GES',
    'WERT_BEST_GES',
    'MAHN_AKT',
    'MAHN_HOECHST',
]

categorical_features = [
    'B_EMAIL',
    'B_TELEFON',
    'FLAG_LRIDENTISCH',
    'FLAG_NEWSLETTER',
    'Z_METHODE',
    'Z_CARD_ART',
    'Z_CARD_VALID',
    'Z_LAST_NAME',
    'TAG_BEST',
    'CHK_LADR',
    'CHK_RADR',
    'CHK_KTO',
    'CHK_CARD',
    'CHK_COOKIE',
    'CHK_IP',
    'FAIL_LPLZ',
    'FAIL_LORT',
    'FAIL_LPLZORTMATCH',
    'FAIL_RPLZ',
    'FAIL_RORT',
    'FAIL_RPLZORTMATCH',
    'NEUKUNDE',
]

__datetime_features = [
    { 'key': 'B_GEBDATUM', 'type': 'date' },
    { 'key': 'TIME_BEST', 'type': 'time' },
    { 'key': 'DATUM_LBEST', 'type': 'date' },
]

__product_features = [
    'ANUMMER_01',
    'ANUMMER_02',
    'ANUMMER_03',
    'ANUMMER_04',
    'ANUMMER_05',
    'ANUMMER_06',
    'ANUMMER_07',
    'ANUMMER_08',
    'ANUMMER_09',
    'ANUMMER_10',
]

products_feature = 'ANUMMERS'

categorical_list_features = [products_feature] + [
    feature['key'] for feature in __datetime_features
]

def __to_datetime(dataframe: pd.DataFrame, feature: dict):
    datetime = pd.to_datetime(dataframe[feature['key']])

    match feature['type']:
        case 'date':
            return [y * 365 + m * 30 + d for y, m, d in zip(datetime.dt.year, datetime.dt.month, datetime.dt.month)]
        case 'time':
            return [h * 60 + m for h, m in zip(datetime.dt.hour, datetime.dt.minute)]

def __to_zipped_products(dataframe: pd.DataFrame):
    return [
        list(zipped) for zipped in zip(*[dataframe[feature].fillna('0').astype('string') for feature in __product_features])
    ]

def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    parsed_dataframe = pd.DataFrame()

    for feature in numerical_features:
        parsed_dataframe[feature] = dataframe[feature].astype('string').str.replace(',', '.').astype(float)

    for feature in categorical_features:
        parsed_dataframe[feature] = dataframe[feature].fillna('0').astype('string')

    for feature in __datetime_features:
        parsed_dataframe[feature['key']] = __to_datetime(dataframe, feature)

    parsed_dataframe[products_feature] = __to_zipped_products(dataframe)

    if 'TARGET_BETRUG' in dataframe:
        parsed_dataframe['TARGET_BETRUG'] = np.where(dataframe['TARGET_BETRUG'] == 'ja', 1, 0)

    return parsed_dataframe
