import tensorflow as tf
import pandas as pd
import numpy as np
import keras as k

class DatetimeType(enumerate):
    DATE = 'date'
    TIME = 'time'

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
    { 'key': 'B_GEBDATUM', 'type': DatetimeType.DATE },
    { 'key': 'TIME_BEST', 'type': DatetimeType.TIME },
    { 'key': 'DATUM_LBEST', 'type': DatetimeType.DATE },
]

datetime_features = [
    feature['key'] for feature in __datetime_features
]

products_feature = 'ANUMMERS'

product_features = [
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

def __to_datetime(dataframe: pd.DataFrame, feature: dict):
    datetime = pd.to_datetime(dataframe[feature['key']])

    match feature['type']:
        case DatetimeType.DATE:
            return [[y, m, d] for y, m, d in zip(datetime.dt.year, datetime.dt.month, datetime.dt.month)]
        case DatetimeType.TIME:
            return [[h, m] for h, m in zip(datetime.dt.hour, datetime.dt.minute)]

def __to_zipped_products(dataframe: pd.DataFrame):
    return [
        list(zipped) for zipped in zip(*[dataframe[feature].fillna('0').astype('string') for feature in product_features])
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

def dataframe_to_dataset(dataframe: pd.DataFrame, batch_size: int = 32) -> tf.data.Dataset:
    targets = dataframe.copy().pop('TARGET_BETRUG')
    
    return tf.data.Dataset.from_tensor_slices(
        tensors=(
            {
                **{
                    feature: np.expand_dims(dataframe[feature], 1) for feature in numerical_features + categorical_features
                },
                **{
                    feature: dataframe[feature].to_list() for feature in datetime_features + [products_feature]
                },
            },
            targets,
        )
    ).shuffle(
        buffer_size=len(dataframe),
    ).batch(
        batch_size=batch_size,
    ).prefetch(
        buffer_size=batch_size,
    )

def dataframe_to_tensors(dataframe: pd.DataFrame):
    return {
        **{
            feature: tf.convert_to_tensor(np.expand_dims(dataframe[feature], 1)) for feature in numerical_features + categorical_features
        },
        **{
            feature: tf.convert_to_tensor(dataframe[feature].to_list()) for feature in datetime_features + [products_feature]
        },
    }
    
