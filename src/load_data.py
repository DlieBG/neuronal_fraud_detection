import pandas as pd
import numpy as np

sequential_columns = [
    'WERT_BEST',
    'ANZ_BEST',
    'SESSION_TIME',
    'ANZ_BEST_GES',
    'WERT_BEST_GES',
    'MAHN_AKT',
    'MAHN_HOECHST',
]

discreet_columns = [
    'B_EMAIL',
    'B_TELEFON',
    'FLAG_LRIDENTISCH',
    'FLAG_NEWSLETTER',
    'Z_METHODE',
    'Z_CARD_ART',
    'Z_CARD_VALID',
    'Z_LAST_NAME',
    'TAG_BEST',
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

__datetime_columns = [
    { 'raw_key': 'B_GEBDATUM', 'new_key': 'B_GEBDATUM.year', 'type': 'year' },
    { 'raw_key': 'B_GEBDATUM', 'new_key': 'B_GEBDATUM.month', 'type': 'month' },
    { 'raw_key': 'B_GEBDATUM', 'new_key': 'B_GEBDATUM.day', 'type': 'day' },
    { 'raw_key': 'TIME_BEST', 'new_key': 'TIME_BEST.hour', 'type': 'hour' },
    { 'raw_key': 'TIME_BEST', 'new_key': 'TIME_BEST.minute', 'type': 'minute' },
    { 'raw_key': 'DATUM_LBEST', 'new_key': 'DATUM_LBEST.year', 'type': 'year' },
    { 'raw_key': 'DATUM_LBEST', 'new_key': 'DATUM_LBEST.month', 'type': 'month' },
    { 'raw_key': 'DATUM_LBEST', 'new_key': 'DATUM_LBEST.day', 'type': 'day' },
]

datetime_columns = [
    datetime_column['new_key'] for datetime_column in __datetime_columns
]

predict_columns = sequential_columns + discreet_columns + datetime_columns

def __to_datetime(raw_data: pd.DataFrame, datetime_column: dict) -> int:
    datetime = pd.to_datetime(raw_data[datetime_column['raw_key']])

    match datetime_column['type']:
        case 'year':
            return datetime.dt.year
        case 'month':
            return datetime.dt.month
        case 'day':
            return datetime.dt.day
        case 'hour':
            return datetime.dt.hour
        case 'minute':
            return datetime.dt.minute
        case _:
            return 0

def load_data(csv_path: str, include_id: bool = False) -> pd.DataFrame:
    raw_data = pd.read_csv(
        filepath_or_buffer=csv_path,
        delimiter=';',
    )

    parsed_data = pd.DataFrame()

    if include_id:
        parsed_data['BESTELLIDENT'] = raw_data['BESTELLIDENT']
            
    if 'TARGET_BETRUG' in raw_data:
        parsed_data['TARGET_BETRUG'] = np.where(raw_data['TARGET_BETRUG'] == 'ja', 1, 0)

    for sequential_column in sequential_columns:
        if sequential_column in raw_data:
            parsed_data[sequential_column] = raw_data[sequential_column].astype('string').str.replace(',', '.').astype(float)

    for discreet_column in discreet_columns:
        if discreet_column in raw_data:
            parsed_data[discreet_column] = raw_data[discreet_column].fillna('0').astype('string')

    for datetime_column in __datetime_columns:
        if datetime_column['raw_key'] in raw_data:
            parsed_data[datetime_column['new_key']] = __to_datetime(raw_data, datetime_column)

    return parsed_data
