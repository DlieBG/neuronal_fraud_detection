import pandas as pd
import numpy as np

def __to_bool(data_frame: pd.DataFrame, key: str, true: str = 'ja') -> pd.Series:
    return data_frame[key].where(data_frame[key] == true)

def __to_datetime(data_frame: pd.DataFrame, key: str) -> pd.Timestamp:
    return pd.to_datetime(data_frame[key])

def load_data(csv_path: str, include_id: bool = True) -> pd.DataFrame:
    raw_data = pd.read_csv(
        filepath_or_buffer=csv_path,
        delimiter=';',
    )

    parsed_data = pd.DataFrame()

    if include_id:
        parsed_data['id'] = raw_data['BESTELLIDENT']

    parsed_data['is_fraud'] = np.where(raw_data['TARGET_BETRUG'] == 'ja', 1, 0)
    parsed_data['has_email'] = raw_data['B_EMAIL']
    parsed_data['has_telefon'] = raw_data['B_TELEFON']
    
    parsed_data['birth_date.year'] = __to_datetime(raw_data, 'B_GEBDATUM').dt.year
    parsed_data['birth_date.month'] = __to_datetime(raw_data, 'B_GEBDATUM').dt.month
    parsed_data['birth_date.day'] = __to_datetime(raw_data, 'B_GEBDATUM').dt.day

    parsed_data['has_identical_addresses'] = raw_data['FLAG_LRIDENTISCH']
    parsed_data['has_newsletter'] = raw_data['FLAG_NEWSLETTER']
    parsed_data['payment_method'] = raw_data['Z_METHODE']
    parsed_data['payment_card_type'] = raw_data['Z_CARD_ART'].fillna('unbekannt')
    parsed_data['payment_card_valid'] = raw_data['Z_CARD_VALID']
    parsed_data['has_payment_last_name'] = raw_data['Z_LAST_NAME'].fillna('unbekannt')

    parsed_data['order_value'] = raw_data['WERT_BEST'].str.replace(',', '.').astype(float)

    parsed_data['order_day_of_week'] = raw_data['TAG_BEST']

    parsed_data['order_time.hour'] = __to_datetime(raw_data, 'TIME_BEST').dt.hour
    parsed_data['order_time.minute'] = __to_datetime(raw_data, 'TIME_BEST').dt.minute

    parsed_data['product_count'] = raw_data['ANZ_BEST']

    # ANUMMER_XX
    # parsed_data['has_payment_last_name'] = raw_data['Z_LAST_NAME'].fillna('unbekannt')

    if include_id:
        parsed_data.set_index(
            keys='id'
        )

    return parsed_data
 