import pandas as pd

def __to_bool(data_frame: pd.DataFrame, key: str, true: str) -> pd.Series:
    return data_frame[key].where(data_frame[key] == true)

def load_data(csv_path: str) -> pd.DataFrame:
    raw_data = pd.read_csv(
        filepath_or_buffer=csv_path,
        delimiter=';',
    )

    parsed_data = pd.DataFrame()

    parsed_data['id'] = raw_data['BESTELLIDENT']
    parsed_data['fraud'] = __to_bool(raw_data, 'TARGET_BETRUG', 'ja')

    parsed_data.set_index(
        keys='id'
    )

    return parsed_data
 
print(
    load_data('data/Trainingsdaten.csv').head(),
    load_data('data/Trainingsdaten.csv').info()
)

