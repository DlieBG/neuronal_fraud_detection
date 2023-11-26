import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from load_data import sequential_columns, discreet_columns, datetime_columns

standard_scaler_columns = sequential_columns
label_encoder_columns = discreet_columns + datetime_columns

standard_scaler = StandardScaler()
label_encoders = {}

def transform_data(*dataframes: pd.DataFrame) -> tuple[pd.DataFrame]:
    standard_scaler.fit(pd.concat(dataframes)[standard_scaler_columns])

    for dataframe in dataframes:
        dataframe[standard_scaler_columns] = standard_scaler.transform(dataframe[standard_scaler_columns])

    for label_encoder_column in label_encoder_columns:
        label_encoders[label_encoder_column] = LabelEncoder()
        label_encoders[label_encoder_column].fit(pd.concat(dataframes)[label_encoder_column])

        for dataframe in dataframes:
            dataframe[label_encoder_column] = label_encoders[label_encoder_column].transform(dataframe[label_encoder_column])

    return tuple(dataframes)

def inverse_transform_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe[standard_scaler_columns] = standard_scaler.inverse_transform(dataframe[standard_scaler_columns])
    
    for label_encoder_column in label_encoder_columns:
        dataframe[label_encoder_column] = label_encoders[label_encoder_column].inverse_transform(dataframe[label_encoder_column])

    return dataframe
