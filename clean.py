import pandas
import numpy
numpy.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import label_types

def get_training_data():
    df = pandas.read_csv(
        filepath_or_buffer="./Trainingsdaten.csv",
        delimiter=";",
        index_col='BESTELLIDENT',
    )
    clean(df)
    df[label_types.traget_col] = label_types.yes_no_encoder.transform(df[label_types.traget_col])
    df.info()
    return df

def get_classification_data():
    df = pandas.read_csv(
        filepath_or_buffer="./Klassifizierungsdaten.csv",
        delimiter=";",
        index_col='BESTELLIDENT',
    )
    clean(df)
    return df

def clean_dates(df):
    # B_GEBDATUM
    df['B_GEBDATUM'] = pandas.to_datetime(df['B_GEBDATUM'])

    df['B_GEBDATUM_YEAR'] = df['B_GEBDATUM'].dt.year
    df['B_GEBDATUM_MONTH'] = df['B_GEBDATUM'].dt.month
    df['B_GEBDATUM_DAY'] = df['B_GEBDATUM'].dt.day

    df.drop('B_GEBDATUM', axis=1, inplace=True)

    #DATUM_LBEST
    df['DATUM_LBEST'] = pandas.to_datetime(df['DATUM_LBEST'])

    df['DATUM_LBEST_YEAR'] = df['DATUM_LBEST'].dt.year
    df['DATUM_LBEST_MONTH'] = df['DATUM_LBEST'].dt.month
    df['DATUM_LBEST_DAY'] = df['DATUM_LBEST'].dt.day

    df.drop('DATUM_LBEST', axis=1, inplace=True)
    
    # TIME_BEST
    df['TIME_BEST'] = pandas.to_datetime(df['TIME_BEST'])

    df['TIME_BEST_HOUR'] = df['TIME_BEST'].dt.hour
    df['TIME_BEST_MINUTES'] = df['TIME_BEST'].dt.minute

    df.drop('TIME_BEST', axis=1, inplace=True)

def clean_days(df):
    df[label_types.day_column] = label_types.days_encoder.transform(df[label_types.day_column])

def clean_yes_no(df):
    for col in label_types.yes_no_columns:
        df[col].fillna(label_types.na, inplace=True)
        df[col] = label_types.yes_no_encoder.transform(df[col])

def clean_articels(df):
    for col in label_types.articel_columns:
        df[col].fillna(label_types.na_articel, inplace=True)
        df[col] = label_types.articel_encoder.transform(df[col])

def clean_payment(df):
    df[label_types.payment_method_cloumn] = label_types.payment_method_encoder.transform(df[label_types.payment_method_cloumn])

def clean_cards(df):
    df[label_types.card_type_column].fillna(label_types.na, inplace=True)
    df[label_types.card_type_column] = label_types.card_type_encoder.transform(df[label_types.card_type_column])
    df[label_types.card_valid_id_column].fillna(label_types.na, inplace=True)
    df[label_types.card_valid_id_column] = label_types.card_valid_id_encoder.transform(df[label_types.card_valid_id_column])

def clean_numbers(df):
    for col in label_types.float_columns:
        df[col] = df[col].str.replace(',', '.').astype(float)

def clean(df):
    clean_days(df)
    clean_yes_no(df)
    clean_articels(df)
    clean_payment(df)
    clean_cards(df)
    clean_numbers(df)
    clean_dates(df)
