import pandas
import numpy
numpy.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_and_type(df):
    # if 'BESTELLIDENT' in df:
    #     df.drop('BESTELLIDENT', axis=1, inplace=True)
    if 'TARGET_BETRUG' in df:
        df['TARGET_BETRUG'] = numpy.where(df['TARGET_BETRUG'] == 'ja', 1, 0)
    df['B_EMAIL'] = numpy.where(df['B_EMAIL'] == 'ja', 1, 0)
    df['B_TELEFON'] = numpy.where(df['B_TELEFON'] == 'ja', 1, 0)
    df['FLAG_LRIDENTISCH'] = numpy.where(df['FLAG_LRIDENTISCH'] == 'ja', 1, 0)
    df['FLAG_NEWSLETTER'] = numpy.where(df['FLAG_NEWSLETTER'] == 'ja', 1, 0)
    df['FAIL_LPLZ'] = numpy.where(df['FAIL_LPLZ'] == 'ja', 1, 0)
    df['FAIL_LORT'] = numpy.where(df['FAIL_LORT'] == 'ja', 1, 0)
    df['FAIL_LPLZORTMATCH'] = numpy.where(df['FAIL_LPLZORTMATCH'] == 'ja', 1, 0)
    df['FAIL_RPLZ'] = numpy.where(df['FAIL_RPLZ'] == 'ja', 1, 0)
    df['FAIL_RORT'] = numpy.where(df['FAIL_RORT'] == 'ja', 1, 0)
    df['FAIL_RPLZORTMATCH'] = numpy.where(df['FAIL_RPLZORTMATCH'] == 'ja', 1, 0)
    df['NEUKUNDE'] = numpy.where(df['NEUKUNDE'] == 'ja', 1, 0)
    df['CHK_LADR'] = numpy.where(df['CHK_LADR'] == 'ja', 1, 0)
    df['CHK_RADR'] = numpy.where(df['CHK_RADR'] == 'ja', 1, 0)
    df['CHK_KTO'] = numpy.where(df['CHK_KTO'] == 'ja', 1, 0)
    df['CHK_CARD'] = numpy.where(df['CHK_CARD'] == 'ja', 1, 0)
    df['CHK_COOKIE'] = numpy.where(df['CHK_COOKIE'] == 'ja', 1, 0)
    df['CHK_IP'] = numpy.where(df['CHK_IP'] == 'ja', 1, 0)

    df['B_GEBDATUM'] = pandas.to_datetime(df['B_GEBDATUM'])

    df['B_GEBDATUM_YEAR'] = df['B_GEBDATUM'].dt.year
    df['B_GEBDATUM_MONTH'] = df['B_GEBDATUM'].dt.month
    df['B_GEBDATUM_DAY'] = df['B_GEBDATUM'].dt.day

    df.drop('B_GEBDATUM', axis=1, inplace=True)

    df['TIME_BEST'] = pandas.to_datetime(df['TIME_BEST'])

    df['TIME_BEST_HOUR'] = df['TIME_BEST'].dt.hour
    df['TIME_BEST_MINUTES'] = df['TIME_BEST'].dt.minute

    df.drop('TIME_BEST', axis=1, inplace=True)

    df['ANUMMER_08'] = df['ANUMMER_08'].fillna(0).astype(int)
    df['ANUMMER_09'] = df['ANUMMER_09'].fillna(0).astype(int)
    df['ANUMMER_10'] = df['ANUMMER_10'].fillna(0).astype(int)

    df['WERT_BEST'] = df['WERT_BEST'].str.replace(',', '.').astype(float)
    df['WERT_BEST_GES'] = df['WERT_BEST_GES'].str.replace(',', '.').astype(float)