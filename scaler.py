from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

def scale_fit2(df):
    columns = df.columns.tolist()
    columns.remove('TARGET_BETRUG')
    scaler = StandardScaler()
    scaler.fit(df[columns])
    with open('scaler_settings.pkl', 'wb') as file:
        pickle.dump(scaler, file)

def scale_fit(df):
    scaler = {}
    for col in df.columns:
        if col == 'TARGET_BETRUG':
            continue
        scaler[col] = StandardScaler()
        scaler[col].fit(df[col].values.reshape(-1, 1))
    with open('scaler_settings.pkl', 'wb') as file:
        pickle.dump(scaler, file)

def scale_transform2(df):
    columns = df.columns.tolist()
    if 'TARGET_BETRUG' in columns:
        columns.remove('TARGET_BETRUG')
    with open('scaler_settings.pkl', 'rb') as file:
        scaler = pickle.load(file)
    df[columns] = scaler.transform(df[columns])

def scale_transform(df):
    with open('scaler_settings.pkl', 'rb') as file:
        scaler = pickle.load(file)
    for col in scaler:
        df[col] = scaler[col].transform(df[col].values.reshape(-1, 1))
