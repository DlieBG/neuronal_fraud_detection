import pandas
import numpy
numpy.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# cols = ['B_EMAIL', 'B_TELEFON', 'B_GEBDATUM', 'FLAG_LRIDENTISCH', 'FLAG_NEWSLETTER',
#                 'Z_METHODE', 'Z_CARD_ART', 'Z_LAST_NAME', 'TAG_BEST', 'TIME_BEST', 'CHK_LADR',
#                 'CHK_RADR', 'CHK_KTO', 'CHK_CARD', 'CHK_COOKIE', 'CHK_IP', 'FAIL_LPLZ',
#                 'FAIL_LORT', 'FAIL_LPLZORTMATCH', 'FAIL_RPLZ', 'FAIL_RORT',
#                 'FAIL_RPLZORTMATCH', 'NEUKUNDE', 'DATUM_LBEST', 'Z_CARD_VALID', 'WERT_BEST', 'ANZ_BEST', 'ANUMMER_01', 'ANUMMER_02',
#                 'ANUMMER_03', 'ANUMMER_04', 'ANUMMER_05', 'ANUMMER_06', 'ANUMMER_07',
#                 'ANUMMER_08', 'ANUMMER_09', 'ANUMMER_10', 'SESSION_TIME', 'ANZ_BEST_GES',
#                 'WERT_BEST_GES', 'MAHN_AKT', 'MAHN_HOECHST']

df = pandas.read_csv(
    filepath_or_buffer="./Trainingsdaten.csv",
    delimiter=";",
    index_col='BESTELLIDENT',
    # dtype={col: 'category' for col in cat_cols}
)

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

df.info()

input = df.drop(columns='TARGET_BETRUG')
output = df['TARGET_BETRUG']

numerical_cols = input.select_dtypes(include=numpy.number).columns.tolist()
categorical_cols = list(set(input.columns) - set(numerical_cols))

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=.2)

scaler = StandardScaler()
input_train[numerical_cols] = scaler.fit_transform(input_train[numerical_cols])
input_test[numerical_cols] = scaler.fit_transform(input_test[numerical_cols])

for col in categorical_cols:
    labelenc = LabelEncoder()
    input_train[col] = labelenc.fit_transform(input_train[col])
    input_test[col] = labelenc.fit_transform(input_test[col])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(input_train, output_train, epochs=10, batch_size=32, validation_data=(input_test, output_test))

loss, accuracy = model.evaluate(input_test, output_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
