import pandas
import numpy
numpy.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from clean import clean_and_type

training_df = pandas.read_csv(
    filepath_or_buffer="./Trainingsdaten.csv",
    delimiter=";",
    index_col='BESTELLIDENT',
)

prediction_df = pandas.read_csv(
    filepath_or_buffer="./Klassifizierungsdaten.csv",
    delimiter=";",
    index_col='BESTELLIDENT',
)

clean_and_type(training_df)
clean_and_type(prediction_df)

numerical_cols = training_df.select_dtypes(include=numpy.number).columns.tolist()
categorical_cols = list(set(training_df.columns) - set(numerical_cols))

label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    training_df[col] = label_encoders[col].fit_transform(training_df[col])
    prediction_df[col] = label_encoders[col].transform(prediction_df[col])

numerical_cols.remove('TARGET_BETRUG')
standardscaler = StandardScaler()
training_df[numerical_cols] = standardscaler.fit_transform(training_df[numerical_cols])
prediction_df[numerical_cols] = standardscaler.transform(prediction_df[numerical_cols])

input = training_df.drop(columns='TARGET_BETRUG')
output = training_df['TARGET_BETRUG']

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=.2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(input_train, output_train, epochs=10, batch_size=32, validation_data=(input_test, output_test))

loss, accuracy = model.evaluate(input_test, output_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# prediction_input = prediction_df.drop(columns=['BESTELLIDENT'])

prediction_df['TARGET_BETRUG'] = model.predict(prediction_df)

# print(prediction_df.head())

print(prediction_df.sort_values(by='TARGET_BETRUG', ascending=False).head())