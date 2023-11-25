from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from load_data import load_data
import tensorflow as tf
import numpy as np

training_data = load_data('data/Trainingsdaten.csv', include_id=False)


print(training_data.info())

input = training_data.drop(columns='is_fraud')
output = training_data['is_fraud']

numerical_cols = input.select_dtypes(include=np.number).columns.tolist()
categorical_cols = list(set(input.columns) - set(numerical_cols))

print(numerical_cols)

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