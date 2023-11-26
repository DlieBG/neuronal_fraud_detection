from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from load_data import load_data, predict_columns
import tensorflow as tf
import numpy as np
from transform_data import transform_data, inverse_transform_data

training_data = load_data('data/Trainingsdaten.csv')
prediction_data = load_data('data/Klassifizierungsdaten.csv', include_id=True)

training_data, prediction_data = transform_data(training_data, prediction_data)

input = training_data.drop(columns='TARGET_BETRUG')
output = training_data['TARGET_BETRUG']

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=.2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_train.shape[1],)),
    tf.keras.layers.Dropout(.7),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(.7),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(input_train, output_train, epochs=10, batch_size=32, validation_data=(input_test, output_test))

loss, accuracy = model.evaluate(input_test, output_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

prediction_data['TARGET_BETRUG'] = model.predict(prediction_data[predict_columns])

prediction_data = inverse_transform_data(prediction_data)

prediction_data.sort_values(by='TARGET_BETRUG', ascending=False).to_csv('predict.csv')
