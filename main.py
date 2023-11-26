import pandas
import numpy
numpy.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import clean
import scaler

training_df = clean.get_training_data()
scaler.scale_fit2(training_df)
scaler.scale_transform2(training_df)
prediction_df = clean.get_classification_data()
scaler.scale_transform2(prediction_df)

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

prediction_df['TARGET_BETRUG'] = model.predict(prediction_df)

print(prediction_df.sort_values(by='TARGET_BETRUG', ascending=False).head())