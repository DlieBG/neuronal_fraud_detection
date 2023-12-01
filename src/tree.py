from sklearn.model_selection import train_test_split
from load_data import load_data
import tensorflow_decision_forests as tfdf

training_data = load_data('data/Trainingsdaten.csv')
prediction_data = load_data('data/Klassifizierungsdaten.csv', include_id=True)

prediction_ds = tfdf.keras.pd_dataframe_to_tf_dataset(prediction_data)

train_df, test_df = train_test_split(training_data, test_size=.2)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label='TARGET_BETRUG')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label='TARGET_BETRUG')

model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

model.summary()

prediction_data['TARGET_BETRUG'] = model.predict(prediction_ds)

prediction_data.sort_values(by='TARGET_BETRUG', ascending=False).to_csv('predict_random_forest.csv')
