import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import timesynth as ts
import json
import time
import datetime as dt
from google.cloud import pubsub_v1
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load data
df = pd.read_csv('Airwave_OG.csv')

# Interpolate missing values
df.interpolate(inplace=True)

# Drop unnecessary columns
data = df.drop([
    'SIP Dropped Calls',
    'Cat M1 Bearer Drop pct',
    'Pct CA ScheduledUE with 0 EScell DL',
    'Pct CA ScheduledUE with 1 EScell DL',
    'Pct CA ScheduledUE with 2 EScell DL',
    'Pct CA ScheduledUE with 3 EScell DL',
    'SIP Calls with a Leg',
    'Pct CA ScheduledUE with 4 EScell DL',
    'Cat M1 Bearer Setup Failure pct',
    '_80th_percentile_traffic',
    'SIP DC%',
    'Pct CA ScheduledUE with 1 Scell UL',
    'Pct CA ScheduledUE with 2 Scell UL',
    'Pct CA ScheduledUE with 3 Scell UL',
    'HO_fail_PCT_InterFreq',
    'day'
], axis=1)

# Identify columns to iterate over
columns_to_iterate = []

for i in data.columns:
    if i == 'Avg_Connected_UEs':
        continue
    if df[i].dtypes != 'object':
        correlation = df[[i, 'Avg_Connected_UEs']].corr().iloc[0, 1]
        if not pd.isna(correlation):
            columns_to_iterate.append(i)

# TimeSynth setup
time_sampler_pp = ts.TimeSampler(stop_time=144.0)
irregular_time_samples_pp = time_sampler_pp.sample_irregular_time(resolution=0.1, keep_percentage=100)
pseudo_periodic = ts.signals.PseudoPeriodic(frequency=0.02, freqSD=0.001, ampSD=0.4)
timeseries_pp = ts.TimeSeries(pseudo_periodic)
samples_pp, _, _ = timeseries_pp.sample(irregular_time_samples_pp)
samples_pp = (abs(samples_pp) * 8) + 1

X_unseen = samples_pp
X = df['Avg_Connected_UEs']
predicted_values = {}
best_models = {}

# Training models and saving best models
for column in tqdm(columns_to_iterate, leave=False):
    y = df[column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': (RandomForestRegressor(), {'n_estimators': [50, 100, 150]}),
        'XGBoost': (xgb.XGBRegressor(), {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5, 7]})
    }
    
    best_score = -float('inf')
    best_model = None
    
    for name, (model, parameters) in models.items():
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train.values.reshape(-1, 1), y_train)
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    
    best_models[column] = best_model

# Save best models to file
joblib.dump(best_models, 'best_models.pkl')

# Load best models and make predictions on unseen data
best_models = joblib.load('best_models.pkl')

# Prepare to store predictions for each time point in X_unseen
all_predicted_values = []

# Get project details for Pub/Sub
project_id = "gcp-dataeng-demos-355417"
topic_id = "pubsub_dataflow_demo"

# Create a Publisher client
publisher = pubsub_v1.PublisherClient()

# Construct the topic path
topic_path = publisher.topic_path(project_id, topic_id)

for x_unseen in X_unseen:
    single_predicted_values = {'Avg_Connected_UEs': float(x_unseen)}
    for column, model in best_models.items():
        x_unseen_reshaped = np.array([[x_unseen]])
        prediction = model.predict(x_unseen_reshaped)[0]
        single_predicted_values[column] = float(prediction)  # Convert to float to ensure JSON serializability
    
    # Append the predictions to the list
    all_predicted_values.append(single_predicted_values)

    #adding network-id
    single_predicted_values['network_id'] = '154.29.15.1'
    
    # Convert to JSON and print
    json_output = json.dumps(single_predicted_values, indent=4)
    print(json_output)

    # Publish the message to Pub/Sub
    data_str = json_output
    data = data_str.encode("utf-8")
    future = publisher.publish(topic_path, data, origin="python-sample", username="gcp")
    future.result()  # Ensure the message is published
    
    # Sleep for 1 minute
    time.sleep(60)




