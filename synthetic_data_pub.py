import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import timesynth as ts
import json
import time
import random
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

#Adding mttr values
mttr_values = np.random.randint(4, 25, size=1440)
mttr_values = mttr_values.tolist()

#Adding Score
G_Score = np.random.randint(80, 101, size=1440)
score_values = G_Score.tolist()

#Adding Jitter
num_rows = 1440 # Example size, adjust as needed
low_range_percentage = 0.7
low_range = (30, 50)
full_range = (20, 80)
num_low_range = int(num_rows * low_range_percentage)
num_full_range = num_rows - num_low_range
low_range_values = [random.randint(*low_range) for _ in range(num_low_range)]
full_range_values = [random.randint(*full_range) for _ in range(num_full_range)]
jitter = low_range_values + full_range_values
random.shuffle(jitter)

#Adding RTT
num_rows = 1440
low_range_percentage = 0.7
low_range = (100, 200)
full_range = (80, 220)
num_low_range = int(num_rows * low_range_percentage)
num_full_range = num_rows - num_low_range
low_range_values = [random.randint(*low_range) for _ in range(num_low_range)]
full_range_values = [random.randint(*full_range) for _ in range(num_full_range)]
rtt_values = low_range_values + full_range_values
random.shuffle(rtt_values)

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

#Load best models and make predictions on unseen data
best_models = joblib.load('best_models.pkl')

# Prepare to store predictions for each time point in X_unseen
all_predicted_values = []

# Get project details for Pub/Sub
#FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL
project_id = ""
topic_id = ""

# Create a Publisher client
publisher = pubsub_v1.PublisherClient()

# Construct the topic path
topic_path = publisher.topic_path(project_id, topic_id)

for idx, x_unseen in enumerate(X_unseen):
    single_predicted_values = {'Avg_Connected_UEs': float(x_unseen)}
    for column, model in best_models.items():
        x_unseen_reshaped = np.array([[x_unseen]])
        prediction = model.predict(x_unseen_reshaped)[0]
        single_predicted_values[column] = float(prediction)  # Convert to float to ensure JSON serializability

    # Add the additional key-value pairs
    single_predicted_values['network_id'] = '154.29.15.1'

    single_predicted_values['Cell Availability%'] = 100
  
    if idx < len(mttr_values):
        single_predicted_values['MTTR'] = float(mttr_values[idx])

    if idx < len(jitter):
        single_predicted_values['Jitter'] = float(jitter[idx])

    if idx < len(rtt_values):
        single_predicted_values['RTT'] = float(rtt_values[idx])

    if idx < len(score_values):
        single_predicted_values['Score'] = float(score_values[idx])

    if score_values[idx] >= 97:
            reliability = "good"
    elif score_values[idx] >= 92:
        reliability = "fair"
    elif score_values[idx] >= 88:
        reliability = "average"
    elif score_values[idx] >= 84:
        reliability = "poor"
    elif score_values[idx] >= 80:
        reliability = "very poor"
    else:
        reliability = ""
        
    single_predicted_values['5g_reliability_carrier'] = reliability    
    
    # Convert to JSON and print
    json_output = json.dumps(single_predicted_values, indent=4)
    print(json_output)

    # Publish the message to Pub/Sub
    data_str = json_output
    data = data_str.encode("utf-8")

    time.sleep(1)
    
    future = publisher.publish(topic_path, data")
    future.result()  # Ensure the message is published
    
    # Sleep for 1 minute
    time.sleep(60)




