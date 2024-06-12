import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import timesynth as ts
import json
import time
import random
import datetime as dt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# Load data
df = pd.read_csv('Airwave_OG.csv')

# Replace spaces with underscores in column names
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('%', '_Pct')

# Interpolate missing values
df.interpolate(inplace=True)

# Drop unnecessary columns
data = df.drop([
    'SIP_Dropped_Calls',
    'Bearer_Setup_Failure_Voice_Num',
    'Bearer_Setup_Failure_Voice_Pct',
    'Cat_M1_Bearer_Drop_pct',
    'Pct_CA_ScheduledUE_with_0_EScell_DL',
    'Pct_CA_ScheduledUE_with_1_EScell_DL',
    'Pct_CA_ScheduledUE_with_2_EScell_DL',
    'Pct_CA_ScheduledUE_with_3_EScell_DL',
    'SIP_Calls_with_a_Leg',
    'Pct_CA_ScheduledUE_with_4_EScell_DL',
    'Cat_M1_Bearer_Setup_Failure_pct',
    '_80th_percentile_traffic',
    'SIP_DC_Pct',
    'Pct_CA_ScheduledUE_with_1_Scell_UL',
    'Pct_CA_ScheduledUE_with_2_Scell_UL',
    'Pct_CA_ScheduledUE_with_3_Scell_UL',
    'HO_fail_PCT_InterFreq',
    'day',
    'hr',
    'weekend'
], axis=1)

#Adding MTTR values
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

df = pd.read_pickle('best_models.pkl')

# Rename the column
df.rename(columns={'PCT_Time_MIMO': 'Pct_Time_MIMO'}, inplace=True)

# Save the DataFrame back to the .pkl file
df.to_pickle('best_models.pkl')

# Prepare to store predictions for each time point in X_unseen
all_predicted_values = []

# Get project details for Pub/Sub
#FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL #FILL
project_id = "networkperformanceassessment"
topic_id = "ran"

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

    single_predicted_values['Cell Availability_Pct'] = 100
  
    if idx < len(mttr_values):
        single_predicted_values['MTTR'] = float(mttr_values[idx])

    if idx < len(jitter):
        single_predicted_values['Jitter'] = float(jitter[idx])

    if idx < len(rtt_values):
        single_predicted_values['RTT'] = float(rtt_values[idx])

    if idx < len(score_values):
        single_predicted_values['Score'] = float(score_values[idx])

    single_predicted_values['network_id'] = '154.29.15.1'
    
    # Convert to JSON and print
    json_output = json.dumps(single_predicted_values, indent=4)
    print(json_output)

    # Publish the message to Pub/Sub
    data_str = json_output
    data = data_str.encode("utf-8")

    time.sleep(1)
    
    future = publisher.publish(topic_path, data)
    future.result()  # Ensure the message is published
    
    # Sleep for 1 minute
    time.sleep(60)




