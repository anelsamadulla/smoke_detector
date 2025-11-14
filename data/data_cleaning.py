#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


# In[3]:


alarms = pd.read_csv('Alarms_of_the_sensors.csv', sep=None, engine='python')


# In[4]:


alarms.head()


# In[5]:


print(alarms.columns)


# In[6]:


alarms.head()


# In[7]:


alarms = alarms.drop(columns=['Assignee', 'delta'], errors='ignore')


# In[8]:


columns_to_show = ['Timestamp', 'sensor_id', 'alarm_type', 'severity', 'status']
existing_columns = [col for col in columns_to_show if col in alarms.columns]
print(alarms[existing_columns].head())


# In[9]:


alarms['alarm_type'].unique()


# In[10]:


alarms['severity'].unique()


# In[11]:


from datetime import datetime, timezone


# In[12]:


alarms = pd.read_csv('Alarms_of_the_sensors.csv', sep=None, engine='python')


# In[13]:


alarms.head()


# In[14]:


alarms['time'] = pd.to_datetime(alarms['time'], utc=True)

# Convert from its current timezone to UTC+5 if needed
# If you know the timestamps are already UTC+5 logically but in UTC format:
alarms['time'] = alarms['time'].dt.tz_convert('Etc/GMT-5')

# Then convert to pure UTC for database consistency
alarms['time_utc'] = alarms['time'].dt.tz_convert('UTC')

# Create UNIX timestamp column (seconds)
alarms['timestamp'] = alarms['time_utc'].astype('int64') // 10**9

alarms.head()


# In[18]:


sensor_ids = [
    "Sensor_0013",
    "Sensor_0039",
    "Sensor_0045",
    "Sensor_0057",
    "Sensor_0068",
    "Sensor_0079",
    "Sensor_0080",
    "Sensor_0089"
]

sensor_data = {}

for sensor in sensor_ids:
    filename = f"{sensor}_150_days.csv"
    df = pd.read_csv(filename, sep=None, engine="python")

    # Convert timestamp
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_convert("Etc/GMT-5")
    df['time_utc'] = df['time'].dt.tz_convert("UTC")
    df['timestamp'] = df['time_utc'].astype("int64") // 10**9

    # Drop unwanted columns if they exist
    for col in ['delta', 'time', 'time_utc', 'ma']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Reorder columns
    new_order = ['timestamp', 'atm_pm_1_0', 'atm_pm_2_5', 'co2', 'hum', 'temp']
    # keep only those columns that exist
    new_order = [col for col in new_order if col in df.columns]
    df = df[new_order]

    # Save into dictionary
    sensor_data[sensor] = df

# Example access:
print(sensor_data["Sensor_0057"].head())


# In[38]:


print(sensor_data["Sensor_0039"].head())


# In[39]:


from math import sqrt
from sklearn.metrics import mean_squared_error


# In[40]:


print(sensor_data["Sensor_0045"].head())


# In[36]:


def make_multitarget_lagged(df, target_cols, n_lags=10, horizon=1):
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Create lag features
    for col in target_cols:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Create future targets
    for col in target_cols:
        df[f"{col}_future"] = df[col].shift(-horizon)

    # Drop all rows with missing lagged values
    df_model = df.dropna().reset_index(drop=True)

    # Build feature matrix
    lag_cols = []
    for col in target_cols:
        lag_cols += [f"{col}_lag_{lag}" for lag in range(1, n_lags + 1)]

    X = df_model[lag_cols]
    y = df_model[[f"{col}_future" for col in target_cols]]

    return df_model, X, y


# In[ ]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

target_cols = ["co2", "atm_pm_1_0", "atm_pm_2_5", "temp", "hum"]
n_lags = 10
horizon = 1

models = {}
metrics = {}

for fname in files:
    sensor_name = fname.replace("_150_days.csv", "")
    df = sensor_data[sensor_name]  # <-- USE CLEANED DATA
    df_model, X, y = make_multitarget_lagged(df, target_cols, n_lags, horizon)

    if len(X) < 100:
        print(f"⚠️ Skipping {fname}: not enough rows after lagging")
        continue

    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse_dict = {
        col: sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        for i, col in enumerate(target_cols)
    }

    models[fname] = model
    metrics[fname] = rmse_dict

print("\n=== RMSE per sensor file ===")
for fname, rmse_dict in metrics.items():
    print(f"\n{fname}:")
    for col, rmse in rmse_dict.items():
        print(f"  {col}: {rmse:.3f}")


# In[ ]:


# 1) Load
sensor_0013_df = pd.read_csv("Sensor_0013_150_days.csv", sep=None, engine="python")

# If your file already has 'timestamp' and 'time_utc' etc, keep at least 'time' and numeric columns
# e.g. sensor_0013_df = sensor_0013_df[["time", "co2", "atm_pm_2_5", "ma", "temp"]]

# 2) Build lagged dataset (using co2 as target, 10 lags, forecast 1 step ahead)
df_0013_model, X, y = make_lagged_features(sensor_0013_df, target_col="co2",
                                           n_lags=10, horizon_steps=1)

# 3) Time-based train/test split (last 20% as test)
split_idx = int(len(df_0013_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 4) Train model
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 5) Evaluate
y_pred = rf.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
# rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE (Sensor_0013, forecast 1 step ahead): {rmse:.3f}")


# In[ ]:


# Attach predictions & residuals to the test part of df
df_test = df_0013_model.iloc[split_idx:].copy()
df_test["co2_pred"] = y_pred
df_test["residual"] = df_test["co2_pred"] - df_test["co2"]

# Rolling statistics of residuals
window = 300  # e.g. 300 points (tune to your frequency)
df_test["resid_mean"] = df_test["residual"].rolling(window).mean()
df_test["resid_std"] = df_test["residual"].rolling(window).std()

# Simple drift/anomaly rule:
k = 3  # 3-sigma rule
df_test["drift_flag"] = (
    (df_test["residual"] - df_test["resid_mean"]).abs() > k * df_test["resid_std"]
)

print(df_test[["time", "co2", "co2_pred", "residual", "drift_flag"]].tail())


# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

files = [
    "Sensor_0013_150_days.csv",
    "Sensor_0039_150_days.csv",
    "Sensor_0045_150_days.csv",
    "Sensor_0057_150_days.csv",
    "Sensor_0068_150_days.csv",
    "Sensor_0079_150_days.csv",
    "Sensor_0080_150_days.csv",
    "Sensor_0089_150_days.csv",
]

models = {}
metrics = {}

for fname in files:
    sensor_name = fname.replace("_150_days.csv", "")  # convert to dict key
    df = sensor_data[sensor_name]  # <-- use cleaned data

    df_model, X, y = make_lagged_features(df, target_col="co2",
                                          n_lags=10, horizon_steps=1)

    if len(X) < 100:
        print(f"⚠️ Skipping {fname}: not enough rows after lagging")
        continue

    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, y_pred))

    models[fname] = model
    metrics[fname] = rmse

print("\nRMSE per file:")
for fname, rmse in metrics.items():
    print(f"{fname}: {rmse:.3f}")


# In[35]:


print(df_model.shape)
print("Feature rows:", X.shape[0])
print("Target rows:", y.shape[0])
print("Time unique count:", df["time"].nunique())
print("Total rows loaded:", df.shape[0])
print("NaN rows dropped:", df.shape[0] - df_model.shape[0])


# In[ ]:




