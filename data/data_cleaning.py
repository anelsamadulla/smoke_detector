import pandas as pd
import numpy as np


df = pd.read_csv('Alarms_of_the_sensors.csv', sep=';')

df.head()

df = df.rename(columns={
    "Created time": "timestamp",
    "Originator": "sensor_id",
    "Type": "alarm_type",
    "Severity": "severity",
    "Status": "status",
    "atm_pm_1_0": "atm_pm_1_0",
    "atm_pm_2_5": "atm_pm_2_5",
    "atm_pm_10_0": "atm_pm_10_0",
    "co2": "co2",
    "delta": "delta",
    "hum": "humidity",
    "ma": "ma",
    "temp": "temperature"
})

df.columns

df.to_csv("Alarms_of_the_sensors_renamed.csv", index=False)
