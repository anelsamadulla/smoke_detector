# import pandas as pd

# df = pd.read_csv('CUBA_SMOKE_0057_for_the_last_90_days.csv', sep=';')

# # Drop the useless column
# df = df.drop(columns=['delta'], errors='ignore')

# # Convert timestamp
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# # Replace NaN with 0 or with forward-fill, depending on context
# df = df.fillna(0)

# # Optional: reorder columns for clarity
# df = df[['Timestamp', 'temp', 'hum', 'co2', 'atm_pm_1_0', 'atm_pm_2_5', 'atm_pm_10_0', 'ma']]

# # Inspect
# print(df.head())

# import matplotlib.pyplot as plt

# # Plot temperature and humidity
# plt.figure(figsize=(10, 4))
# plt.plot(df['Timestamp'], df['temp'], label='Temperature (°C)')
# plt.plot(df['Timestamp'], df['hum'], label='Humidity (%)')
# plt.legend()
# plt.title('Environmental Conditions')
# plt.show()

# plt.figure(figsize=(10, 4))
# plt.plot(df['Timestamp'], df['co2'], label='CO₂ (ppm)')
# plt.plot(df['Timestamp'], df['atm_pm_2_5'], label='PM2.5 (µg/m³)')
# plt.legend()
# plt.title('CO₂ and Particulate Matter')
# plt.show()

# X_new = df.rename(columns={
#     'temp': 'Temperature[C]',
#     'hum': 'Humidity[%]',
#     'co2': 'eCO2[ppm]',
#     'atm_pm_1_0': 'PM1.0',
#     'atm_pm_2_5': 'PM2.5',
#     'atm_pm_10_0': 'PM10.0'
# })

# X_new = X_new[['Temperature[C]', 'Humidity[%]', 'eCO2[ppm]', 'PM1.0', 'PM2.5', 'PM10.0']]

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv('CUBA_SMOKE_0057_for_the_last_90_days.csv', sep=';')
df = df.drop(columns=['delta'], errors='ignore')
df = df.fillna(0)

# Use your available features
X = df[['temp', 'hum', 'co2', 'atm_pm_1_0', 'atm_pm_2_5', 'atm_pm_10_0', 'ma']]

# Create a dummy target for now (you’ll need real labels later)
# For example, assume 'smoke' if co2 > 2000 or pm2.5 > 10
y = ((df['co2'] > 2000) | (df['atm_pm_2_5'] > 10)).astype(int)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save it
joblib.dump(model, 'smoke_model_simple.pkl')
print("✅ Model trained and saved as smoke_model_simple.pkl")


