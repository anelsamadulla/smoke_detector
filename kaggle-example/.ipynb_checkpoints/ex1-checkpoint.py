import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import kagglehub
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import RobustScaler

import shutil

src = "/home/codespace/.cache/kagglehub/datasets/deepcontractor/smoke-detection-dataset/versions/1/smoke_detection_iot.csv"
dst = "/workspaces/smoke_detector/smoke_detection_iot.csv"

shutil.copyfile(src, dst)
print("✅ File copied to workspace folder:", dst)


# Download latest version
path = kagglehub.dataset_download("deepcontractor/smoke-detection-dataset")

print("Path to dataset files:", path)

print(os.listdir(path))

df = pd.read_csv(os.path.join(path,'smoke_detection_iot.csv'))

df.head()

df.isna().sum().sum()

df.duplicated().sum()

df.info()

df.shape

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Get the correlation with the 'Fire Alarm' column
fire_alarm_correlation = correlation_matrix['Fire Alarm'].sort_values(ascending=False)

# Print the correlation
print("Correlation with Fire Alarm:")
print(fire_alarm_correlation)

# Visualize the correlation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Prepare data for feature importance
X = df.drop('Fire Alarm', axis=1)
y = df['Fire Alarm']

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Visualize feature importance
plt.figure(figsize=(8, 4))
feature_importances.plot(kind='bar')
plt.title('Feature Importance for Fire Alarm Prediction')
plt.ylabel('Importance')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns for outlier visualization
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Exclude the target variable 'Fire Alarm' if it's in the list
if 'Fire Alarm' in numerical_cols:
    numerical_cols = numerical_cols.drop('Fire Alarm')
if 'Unnamed: 0' in numerical_cols:
    numerical_cols = numerical_cols.drop('Unnamed: 0')
if 'UTC' in numerical_cols:
    numerical_cols = numerical_cols.drop('UTC')


# Create box plots for each numerical column
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i + 1) # Adjusted subplot grid to 4x4
    sns.boxplot(x=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

drops = ["Unnamed: 0","NC2.5","eCO2[ppm]","NC1.0"]

df = df.drop(drops,axis=1)

df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LinearRegression()

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Linear Regression Baseline")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊RandomForestClassifier")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training metrics
train_rmse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test metrics
test_rmse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"TRAIN → RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
print(f"TEST  → RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Predictions on test data
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - RandomForestClassifier")
plt.show()

# Optional: Detailed classification report
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# If return 0 mean no data leakage

import pandas as pd

# Convert scaled NumPy arrays back to DataFrames with original column names
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

overlap = pd.merge(X_train_df, X_test_df, how='inner')
print(len(overlap))

import joblib

joblib.dump(model, 'smoke_model.pkl')
print("Model saved as smoke_model.pkl ✅")

# Conclusion
# Baseline (Linear Regression)

# RMSE : 0.2822
# MAE : 0.2236
# R² : 0.6117 → Moderate performance; good starting point.
# Random Forest Classifier (initial)

# Model predicted only one class.
# Severe class imbalance detected → model underperformed.
# 3 After addressing class imbalance / retraining:

# Confusion Matrix: [[3575, 0], [ 0, 8951]]
# Accuracy, Precision, Recall, F1-score: 1.00 across all metrics. → Perfect classification achieved.
# Interpretation
# The final model perfectly distinguishes between classes.
# Possible causes: • Genuine deterministic feature-target relationship. • OR data leakage (target information leaked into features).
# Next Steps
# Verify no data leakage (check preprocessing and feature generation steps).
# Validate using k-fold cross-validation.
# Test model stability on unseen data (new or external dataset).
# Consider simpler models (e.g., Logistic Regression) to confirm separability.
# Conclusion
# The RandomForestClassifier achieved perfect classification accuracy. If data integrity is confirmed, this indicates a highly separable dataset with strong predictive features. Otherwise, the performance may reflect information leakage and requires careful verification.