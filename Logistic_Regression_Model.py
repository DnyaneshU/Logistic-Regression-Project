# Logistic_Regression_Model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("vehicle_maintenance_data.csv")
df.columns = df.columns.str.strip()  # Clean any extra spaces

# Define target and features
X = df.drop('Need_Maintenance', axis=1)
y = df['Need_Maintenance']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Handle date columns (dropping for simplicity)
date_cols = ['Last_Service_Date', 'Warranty_Expiry_Date']
X = X.drop(columns=[col for col in date_cols if col in X.columns])
categorical_cols = [col for col in categorical_cols if col not in date_cols]

# Preprocessing
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Logistic Regression pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
cm = confusion_matrix(y_test, y_pred)
labels = ['No Maintenance Needed', 'Needs Maintenance']

print("Confusion Matrix:")
print(pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels], columns=[f"Predicted: {l}" for l in labels]))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save to directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

print("\nConfusion matrix plot saved to:", os.path.join(output_dir, "confusion_matrix.png"))
