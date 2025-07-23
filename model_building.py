import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ------------------------------
# Load the dataset
# ------------------------------
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country', 'salary'
]

df = pd.read_csv('adult.csv', names=columns, na_values=' ?', skipinitialspace=True)

# ------------------------------
# Data Cleaning
# ------------------------------
df.dropna(inplace=True)

# ------------------------------
# Encode Categorical Columns
# ------------------------------
categorical_cols = df.select_dtypes(include='object').columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoder for later use in Streamlit

# ------------------------------
# Split the data
# ------------------------------
X = df.drop('salary', axis=1)
y = df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Train the RandomForest model
# ------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Evaluate the model
# ------------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Save model and encoders
# ------------------------------
joblib.dump(model, 'salary_model.pkl')
joblib.dump(encoders, 'encoders.pkl')

print("✅ Model and encoders saved successfully.")
