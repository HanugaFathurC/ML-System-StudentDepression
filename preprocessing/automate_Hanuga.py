# === Import necessary libraries ===
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# === Create output directory ===
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "student_depression_raw.csv")

# === Load raw dataset ===
df = pd.read_csv(csv_path)

# === Drop unnecessary columns ===
df.drop(columns=['id', 'City', 'Profession', 'Degree'], inplace=True)

# === Drop rows with the error entry  ===
df = df[df['Financial Stress'] != '?']

# === Handle outliers in 'Age' using IQR method ===
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Age'] = df['Age'].clip(lower=lower_bound, upper=upper_bound)

# === Encode categorical features using LabelEncoder ===
# Create a dictionary to store label encoders
label_encoders = {}

# Encode categorical features
categorical_features = df.select_dtypes(include='object').columns.tolist()

# Get  numerical feature
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Encode all categorical (object) columns
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the encoder for inverse_transform

# Save the encoders for future decoding
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, "label_encoders.joblib"))

# === Split features and target ===
X = df.drop(columns=["Depression"])
y = df["Depression"]

# === Scale features ===
if 'Depression' in numerical_features:
    numerical_features.remove('Depression')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_features])

# Save the scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

# === Combine scaled features with categorical features ===
X_scaled = pd.DataFrame(X_scaled, columns=numerical_features)

# Add encoded categorical features back to the DataFrame
for col in categorical_features:
    X_scaled[col] = df[col].values

# Reorder columns to match original DataFrame
X_scaled = X_scaled[X.columns]


# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Save train/test splits as CSV ===
pd.DataFrame(X_train, columns=X.columns).to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

# === Save fully processed dataset ===
df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
df_processed.to_csv(os.path.join(OUTPUT_DIR, "student_depression_processed.csv"), index=False)

print("✅ Preprocessing complete. Files saved in 'output/' folder.")
