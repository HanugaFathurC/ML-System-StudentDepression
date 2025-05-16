# === Import necessary libraries ===
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# === Create output directory ===
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load raw dataset ===
df = pd.read_csv("../student_depression_raw.csv")

# === Drop unnecessary columns ===
df.drop(columns=["id"], inplace=True)

# === Handle outliers in 'Age' using IQR method ===
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Age'] = df['Age'].clip(lower=lower_bound, upper=upper_bound)

# === Encode categorical features using LabelEncoder ===
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Save encoders ===
joblib.dump(label_encoders, os.path.join(OUTPUT_DIR, "label_encoders.joblib"))

# === Split features and target ===
X = df.drop(columns=["Depression"])
y = df["Depression"]

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

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
df_processed = pd.DataFrame(X_scaled, columns=X.columns)
df_processed["Depression"] = y.values
df_processed.to_csv(os.path.join(OUTPUT_DIR, "student_depression_processed.csv"), index=False)

print("✅ Preprocessing complete. Files saved in 'output/' folder.")
