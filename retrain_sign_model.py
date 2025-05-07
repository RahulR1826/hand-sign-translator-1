import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# === CONFIGURATION ===
DATASET_PATH = "hand_signs_data.csv"
MODEL_PATH = "sign_model.joblib"
REPLY_DICT_PATH = "reply_dict.joblib"

# === LOAD & PREPROCESS ===
df = pd.read_csv(DATASET_PATH)

# Rename last column to 'label' if needed
df = df.rename(columns={df.columns[-1]: "label"})
X = df.iloc[:, :-1]
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === TRAIN MODEL ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === SAVE OUTPUT FILES ===
joblib.dump(model, MODEL_PATH)
joblib.dump(dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)), REPLY_DICT_PATH)

print("✅ Model and reply_dict updated successfully!")
