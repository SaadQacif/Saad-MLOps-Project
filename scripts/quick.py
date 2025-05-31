import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the 121-feature dataset
df = pd.read_csv("data/features/potato_features.csv")
feature_cols = [col for col in df.columns if col not in ["image_path", "class"]]
X = df[feature_cols].values

# Fit scaler on 121 features
scaler = StandardScaler()
scaler.fit(X)

# Save to models/bin/scaler.pkl
with open("models/bin/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler for 121 features saved to models/bin/scaler.pkl")