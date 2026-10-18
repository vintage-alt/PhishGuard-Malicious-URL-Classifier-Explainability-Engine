import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
import math
import re

# --- 1. Load Real Data ---
print("Loading dataset...")
# Update these column names after you peek at the CSV!
# I am assuming they are named 'URL' and 'label'
df = pd.read_csv("phishing_data.csv") 

# Drop rows with missing values
df = df.dropna(subset=['URL', 'label'])

# Take a random sample of 30,000 URLs to make training fast (you can change this)
# df = df.sample(n=30000, random_state=42) 

print(f"Dataset shape: {df.shape}")

# --- 2. Feature Engineering (Same as before) ---
def extract_features(url):
    url = str(url) # Ensure it's a string
    features = {}
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_at'] = 1 if '@' in url else 0
    features['num_digits'] = sum(c.isdigit() for c in url)
    
    # Suspicious keywords
    keywords = ['login', 'verify', 'secure', 'account', 'update', 'bank', 'support', 'paypal', 'signin']
    features['suspicious_words'] = sum(1 for kw in keywords if kw in url.lower())
    
    # Shannon Entropy
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    features['entropy'] = entropy
    
    return features

print("Extracting features (this might take a minute for 200k URLs)...")
features_df = df['URL'].apply(extract_features).apply(pd.Series)
X = features_df
y = df['label']

# --- 3. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train Model ---
print("Training Random Forest on real data...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all CPU cores
model.fit(X_train, y_train)

# --- 5. Evaluate ---
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# --- 6. Explainability (SHAP) safely ---
print("\n--- Explainability (SHAP) ---")
explainer = shap.TreeExplainer(model)
# Explain just the first 5 URLs in the test set to save memory
shap_values = explainer.shap_values(X_test.iloc[:5])

# In newer SHAP versions, it returns a 3D array or a list. Let's handle it safely.
# We want to explain a Phishing prediction. 
print(f"Explaining prediction for URL: {df.iloc[X_test.index[0]]['URL']}")
prediction = y_pred[0]
print(f"Model predicted: {'Phishing' if prediction == 1 else 'Benign'}")

# Print feature impacts safely
feature_names = X.columns
# If the model uses all features, shap_values will map to them. 
for i, feature in enumerate(feature_names):
    try:
        # Get the SHAP value for the Phishing class (index 1)
        if isinstance(shap_values, list):
            impact = shap_values[1][0][i]
        else:
            impact = shap_values[0][i][1] # for newer shap versions
        print(f" - {feature}: {impact:.3f}")
    except Exception as e:
        pass

import joblib

# Save the trained model to disk
joblib.dump(model, 'phishing_rf_model.pkl')
print("\nModel successfully saved as 'phishing_rf_model.pkl'")