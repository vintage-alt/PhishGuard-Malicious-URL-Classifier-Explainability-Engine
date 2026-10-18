import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import math
import re

# --- 1. Generate Synthetic Data (Replace with real dataset later) ---
# 0 = Benign, 1 = Phishing
data = {
    'url': [
        'https://www.google.com', 'https://github.com/openai', 'https://wikipedia.org',
        'http://paypal-login-secure.update-account.com/verify', 
        'http://192.168.1.105/login.php', 'http://apple-id-verify.support-alert.net',
        'https://stackoverflow.com/questions', 'https://www.amazon.com/dp/B08N5WRWNW'
    ],
    'label': [0, 0, 0, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

# --- 2. Feature Engineering ---
def extract_features(url):
    features = {}
    features['length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_at'] = 1 if '@' in url else 0
    features['num_digits'] = sum(c.isdigit() for c in url)
    
    # Suspicious keywords
    keywords = ['login', 'verify', 'secure', 'account', 'update', 'bank', 'support']
    features['suspicious_words'] = sum(1 for kw in keywords if kw in url.lower())
    
    # Shannon Entropy (measures randomness)
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    features['entropy'] = entropy
    
    return features

# Apply feature extraction
print("Extracting features...")
features_df = df['url'].apply(extract_features).apply(pd.Series)
X = features_df
y = df['label']

# --- 3. Train/Test Split & Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluation ---
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

# --- 5. Explainability (SHAP) ---
print("\n--- Explainability (SHAP) ---")
# Since our dataset is tiny for the demo, we use a TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Let's explain the first test prediction
# (Note: shap_values[1] is the impact on the Phishing class)
print(f"Explaining prediction for URL: {df.iloc[X_test.index[0]]['url']}")
print(f"Model predicted: {'Phishing' if y_pred[0] == 1 else 'Benign'}")

# Print feature impacts
feature_names = X.columns
for i in range(len(feature_names)):
    impact = shap_values[1][0][i]
    if abs(impact) > 0:
        print(f" - {feature_names[i]}: {impact:.3f}")
