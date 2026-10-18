import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
import joblib

print("Loading dataset...")
df = pd.read_csv("phishing_data.csv")

columns_to_drop = ['URL', 'Domain', 'TLD', 'Title']
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

df = df.dropna()

X = df.drop(columns=['label'])
y = df['label']

print(f"Training on {X.shape[1]} features and {len(df)} URLs.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest on real data...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\n--- Explainability (SHAP) ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:1])

print(f"Explaining prediction for URL index: {X_test.index[0]}")
prediction = y_pred[0]
print(f"Model predicted: {'Phishing' if prediction == 1 else 'Benign'}")

feature_impacts = list(zip(X.columns, shap_values[1][0]))
feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top 5 features driving this decision:")
for feature, impact in feature_impacts[:5]:
    print(f" - {feature}: {impact:.3f}")

joblib.dump(model, 'phishing_rf_model.pkl')
print("\nModel successfully saved as 'phishing_rf_model.pkl'")
