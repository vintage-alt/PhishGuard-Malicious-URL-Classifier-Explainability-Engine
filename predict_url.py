import pandas as pd
import joblib

def check_random_urls():
    print("Loading model...")
    model = joblib.load('phishing_rf_model.pkl')
    
    print("Loading dataset...")
    df = pd.read_csv("phishing_data.csv")
    df = df.dropna()
    
    # Pick 5 random URLs from the dataset
    sample = df.sample(5)
    
    # Prepare the features (drop text columns and label)
    X_sample = sample.drop(columns=['URL', 'Domain', 'TLD', 'Title', 'label'], errors='ignore')
    
    # Predict
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)
    
    # Print results
    for i in range(5):
        url = sample.iloc[i]['URL']
        actual_label = sample.iloc[i]['label']
        pred = predictions[i]
        conf = probabilities[i][pred] * 100
        
        status = "PHISHING 🚨" if pred == 1 else "BENIGN ✅"
        actual = "Phishing" if actual_label == 1 else "Benign"
        
        print(f"\nURL: {url}")
        print(f"Prediction: {status} (Confidence: {conf:.2f}%)")
        print(f"Actual Label: {actual}")

if __name__ == "__main__":
    check_random_urls()