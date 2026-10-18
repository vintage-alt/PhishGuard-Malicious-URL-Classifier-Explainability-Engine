# PhishGuard: Malicious URL Classifier & Explainability Engine

## Overview
PhishGuard is a classical Machine Learning pipeline designed to classify URLs as either benign or phishing attempts based on lexical and structural features. Using a Random Forest classifier trained on the PhiUSIIL dataset (235,000+ URLs), the model achieves an 82.29% overall accuracy while maintaining a 95% recall rate for malicious URLs. 

Crucially, this project implements **SHAP (SHapley Additive exPlanations)** to demystify the "black box" of machine learning, providing exact mathematical reasoning for why a specific URL was flagged.

## Motivation & Architectural Choices
In my previous project (AI-Assisted Nmap Analyzer), I utilized a Large Language Model (LLM) for its reasoning capabilities. For this project, I intentionally chose **Classical Machine Learning (Random Forest)** over an LLM for the following engineering reasons:
1. **Speed & Cost:** Evaluating 235,000 URLs with an LLM requires immense GPU compute. A Random Forest evaluates thousands of URLs per second on a standard laptop CPU.
2. **Tabular Data:** LLMs excel at natural language, but Random Forests are mathematically superior for classification based on structured numerical features (length, entropy, character counts).
3. **Explainability:** Random Forests allow for deterministic feature attribution via SHAP, which is vital for security analysts who need to know *why* a URL was blocked.

## Feature Engineering
The model does not process raw text. It extracts numerical features from the URLs, including:
- URL Length, number of dots, hyphens, and digits.
- Presence of an IP address (e.g., `192.168.1.1`) instead of a domain name.
- Presence of suspicious keywords (`login`, `verify`, `secure`, `account`, `bank`, etc.).
- **Shannon Entropy:** A mathematical measure of randomness in the URL string.

## Results & Evaluation
The dataset was split into an 80% training set and a 20% test set. 
- **Overall Accuracy:** 82.29%
- **Precision (Phishing):** 79%
- **Recall (Phishing):** 95%

**Why optimize for Recall?** In cybersecurity, a False Negative (letting a phishing URL through) is far more dangerous than a False Positive (flagging a benign URL). The model was tuned to aggressively catch malicious URLs, successfully identifying 95% of all phishing attempts in the test set.

## Explainability (SHAP) Example
When the model flagged `https://www.northcm.ac.th` as phishing, SHAP revealed the exact feature weights that triggered the alarm:
- `length` (+0.079): The URL is unusually long.
- `num_digits` (+0.081): The URL contains an abnormal amount of numbers.
- `num_hyphens` (+0.029): Suspicious use of hyphens.
- `entropy` (-0.054): Reduced the risk slightly (the characters were less random than a typical obfuscated URL).

## How to Run
1. Download the [PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset) and save it as `phishing_data.csv`.
2. Install dependencies: `pip install pandas scikit-learn shap`
3. Run the training script: `python train_real_model.py`
