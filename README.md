# ❤️ Heart Disease Prediction

A machine learning web app that predicts the **presence or absence of heart disease** from 13 clinical measurements.

Built with Python, scikit-learn and Streamlit.

## Problem
Given patient data such as age, blood pressure, cholesterol, chest pain type and ECG results, classify whether heart disease is present.

## Dataset
- 270 patient records, 13 features, 1 target (`Heart Disease`)
- Features: Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120, EKG results, Max HR, Exercise angina, ST depression, Slope of ST, Number of vessels (fluoroscopy), Thallium
- No missing values

## Approach
1. **Exploratory data analysis:** data types, statistics, missing value check
2. **Preprocessing:** one-hot encoding of categorical features (drop-first), StandardScaler on numerical features
3. **Train-test split:** 80/20
4. **Model comparison:** 6 algorithms trained and evaluated
5. **Overfitting check:** training vs testing accuracy plotted for every model
6. **Evaluation:** classification report, confusion matrix, ROC curve
7. **Deployment:** Streamlit web app

## Results

| Model | Test Accuracy |
|---|---|
| **Logistic Regression** | **88.9%** |
| KNN | 83.3% |
| Naive Bayes | 81.5% |
| SVM | 79.6% |
| Random Forest | 77.8% |
| Decision Tree | 64.8% |

**Logistic Regression** was selected as the final model. It had:
- the highest test accuracy (88.9%)
- **ROC AUC = 0.91**
- the best generalization: Random Forest and Decision Tree overfitted heavily

## Project Structure
```
├── Heart_Disease_Predict_Model.ipynb   # Full training notebook
├── Heart_Disease_Prediction.csv        # Dataset
├── streamlit_app.py                    # Web app
├── heart_disease_model.pkl             # Trained Logistic Regression model
├── model_columns.pkl                   # Feature column order
├── scaler.pkl
└── requirements.txt
```

## Run Locally
```bash
git clone https://github.com/Menuka-Jeewantha/Heart_Disease_Predict_app.git
cd Heart_Disease_Predict_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tech Stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit · joblib

## Disclaimer
This is an educational project and **not medical advice**. Always consult a qualified doctor.

## Author
**Menuka Jeewantha**, Electrical and Electronic Engineering undergraduate, University of Sri Jayewardenepura
