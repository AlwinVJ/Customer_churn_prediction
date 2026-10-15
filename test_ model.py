import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/customer_churn_pipeline.joblib")

model = joblib.load(MODEL_PATH)

sample_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0
}])

prediction = model.predict(sample_customer)[0]
probability = model.predict_proba(sample_customer)[0, 1]

print("Prediction:", "Churn" if prediction == 1 else "No Churn")
print(f"Estimated churn probability: {probability:.2%}")