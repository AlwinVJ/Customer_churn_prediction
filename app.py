import joblib
import pandas as pd
import streamlit as st
import sklearn
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "customer_churn_pipeline.joblib"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("Customer Churn Prediction")

st.markdown(
    """
    Enter customer details to estimate the likelihood of
    customer churn using the trained Logistic Regression model.

    This application demonstrates a machine learning workflow
    and should not be treated as a definitive business decision tool.
    """
)

st.divider()


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

st.header("1. Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    partner = st.selectbox(
        "Has Partner?",
        ["Yes", "No"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    dependents = st.selectbox(
        "Has Dependents?",
        ["Yes", "No"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

with col3:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


st.divider()


# --------------------------------------------------
# SERVICE INFORMATION
# --------------------------------------------------

st.header("2. Service Information")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


st.divider()


# --------------------------------------------------
# CONTRACT AND BILLING
# --------------------------------------------------

st.header("3. Contract and Billing")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=10000.0,
        value=70.0,
        step=1.0
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=1000000.0,
        value=840.0,
        step=10.0
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        prediction = int(model.predict(customer_data)[0])

        churn_probability = float(
            model.predict_proba(customer_data)[0, 1]
        )

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("Prediction: Customer may churn")
            else:
                st.success("Prediction: Customer may stay")

        with result_col2:
            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability:.2%}"
            )

        st.progress(churn_probability)

        st.caption(
            "The probability is a model estimate, not a guarantee. "
            "It depends on the model's training data and calibration."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Classification | "
    "Logistic Regression | Built with Streamlit"
)