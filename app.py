import streamlit as st
import pandas as pd
import joblib

# Import custom transformer before loading the model
from custom_transformers import InvalidWatchTimeHandler

st.set_page_config(
    page_title="Netflix Customer Churn Dashboard",
    page_icon="🎬",
    layout="wide"
)
with st.sidebar:
    st.title("🎬 Churn Dashboard")

    st.header("About the Project")

    st.write(
        "A machine learning application that predicts "
        "customer churn using XGBoost."
    )

    st.divider()

    st.subheader("Model Performance")

    st.metric("Test Accuracy", "99.30%")
    st.metric("Test F1-Score", "99.30%")
    st.metric("Test ROC-AUC", "0.9997")

    st.divider()

    st.caption(
        "Built with Python, Scikit-learn, "
        "XGBoost and Streamlit."
    )
st.title("🎬 Customer Churn Prediction Dashboard")

st.write(
    "Predict customer churn using a trained XGBoost model."
)

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

st.success("Model loaded successfully!")



st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"]
    )

    region = st.selectbox(
        "Region",
        ["Asia", "Europe", "North America",
         "South America", "Africa", "Oceania"]
    )

    watch_hours = st.number_input(
        "Watch Hours",
        min_value=0.0,
        value=10.0
    )

    last_login_days = st.number_input(
        "Days Since Last Login",
        min_value=0,
        value=10
    )

with col2:
    monthly_fee = st.number_input(
        "Monthly Fee",
        min_value=0.0,
        value=13.99
    )

    number_of_profiles = st.number_input(
        "Number of Profiles",
        min_value=1,
        max_value=10,
        value=2
    )

    avg_watch_time_per_day = st.number_input(
        "Average Watch Time Per Day (Hours)",
        min_value=0.0,
        max_value=24.0,
        value=1.0
    )

    device = st.selectbox(
        "Device",
        ["Mobile", "TV", "Laptop", "Tablet", "Desktop"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Debit Card", "PayPal",
         "Crypto", "Gift Card"]
    )

    favorite_genre = st.selectbox(
        "Favorite Genre",
        ["Action", "Comedy", "Drama", "Horror",
         "Romance", "Sci-Fi", "Documentary"]
    )


    # Create a prediction button
if st.button("Predict Customer Churn"):

    # Convert customer inputs into a DataFrame
    customer_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "subscription_type": subscription_type,
        "region": region,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "monthly_fee": monthly_fee,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day,
        "device": device,
        "payment_method": payment_method,
        "favorite_genre": favorite_genre
    }])

    # Predict churn class
    prediction = model.predict(customer_data)[0]

    # Predict probability of churn
    probability = model.predict_proba(customer_data)[0][1]

    # Display results
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Churn Risk: Customer is predicted to churn.")
    else:
        st.success("Low Churn Risk: Customer is predicted to stay.")

    st.metric(
        label="Predicted Churn Probability",
        value=f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

        # Customer retention recommendations
    st.subheader("Retention Recommendations")

    recommendations = []

    if last_login_days > 30:
        recommendations.append(
            "Send a personalized re-engagement notification."
        )

    if watch_hours < 5:
        recommendations.append(
            f"Recommend popular {favorite_genre} content "
            "to encourage engagement."
        )

    if subscription_type == "Basic":
        recommendations.append(
            "Highlight relevant benefits of other subscription plans."
        )

    if probability >= 0.7:
        recommendations.append(
            "Prioritize this customer for a personalized "
            "retention campaign."
        )

    if recommendations:
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")
    else:
        st.info(
            "No specific retention action identified "
            "from the current rules."
        )

st.divider()

with st.expander("About the Prediction Model"):
    st.write("""
    **Algorithm:** XGBoost Classifier

    **Dataset:** Netflix-like customer churn dataset

    **Training observations:** 4,000

    **Testing observations:** 1,000

    **Features:** 12 customer attributes

    **Preprocessing:**
    - Invalid watch-time handling
    - Median imputation for numerical features
    - Most-frequent imputation for categorical features
    - One-Hot Encoding

    **Model Evaluation:**
    - 5-fold stratified cross-validation
    - Accuracy, Precision, Recall, F1-score and ROC-AUC
    """)

    st.info(
        "This application is a demonstration using a "
        "synthetic dataset. Predictions are estimates "
        "and should not be interpreted as actual "
        "Netflix customer behavior."
    )