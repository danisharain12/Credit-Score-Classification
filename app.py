import streamlit as st
import gdown
import os
import joblib
import pandas as pd

# File IDs and URLs
FILE_ID = "1MclSppXYEi6VPOpNvkHII-Cq-_ZOuam_"  # YOUR FILE ID HERE
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Local path to save after downloading
model_file = "Credit_Score_Classification.joblib"

# Download if not already downloaded
if not os.path.exists(model_file):
    st.info("Downloading from Google Drive...")
    gdown.download(URL, model_file, quiet=False)
    st.success("Model downloaded successfully.")

# Confirm and load the model
if os.path.exists(model_file):

    # Loading the model
    try:
        model = joblib.load(model_file)
    except Exception as e:
        st.error(f"Error while loading the model: {e}")

        st.stop()
else:
    st.error("Model not found after attempted download.")
    st.stop()


st.title("Credit Score Classification")
st.write("Analyze financial data to predict Credit Score Classification (Poor, Standard, or Good)")

st.info("Adjust the inputs and click **Predict Credit Score Classification**")

st.divider()

st.subheader("Provide Credit-related Information")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, max_value=1000000.0, value=809.98)
        credit_mix = st.number_input("Credit Mix ( Number of Credit Types)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        changed_credit = st.number_input("Change in Credit Limit (%)", min_value=0.0, max_value=100.0, value=11.27)
        amount_invested = st.number_input("Amount Invested Monthly ($)", min_value=0.0, max_value=1000000.0, value=80.42)
        num_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=100, value=4)
        total_emi = st.number_input("Total EMI per month ($)", min_value=0.0, max_value=1000000.0, value=49.57)
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, max_value=1000000.0, value=19114.12)

    with col2:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.5)
        delay = st.number_input("Delay From Due Date (days)", min_value=0.0, max_value=100.0, value=3.0)
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, max_value=1000000.0, value=312.490089)
        credit_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=26.83)
        num_delayed = st.number_input("Number of Delayed Payments", min_value=0, max_value=100, value=7)
        num_credit_card = st.number_input("Number of Credit Cards(s)", min_value=0, max_value=100, value=4)
        monthly_inhand = st.number_input("Monthly In-Hand Salary ($)", min_value=0.0, max_value=1000000.0, value=1824.83)


st.divider()
st.success("Click the button below to predict your Credit Score Classification")

# Perform prediction
if st.button("Predict Credit Score"):

    input_df = pd.DataFrame([{
        "Outstanding_Debt": outstanding_debt,
        "Interest_Rate_%": interest_rate,
        "Credit_Mix": credit_mix,
        "Delay_from_due_date": delay,
        "Changed_Credit_Limit_%": changed_credit,
        "Monthly_Balance": monthly_balance,
        "Amount_invested_monthly": amount_invested,
        "Credit_Utilization_Ratio": credit_ratio,
        "Num_Credit_Inquiries": num_inquiries,
        "Num_of_Delayed_Payment": num_delayed,
        "Total_EMI_per_month": total_emi,
        "Num_Credit_Card": num_credit_card,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand,
    }])

    predicted_code = model.predict(input_df)[0]
    label_map = {0: "Poor", 1: "Standard", 2: "Good"}
    predicted_label = label_map[predicted_code]

    if predicted_label == "Good":
        st.success(f"Credit Score Classification: **{predicted_label}**")
    elif predicted_label == "Standard":
        st.info(f"ℹ Credit Score Classification: **{predicted_label}**")
    else:
        st.error(f"Credit Score Classification: **{predicted_label}**")

    st.write("## Input Data")
    st.dataframe(input_df)


with st.expander("ℹ About this Model and App"):
    st.write("""
This Credit Score Classification Model was trained with historical financial data.  
It assesses financial stability and classifies the score into **Poor**, **Standard**, or **Good**.  
The algorithm considers multiple financial indicators.

**Task:**  
- You are working as a data scientist in a global finance company.  
- The company wants to reduce manual efforts by building a smart system for Credit Score Classification.  
- This application helps streamline the process by evaluating financial data efficiently.
    """)

