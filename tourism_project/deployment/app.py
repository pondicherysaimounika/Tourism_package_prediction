import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Debugdemon/Tourism-package-prediction/tourism-package-prediction-model", filename="tourism_package_prediction_model.joblib")
model = joblib.load(model_path)

# Streamlit UI

st.title("Tourism Package Prediction")


# User Inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

citytier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Free Lancer", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

numberofpersonvisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

numberoftrips = st.number_input(
    "Number Of Trips Per Year",
    min_value=0,
    max_value=50,
    value=2
)

passport = st.selectbox(
    "Passport Available",
    [0, 1]
)

owncar = st.selectbox(
    "Owns Car",
    [0, 1]
)

numberofchildrenvisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=1000000,
    value=30000
)


# Assemble input data

input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome
}])


# Prediction button

if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success("Customer is likely to purchase the tourism package.")
    else:
        st.error("Customer is unlikely to purchase the tourism package.")
