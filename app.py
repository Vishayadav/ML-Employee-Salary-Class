import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")

st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Classification")
st.write("Predict whether an employee earns >50K or ≤50K.")

st.sidebar.header("👤 Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)

workclass_options = [
    "Federal-gov",
    "Local-gov",
    "Others",
    "Private",
    "Self-emp-inc",
    "Self-emp-not-inc",
    "State-gov"
]
workclass = st.sidebar.selectbox("Workclass", workclass_options)

fnlwgt = st.sidebar.number_input(
    "Final Weight (fnlwgt)",
    min_value=10000,
    max_value=1500000,
    value=200000
)

education_num = st.sidebar.slider(
    "Education Number", 5, 16, 9
)

marital_options = [
    "Divorced",
    "Married-AF-spouse",
    "Married-civ-spouse",
    "Married-spouse-absent",
    "Never-married",
    "Separated",
    "Widowed"
]
marital_status = st.sidebar.selectbox(
    "Marital Status", marital_options
)

occupation_options = [
    "Adm-clerical",
    "Armed-Forces",
    "Craft-repair",
    "Exec-managerial",
    "Farming-fishing",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Others",
    "Other-service",
    "Priv-house-serv",
    "Prof-specialty",
    "Protective-serv",
    "Sales",
    "Tech-support",
    "Transport-moving"
]
occupation = st.sidebar.selectbox(
    "Occupation", occupation_options
)

relationship_options = [
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Own-child",
    "Unmarried",
    "Wife"
]
relationship = st.sidebar.selectbox(
    "Relationship", relationship_options
)

race_options = [
    "Amer-Indian-Eskimo",
    "Asian-Pac-Islander",
    "Black",
    "Other",
    "White"
]
race = st.sidebar.selectbox("Race", race_options)

sex_options = ["Female", "Male"]
sex = st.sidebar.selectbox("Sex", sex_options)

capital_gain = st.sidebar.number_input(
    "Capital Gain", 0, 100000, 0
)

capital_loss = st.sidebar.number_input(
    "Capital Loss", 0, 10000, 0
)

hours_per_week = st.sidebar.slider(
    "Hours per Week", 1, 99, 40
)

country_options = [
    "?",
    "Cambodia",
    "Canada",
    "China",
    "Columbia",
    "Cuba",
    "Dominican-Republic",
    "Ecuador",
    "El-Salvador",
    "England",
    "France",
    "Germany",
    "Greece",
    "Guatemala",
    "Haiti",
    "Holand-Netherlands",
    "Honduras",
    "Hong",
    "Hungary",
    "India",
    "Iran",
    "Ireland",
    "Italy",
    "Jamaica",
    "Japan",
    "Laos",
    "Mexico",
    "Nicaragua",
    "Outlying-US(Guam-USVI-etc)",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Puerto-Rico",
    "Scotland",
    "South",
    "Taiwan",
    "Thailand",
    "Trinadad&Tobago",
    "United-States",
    "Vietnam",
    "Yugoslavia"
]

native_country = st.sidebar.selectbox(
    "Native Country", country_options
)

# --------------------------------------------------
# ENCODE EXACTLY AS THE NOTEBOOK DID
# LabelEncoder assigns alphabetical order starting from 0
# --------------------------------------------------

workclass_map = {
    value: i for i, value in enumerate(workclass_options)
}

marital_map = {
    value: i for i, value in enumerate(marital_options)
}

occupation_map = {
    value: i for i, value in enumerate(occupation_options)
}

relationship_map = {
    value: i for i, value in enumerate(relationship_options)
}

race_map = {
    value: i for i, value in enumerate(race_options)
}

sex_map = {
    value: i for i, value in enumerate(sex_options)
}

country_map = {
    value: i for i, value in enumerate(country_options)
}

# --------------------------------------------------
# CREATE NUMERIC INPUT
# --------------------------------------------------

input_df = pd.DataFrame({
    "age": [age],
    "workclass": [workclass_map[workclass]],
    "fnlwgt": [fnlwgt],
    "education.num": [education_num],
    "marital.status": [marital_map[marital_status]],
    "occupation": [occupation_map[occupation]],
    "relationship": [relationship_map[relationship]],
    "race": [race_map[race]],
    "sex": [sex_map[sex]],
    "capital.gain": [capital_gain],
    "capital.loss": [capital_loss],
    "hours.per.week": [hours_per_week],
    "native.country": [country_map[native_country]]
})

st.write("### 🔎 Input Data")
st.dataframe(input_df)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

if st.button("🔮 Predict Salary Class"):

    prediction = model.predict(input_df)

    st.success(
        f"✅ Prediction: {prediction[0]}"
    )
