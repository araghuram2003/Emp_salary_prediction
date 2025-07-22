import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (must match model's 13 feature columns)
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 90, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Local-gov", "NotListed", "State-gov", "Self-emp-inc", "Federal-gov"
])
fnlwgt = st.sidebar.number_input("fnlwgt (final weight)", min_value=10000, max_value=1000000, value=150000)
educational_num = st.sidebar.slider("Education Number (educational-num)", 1, 16, 9)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.selectbox("Gender", [
    "Male", "Female"
])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Jamaica", "South", "China", "Italy", "Dominican-Republic", "Vietnam", "Guatemala", "Japan", "Poland", "Columbia", "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua", "Peru", "Greece", "France", "Ecuador", "Ireland", "Hong", "Thailand", "Cambodia", "Trinadad&Tobago", "Laos", "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Scotland", "Honduras", "Hungary", "Holand-Netherlands"
])

# Build input DataFrame in correct order
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Define mappings for all categorical columns (must match training)
workclass_mapping = {
    "Private": 0, "Self-emp-not-inc": 1, "Local-gov": 2, "NotListed": 3, "State-gov": 4, "Self-emp-inc": 5, "Federal-gov": 6
}
marital_status_mapping = {
    "Married-civ-spouse": 0, "Never-married": 1, "Divorced": 2, "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5, "Married-AF-spouse": 6
}
occupation_mapping = {
    'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Other-service': 7, 'Priv-house-serv': 8, 'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11, 'Tech-support': 12, 'Transport-moving': 13
}
relationship_mapping = {
    "Husband": 0, "Not-in-family": 1, "Own-child": 2, "Unmarried": 3, "Wife": 4, "Other-relative": 5
}
race_mapping = {
    "White": 0, "Black": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4
}
gender_mapping = {
    "Male": 0, "Female": 1
}
native_country_mapping = {
    "United-States": 0, "Mexico": 1, "Philippines": 2, "Germany": 3, "Canada": 4, "Puerto-Rico": 5, "El-Salvador": 6, "India": 7, "Cuba": 8, "England": 9, "Jamaica": 10, "South": 11, "China": 12, "Italy": 13, "Dominican-Republic": 14, "Vietnam": 15, "Guatemala": 16, "Japan": 17, "Poland": 18, "Columbia": 19, "Taiwan": 20, "Haiti": 21, "Iran": 22, "Portugal": 23, "Nicaragua": 24, "Peru": 25, "Greece": 26, "France": 27, "Ecuador": 28, "Ireland": 29, "Hong": 30, "Thailand": 31, "Cambodia": 32, "Trinadad&Tobago": 33, "Laos": 34, "Yugoslavia": 35, "Outlying-US(Guam-USVI-etc)": 36, "Scotland": 37, "Honduras": 38, "Hungary": 39, "Holand-Netherlands": 40
}

# Encode categorical columns
input_df['workclass'] = input_df['workclass'].map(workclass_mapping)
input_df['marital-status'] = input_df['marital-status'].map(marital_status_mapping)
input_df['occupation'] = input_df['occupation'].map(occupation_mapping)
input_df['relationship'] = input_df['relationship'].map(relationship_mapping)
input_df['race'] = input_df['race'].map(race_mapping)
input_df['gender'] = input_df['gender'].map(gender_mapping)
input_df['native-country'] = input_df['native-country'].map(native_country_mapping)

# Label encoding mappings (must match training order)
education_mapping = {
    'Bachelors': 0,
    'HS-grad': 1,
    'Masters': 2,
    'PhD': 3,
    'Assoc': 4,
    'Some-college': 5
}
occupation_mapping = {
    'Adm-clerical': 0,
    'Armed-Forces': 1,
    'Craft-repair': 2,
    'Exec-managerial': 3,
    'Farming-fishing': 4,
    'Handlers-cleaners': 5,
    'Machine-op-inspct': 6,
    'Other-service': 7,
    'Priv-house-serv': 8,
    'Prof-specialty': 9,
    'Protective-serv': 10,
    'Sales': 11,
    'Tech-support': 12,
    'Transport-moving': 13
}

st.write("### 🔎 Input Data (Encoded)")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    # Apply the same mappings for batch data
    for col, mapping in [
        ('workclass', workclass_mapping),
        ('marital-status', marital_status_mapping),
        ('occupation', occupation_mapping),
        ('relationship', relationship_mapping),
        ('race', race_mapping),
        ('gender', gender_mapping),
        ('native-country', native_country_mapping)
    ]:
        batch_data[col] = batch_data[col].map(mapping)
    batch_preds = model.predict(batch_data[input_df.columns])
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

