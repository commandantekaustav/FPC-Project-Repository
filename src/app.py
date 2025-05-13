import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model, Columns, and Feature Type Lists (KEEP THIS AS IS) ---
@st.cache_resource
def load_model_pipeline(path):
    try:
        pipeline = joblib.load(path)
        return pipeline
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please ensure it's in the correct 'models' directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_joblib_file(path, file_description="file"):
    try:
        data = joblib.load(path)
        return data
    except FileNotFoundError:
        st.error(f"{file_description.capitalize()} not found at {path}. Ensure it's in 'models' directory.")
        return None
    except Exception as e:
        st.error(f"Error loading {file_description}: {e}")
        return None

MODEL_PATH = "models/attrition_rf_pipeline_v1.joblib"
COLUMNS_PATH = "models/X_train_columns_v1.joblib"
NUM_FEATURES_PATH = "models/numerical_features_v1.joblib"
CAT_FEATURES_PATH = "models/categorical_features_v1.joblib"

pipeline = load_model_pipeline(MODEL_PATH)
X_train_columns = load_joblib_file(COLUMNS_PATH, "X_train column list")
numerical_features_list = load_joblib_file(NUM_FEATURES_PATH, "numerical features list")
categorical_features_list = load_joblib_file(CAT_FEATURES_PATH, "categorical features list")

if pipeline is None or X_train_columns is None or numerical_features_list is None or categorical_features_list is None:
    st.error("One or more essential model files could not be loaded. Application cannot proceed.")
    st.stop()

# --- UI Design ---
st.title("🧑‍💼 Employee Attrition Prediction")
st.markdown("""
This application predicts the likelihood of an employee leaving the company.
Please provide the employee's details in the sidebar.
""")

# --- Define Sidebar Inputs ONCE and Get Their Current Values ---
# The widgets are created here. Their current values are read directly.
st.sidebar.header("👤 Employee Details")

# Use unique keys for each widget
age = st.sidebar.slider("Age", 18, 65, 30, key="age_slider")
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000, step=100, key="income_input")
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10, key="total_years_slider")

max_years_at_company = total_working_years if total_working_years > 0 else 40
current_years_at_company_default = 5 if 5 <= max_years_at_company else max_years_at_company
years_at_company = st.sidebar.slider("Years at Company", 0, max_years_at_company, current_years_at_company_default, key="company_years_slider")

distance_from_home = st.sidebar.slider("Distance From Home (km)", 1, 30, 5, key="distance_slider")
percent_salary_hike = st.sidebar.slider("Percent Salary Hike (%)", 10, 25, 15, key="hike_slider")
num_companies_worked = st.sidebar.slider("Number of Companies Worked At", 0, 10, 1, key="num_companies_slider")

business_travel_options = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
business_travel = st.sidebar.selectbox("Business Travel", business_travel_options, index=0, key="bt_select")
department_options = ['Research & Development', 'Sales', 'Human Resources']
department = st.sidebar.selectbox("Department", department_options, index=0, key="dept_select")
education_field_options = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other']
education_field = st.sidebar.selectbox("Education Field", education_field_options, index=0, key="edu_field_select")
gender_options = ['Male', 'Female']
gender = st.sidebar.selectbox("Gender", gender_options, index=0, key="gender_select")
job_role_options = [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative', 'Manager',
    'Sales Representative', 'Research Director', 'Human Resources'
]
job_role = st.sidebar.selectbox("Job Role", job_role_options, index=0, key="job_role_select")
marital_status_options = ['Married', 'Single', 'Divorced']
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options, index=0, key="marital_select")
overtime_options = ['No', 'Yes']
over_time = st.sidebar.selectbox("OverTime", overtime_options, index=0, key="overtime_select")


# --- Prediction Logic ---
if st.sidebar.button("🔮 Predict Attrition"):
    # Construct the input_data dictionary using the current values from the sidebar widgets
    input_data_for_prediction = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'TotalWorkingYears': total_working_years,
        'YearsAtCompany': years_at_company,
        'DistanceFromHome': distance_from_home,
        'PercentSalaryHike': percent_salary_hike,
        'NumCompaniesWorked': num_companies_worked,
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'Gender': gender,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        'OverTime': over_time
    }

    # Apply defaults for features NOT in the UI, using the X_train_columns list
    explicit_defaults = {
        'DailyRate': 800, 'Education': 3, 'EnvironmentSatisfaction': 3, 'HourlyRate': 65,
        'JobInvolvement': 3, 'JobLevel': 2, 'JobSatisfaction': 3, 'MonthlyRate': 14000,
        'PerformanceRating': 3, 'RelationshipSatisfaction': 3, 'StockOptionLevel': 1,
        'TrainingTimesLastYear': 3, 'YearsInCurrentRole': 3, 'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 3
    }

    for col in X_train_columns:
        if col not in input_data_for_prediction: # If not set by the UI widgets above
            if col in explicit_defaults:
                input_data_for_prediction[col] = explicit_defaults[col]
            elif col in numerical_features_list:
                input_data_for_prediction[col] = 0
            elif col in categorical_features_list:
                input_data_for_prediction[col] = "Unknown"
            else:
                st.error(f"Feature '{col}' is of unknown type. Defaulting to np.nan.")
                input_data_for_prediction[col] = np.nan

    # Create DataFrame for prediction
    single_row_data = {c: input_data_for_prediction.get(c) for c in X_train_columns}
    input_df = pd.DataFrame([single_row_data], columns=X_train_columns)

    # Optional: Sanity check
    with st.expander("DEBUG: Input DataFrame Details (Inside Button Click)"):
        st.write("Input DataFrame to pipeline (built inside button click):")
        st.dataframe(input_df)
        st.write("Data types:")
        st.json(input_df.dtypes.apply(lambda x: str(x)).to_dict())

    try:
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]

        st.subheader("📊 Prediction Result:")
        if prediction == 1:
            st.error(f"**High Risk of Attrition** (Probability: {probability[1]*100:.2f}%)")
        else:
            st.success(f"**Low Risk of Attrition** (Probability: {probability[1]*100:.2f}%)")

        with st.expander("🔍 Show Prediction Probabilities"):
            st.write(f"Probability of No Attrition (0): {probability[0]*100:.2f}%")
            st.write(f"Probability of Attrition (1): {probability[1]*100:.2f}%")

        with st.expander("📝 Show Input Data Used for Prediction (Refreshed)"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)

else:
    st.info("Adjust employee details in the sidebar and click 'Predict Attrition'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Project by: [Your Name]")
st.sidebar.markdown("Source Code: [GitHub Repository Link]")
st.sidebar.markdown("Contact: [Your Email]")
st.sidebar.markdown("Version: 1.0")
st.sidebar.markdown("Last Updated: October 2023")
st.sidebar.markdown("**Disclaimer:** This is a demo application for educational purposes.")
st.sidebar.markdown("**Note:** The model is trained on synthetic data and may not reflect real-world scenarios.")