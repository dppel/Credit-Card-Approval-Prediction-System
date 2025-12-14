import numpy as np  
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.explainability import Explainer

# Setup
st.set_page_config(page_title="Credit Card Approval AI", layout="wide")

@st.cache_resource
def load_system():
    model = joblib.load('models/best_model.joblib')
    # Load background data for SHAP (using small sample for performance)
    background_data = pd.read_csv('data/application_record.csv').sample(100)
    explainer = Explainer(model, background_data)
    return model, explainer

model, explainer = load_system()

st.title("🏦 Credit Card Approval Prediction System")
st.markdown("XGBoost-based inference with SHAP & LIME Interpretability")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Applicant Details")
    # Manual Input Form
    gender = st.selectbox("Gender", ["M", "F"])
    car = st.selectbox("Owns Car?", ["Y", "N"])
    realty = st.selectbox("Owns Realty?", ["Y", "N"])
    children = st.number_input("Children", 0, 10, 0)
    income = st.number_input("Total Income", 10000, 1000000, 50000)
    income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
    edu = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
    family = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
    housing = st.selectbox("Housing Type", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
    age = st.slider("Age", 18, 70, 30)
    employed_years = st.slider("Years Employed", 0, 40, 5)
    fam_members = st.number_input("Family Members", 1, 15, 2)
    
    # Convert simplified inputs back to model format
    input_data = {
        'CODE_GENDER': [gender],
        'FLAG_OWN_CAR': [car],
        'FLAG_OWN_REALTY': [realty],
        'CNT_CHILDREN': [children],
        'AMT_INCOME_TOTAL': [income],
        'NAME_INCOME_TYPE': [income_type],
        'NAME_EDUCATION_TYPE': [edu],
        'NAME_FAMILY_STATUS': [family],
        'NAME_HOUSING_TYPE': [housing],
        'DAYS_BIRTH': [-age * 365],
        'DAYS_EMPLOYED': [-employed_years * 365],
        'CNT_FAM_MEMBERS': [fam_members],
        'FLAG_MOBIL': [1], 'FLAG_WORK_PHONE': [0], 'FLAG_PHONE': [0], 'FLAG_EMAIL': [0] # Dummies
    }
    
    if st.button("Predict Risk"):
        input_df = pd.DataFrame(input_data)
        
        # Prediction
        prob = model.predict_proba(input_df)[0, 1]
        
        st.write("---")
        
        # 👇 FIND THIS LINE 👇
        st.metric("Risk Probability", f"{prob:.2%}")
        
        # 👇 PASTE THE NEW CODE HERE 👇
        st.progress(int(prob * 100))
        
        if prob > 0.5:
            st.error(f"🚫 High Credit Risk - Rejected (Confidence: {prob:.1%})")
        else:
            st.success(f"✅ Low Credit Risk - Approved (Confidence: {(1-prob):.1%})")

with col2:
    if 'input_df' in locals():
        st.header("Explainability Analysis")
        
        tab1, tab2 = st.tabs(["LIME (Local)", "SHAP (Global/Local)"])
        
        with tab1:
            st.subheader("LIME: Why this prediction?")
            lime_exp = explainer.explain_lime(input_df)
            
            # Create DataFrame for chart
            lime_df = pd.DataFrame(lime_exp, columns=['Feature', 'Weight'])
            lime_df = lime_df.sort_values(by='Weight', ascending=True)
            
            fig, ax = plt.subplots()
            colors = ['red' if x < 0 else 'green' for x in lime_df['Weight']]
            ax.barh(lime_df['Feature'], lime_df['Weight'], color=colors)
            ax.set_xlabel("Contribution to High Risk")
            st.pyplot(fig)
            
        with tab2:
            st.subheader("SHAP Force Plot")
            shap_data = explainer.explain_shap(input_df)
            
            st.info("SHAP visualizes how each feature pushed the prediction away from the average.")
            
            # 1. Convert to numpy arrays explicitly
            feature_names = np.array(shap_data['feature_names'])
            values = np.array(shap_data['shap_values'])
            
            # 2. Smart Reshaping
            if values.ndim == 2:
                values = values.flatten()
            elif values.ndim == 3:
                values = values[0, :, -1] 
            
            # 3. CRITICAL: Cut both arrays to the length of the shortest one
            min_len = min(len(feature_names), len(values))
            feature_names = feature_names[:min_len]
            values = values[:min_len]
            
            # 4. Sort and Display
            indices = np.argsort(np.abs(values))[::-1][:10] # Top 10
            
            shap_disp = pd.DataFrame({
                'Feature': feature_names[indices],
                'Impact': values[indices]
            })
            
            # Use a bar chart
            st.bar_chart(shap_disp.set_index('Feature'))
            
            st.dataframe(shap_disp)