from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from src.explainability import Explainer

app = FastAPI(title="Credit Risk Prediction API")

# Load Model
model = joblib.load('models/best_model.joblib')

# Initialize Explainer (using a dummy sample for init speed)
# In production, load a saved representative background dataset
dummy_data = pd.read_csv('data/application_record.csv').sample(100)
explainer = Explainer(model, dummy_data)

class ApplicationData(BaseModel):
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int      # Negative value
    DAYS_EMPLOYED: int   # Negative value or 365243
    CNT_FAM_MEMBERS: float

@app.post("/predict")
def predict_credit_risk(data: ApplicationData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    probability = model.predict_proba(input_df)[0, 1]
    prediction = int(probability > 0.5)
    
    # Explain
    lime_exp = explainer.explain_lime(input_df)
    
    return {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "probability": float(probability),
        "lime_explanation": lime_exp
    }