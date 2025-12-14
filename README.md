# 💳 Credit Risk ML System with Explainable AI (XAI)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io/)

A complete machine learning system for credit approval prediction. This project demonstrates an end-to-end pipeline including data engineering, model training (XGBoost/RandomForest), API deployment, and real-time explainability using SHAP and LIME.

## 🎯 Overview

Financial institutions need transparent decision-making systems. This project predicts whether an applicant is a "High Risk" or "Low Risk" borrower and, crucially, **explains why** the decision was made.

### Key Features
* **Machine Learning:** Automated feature engineering pipeline.
* **Explainable AI (XAI):**
    * **SHAP (Global):** Visualizes overall feature impact.
    * **LIME (Local):** Explains individual predictions in plain English.
* **Backend:** High-performance REST API using **FastAPI**.
* **Frontend:** Interactive dashboard using **Streamlit**.
* **Production Ready:** Docker support and Unit Testing included.

## 🏗️ Architecture

1. **Ingestion:** Raw data processing & feature engineering.
2. **Training:** Model training (Random Forest/XGBoost) optimized for F1-score.
3. **Deployment:** FastAPI serves predictions & explanations via JSON.
4. **Interface:** Streamlit consumes the API for end-users.

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/YOUR_USERNAME/credit-risk-xai-system.git](https://github.com/YOUR_USERNAME/credit-risk-xai-system.git)
cd credit-risk-xai-system

# Create & Activate Virtual Environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### 2. Run the App
You can run the entire system (Frontend + Backend) using the provided script or Docker.

**Manual Method:**
```bash
# Terminal 1: Start API
python -m uvicorn src.api:app --reload

# Terminal 2: Start Dashboard
python -m streamlit run app/streamlit_app.py

📊 Data Source
[Kaggle Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

📝 License
MIT License
