# 💳 Credit Risk ML System with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/FastAPI-0.95-green)
![Frontend](https://img.shields.io/badge/Streamlit-1.25-red)
![ML](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-orange)
![License](https://img.shields.io/badge/License-MIT-grey)

A complete machine learning system for credit approval prediction. This project demonstrates an end-to-end pipeline including data engineering, model training, API deployment, and real-time explainability using **SHAP** and **LIME**.

---

## 🎯 Overview

Financial institutions need transparent decision-making systems. This project predicts whether an applicant is a **"High Risk"** or **"Low Risk"** borrower and, crucially, **explains why** the decision was made using Explainable AI techniques.

### 📸 Dashboard Preview
![Streamlit Dashboard](assets/dashboard_screenshot.png)
*(Note: Replace this line with a screenshot of your Streamlit App showing a prediction)*

---

## ✨ Key Features

* **Machine Learning Pipeline:** Automated preprocessing and feature engineering.
* **Explainable AI (XAI):**
    * **SHAP (Global):** Visualizes overall feature impact.
    * **LIME (Local):** Explains individual predictions in plain English.
* **Backend:** High-performance REST API using **FastAPI**.
* **Frontend:** Interactive dashboard using **Streamlit**.
* **Production Focus:** Modular code structure suitable for deployment.

---

## 🏗️ Architecture & File Structure

The project follows a modular architecture ensuring separation of concerns:

```text
├── app/
│   └── streamlit_app.py      # Frontend Dashboard
├── src/
│   ├── api.py                # FastAPI Backend
│   ├── model_training.py     # ML Training Script
│   ├── preprocessing.py      # Data Cleaning & Feature Eng.
│   └── explainability.py     # SHAP/LIME Logic
├── models/
│   └── random_forest.pkl     # Serialized Model
├── notebooks/                # Jupyter Notebooks for experimentation
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
