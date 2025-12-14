import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Convert Days Birth to Age (positive)
        X['Age'] = -X['DAYS_BIRTH'] // 365
        # Convert Days Employed to Years (handle magic number 365243)
        X['Years_Employed'] = X['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else -x // 365)
        
        # Drop unnecessary columns
        drop_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
        return X

def get_target(app_path, credit_path):
    """
    Constructs the target variable: 
    1 (High Risk) if overdue > 60 days in history, else 0 (Low Risk).
    """
    app = pd.read_csv(app_path)
    credit = pd.read_csv(credit_path)
    
    # Create target based on status '2', '3', '4', '5' (overdue > 60 days)
    bad_users = credit[credit['STATUS'].isin(['2', '3', '4', '5'])]['ID'].unique()
    
    app['Target'] = app['ID'].apply(lambda x: 1 if x in bad_users else 0)
    
    # Drop duplicates
    app = app.drop_duplicates(subset=['ID'], keep='first').drop('ID', axis=1)
    return app

def build_preprocessor(X):
    """Builds the Scikit-Learn ColumnTransformer"""
    numeric_features = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'Age', 'Years_Employed']
    categorical_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
                            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor