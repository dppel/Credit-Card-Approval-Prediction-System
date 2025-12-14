import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from src.preprocessing import get_target, build_preprocessor, FeatureEngineer

def train_models():
    # Load and prep data
    print("Loading data...")
    df = get_target('data/application_record.csv', 'data/credit_record.csv')
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Feature Engineering explicitly applied before split or inside pipeline
    # Here we put it inside the pipeline
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Construct full pipeline
        pipeline = Pipeline(steps=[
            ('feature_eng', FeatureEngineer()),
            ('preprocessor', build_preprocessor(X_train)), # Note: fit happens inside pipeline
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        print(f"--- {name} Results ---")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc:.4f}")
        print("-" * 30)
        
        if roc > best_score:
            best_score = roc
            best_model = pipeline

    print(f"Saving best model with ROC AUC: {best_score:.4f}")
    joblib.dump(best_model, 'models/best_model.joblib')

if __name__ == "__main__":
    train_models()