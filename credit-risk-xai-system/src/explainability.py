import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

class Explainer:
    def __init__(self, model, X_train_sample):
        """
        model: The full sklearn pipeline
        X_train_sample: A sample of the training data to initialize SHAP/LIME background
        """
        self.pipeline = model
        self.model_step = model.named_steps['classifier']
        self.preprocessor = model.named_steps['preprocessor']
        self.feature_eng = model.named_steps['feature_eng']
        
        # Transform the background sample
        self.X_transformed = self.preprocessor.transform(self.feature_eng.transform(X_train_sample))
        
        # Get feature names
        try:
            self.feature_names = self.preprocessor.get_feature_names_out()
        except:
            self.feature_names = [f"feat_{i}" for i in range(self.X_transformed.shape[1])]

    def explain_shap(self, row_df):
        """Returns SHAP values for a single prediction."""
        row_transformed = self.preprocessor.transform(self.feature_eng.transform(row_df))
        
        # Use KernelExplainer as a generic fallback (works for any model)
        # Note: In production with XGBoost, TreeExplainer is faster but this is safer for now.
        explainer = shap.KernelExplainer(self.model_step.predict_proba, self.X_transformed)
        shap_values = explainer.shap_values(row_transformed)
        
        # Handle SHAP return type (list for classification, array for regression)
        if isinstance(shap_values, list):
            # For binary classification, index 1 usually corresponds to the "positive" class
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        return {
            "shap_values": vals.tolist(),
            "feature_names": self.feature_names.tolist()
        }

    def explain_lime(self, row_df):
        """Returns LIME explanation as list of tuples."""
        # 1. Transform the input row
        transformed_data = self.preprocessor.transform(self.feature_eng.transform(row_df))
        
        # 2. Check if input is sparse (needs .toarray) or dense (use as is)
        if hasattr(transformed_data, "toarray"):
            row_transformed = transformed_data.toarray()[0]
        else:
            row_transformed = transformed_data[0]
        
        # 3. Check if background data is sparse or dense
        training_data = self.X_transformed
        if hasattr(training_data, "toarray"):
            training_data = training_data.toarray()
            
        # 4. Initialize LIME
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=['Good', 'Bad'],
            mode='classification'
        )
        
        # 5. Explain
        exp = lime_explainer.explain_instance(
            data_row=row_transformed, 
            predict_fn=self.model_step.predict_proba
        )
        return exp.as_list()