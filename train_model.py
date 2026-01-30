#!/usr/bin/env python3
"""
Customer Churn Prediction Model
End-to-end ML pipeline for predicting customer churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath='data/telecom_churn.csv'):
        """Load the dataset"""
        print("Loading data...")
        self.df = pd.read_csv(filepath)
        print(f"Data shape: {self.df.shape}")
        return self.df
    
    def exploratory_analysis(self):
        """Perform EDA"""
        print("\n=== Dataset Info ===")
        print(self.df.info())
        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())
        print("\n=== Statistical Summary ===")
        print(self.df.describe())
        print("\n=== Target Distribution ===")
        print(self.df['Churn'].value_counts())
        
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\nPreprocessing data...")
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Encode categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Churn']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            
        # Encode target variable
        if self.df['Churn'].dtype == 'object':
            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
            
        print("Preprocessing completed!")
        
    def feature_engineering(self):
        """Create new features"""
        print("\nPerforming feature engineering...")
        
        # Example: Create interaction features
        if 'tenure' in self.df.columns and 'MonthlyCharges' in self.df.columns:
            self.df['TenureMonthlyCharges'] = self.df['tenure'] * self.df['MonthlyCharges']
            
        print("Feature engineering completed!")
        
    def prepare_features(self):
        """Prepare X and y for modeling"""
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        return X, y
        
    def train_model(self, X_train, y_train):
        """Train multiple models and select the best"""
        print("\nTraining models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Try multiple models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_score = 0
        best_model_name = ''
        
        for name, model in models.items():
            scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            mean_score = scores.mean()
            print(f"{name} - ROC AUC: {mean_score:.4f} (+/- {scores.std():.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                self.model = model
                
        print(f"\nBest model: {best_model_name} with ROC AUC: {best_score:.4f}")
        
        # Train the best model on full training data
        self.model.fit(X_train_scaled, y_train)
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\n=== Model Evaluation ===")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC AUC Score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return roc_auc
        
    def save_model(self, filepath='models/churn_model.pkl'):
        """Save the trained model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filepath)
        print(f"\nModel saved to {filepath}")
        
    def load_model(self, filepath='models/churn_model.pkl'):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        print(f"Model loaded from {filepath}")
        
def main():
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load data (you'll need to provide your own dataset)
    # For demo purposes, we'll create sample data
    print("Creating sample dataset for demonstration...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    }
    
    predictor.df = pd.DataFrame(data)
    
    # EDA
    predictor.exploratory_analysis()
    
    # Preprocess
    predictor.preprocess_data()
    
    # Feature engineering
    predictor.feature_engineering()
    
    # Prepare features
    X, y = predictor.prepare_features()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    predictor.train_model(X_train, y_train)
    
    # Evaluate
    predictor.evaluate_model(X_test, y_test)
    
    # Save model
    predictor.save_model()
    
    print("\n=== Training Pipeline Completed ===")
    print("Next steps:")
    print("1. Replace sample data with your actual dataset")
    print("2. Fine-tune hyperparameters")
    print("3. Deploy the model using predict.py")
    
if __name__ == "__main__":
    main()
