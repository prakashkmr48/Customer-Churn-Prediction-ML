#!/usr/bin/env python3
"""
Prediction Script for Customer Churn Model
Load trained model and make predictions on new data
"""

import joblib
import pandas as pd
import numpy as np
import argparse
import sys

class ChurnPredictor:
    def __init__(self, model_path='models/churn_model.pkl'):
        """Initialize predictor with trained model"""
        try:
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            print("Please train the model first using train_model.py")
            sys.exit(1)
    
    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        # Handle categorical encoding if needed
        # This should match the preprocessing in training
        return data
    
    def predict(self, input_data):
        """Make predictions on input data"""
        # Preprocess
        processed_data = self.preprocess_input(input_data)
        
        # Scale features
        scaled_data = self.scaler.transform(processed_data)
        
        # Predict
        predictions = self.model.predict(scaled_data)
        probabilities = self.model.predict_proba(scaled_data)
        
        return predictions, probabilities
    
    def predict_single(self, tenure, monthly_charges, total_charges, contract, internet_service):
        """Predict churn for a single customer"""
        # Create DataFrame with single customer data
        data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'InternetService': [internet_service]
        })
        
        # Encode categorical variables (simplified)
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
        
        data['Contract'] = data['Contract'].map(contract_map)
        data['InternetService'] = data['InternetService'].map(internet_map)
        
        # Add feature engineering
        data['TenureMonthlyCharges'] = data['tenure'] * data['MonthlyCharges']
        
        # Predict
        prediction, probability = self.predict(data)
        
        return {
            'will_churn': bool(prediction[0]),
            'churn_probability': float(probability[0][1]),
            'stay_probability': float(probability[0][0])
        }

def main():
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--model', default='models/churn_model.pkl', 
                       help='Path to trained model')
    parser.add_argument('--tenure', type=int, help='Customer tenure in months')
    parser.add_argument('--monthly', type=float, help='Monthly charges')
    parser.add_argument('--total', type=float, help='Total charges')
    parser.add_argument('--contract', type=str, 
                       choices=['Month-to-month', 'One year', 'Two year'],
                       help='Contract type')
    parser.add_argument('--internet', type=str,
                       choices=['DSL', 'Fiber optic', 'No'],
                       help='Internet service type')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ChurnPredictor(args.model)
    
    # Example predictions
    if all([args.tenure, args.monthly, args.total, args.contract, args.internet]):
        # Single customer prediction
        result = predictor.predict_single(
            tenure=args.tenure,
            monthly_charges=args.monthly,
            total_charges=args.total,
            contract=args.contract,
            internet_service=args.internet
        )
        
        print("\n=== Churn Prediction Result ===")
        print(f"Will Churn: {'Yes' if result['will_churn'] else 'No'}")
        print(f"Churn Probability: {result['churn_probability']:.2%}")
        print(f"Stay Probability: {result['stay_probability']:.2%}")
        
        if result['churn_probability'] > 0.7:
            print("\n⚠️  HIGH RISK: Consider retention strategies")
        elif result['churn_probability'] > 0.4:
            print("\n⚡ MEDIUM RISK: Monitor customer engagement")
        else:
            print("\n✅ LOW RISK: Customer likely to stay")
    else:
        # Demo predictions with sample data
        print("\n=== Running Demo Predictions ===")
        
        # Example 1: High-risk customer
        print("\nExample 1: High-risk customer")
        result1 = predictor.predict_single(
            tenure=2,
            monthly_charges=85.0,
            total_charges=170.0,
            contract='Month-to-month',
            internet_service='Fiber optic'
        )
        print(f"Churn Probability: {result1['churn_probability']:.2%}")
        
        # Example 2: Low-risk customer
        print("\nExample 2: Low-risk customer")
        result2 = predictor.predict_single(
            tenure=60,
            monthly_charges=50.0,
            total_charges=3000.0,
            contract='Two year',
            internet_service='DSL'
        )
        print(f"Churn Probability: {result2['churn_probability']:.2%}")
        
        print("\n" + "="*50)
        print("To make predictions for your own data, use:")
        print("python predict.py --tenure 12 --monthly 75.5 --total 900 --contract 'One year' --internet 'Fiber optic'")

if __name__ == "__main__":
    main()
