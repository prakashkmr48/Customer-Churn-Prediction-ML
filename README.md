# Customer Churn Prediction - End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📊 Project Overview

This is a **complete end-to-end machine learning project** for predicting customer churn in the telecom industry. The project demonstrates the full ML lifecycle from data exploration to model deployment, making it ideal for showcasing to interviewers.

### Key Features
- ✅ **Complete ML Pipeline**: Data loading, EDA, preprocessing, feature engineering, training, and evaluation
- ✅ **Multiple Algorithms**: Logistic Regression, Random Forest, and Gradient Boosting with automatic model selection
- ✅ **Production-Ready Code**: Modular, well-documented, and following best practices
- ✅ **Easy to Run**: Single command execution with sample data included
- ✅ **Prediction API**: Ready-to-use inference script for making predictions

## 📦 Project Structure

```
Customer-Churn-Prediction-ML/
├── train_model.py          # Main training script
├── predict.py              # Prediction/inference script  
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore file
└── models/                 # Saved models (created after training)
    └── churn_model.pkl
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/prakashkmr48/Customer-Churn-Prediction-ML.git
cd Customer-Churn-Prediction-ML
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

This will:
- Generate sample data for demonstration
- Perform exploratory data analysis
- Train multiple models
- Select the best performing model
- Save the model to `models/churn_model.pkl`

### 4. Make Predictions
```bash
# Demo predictions with sample data
python predict.py

# Predict for specific customer
python predict.py --tenure 12 --monthly 75.5 --total 900 --contract "One year" --internet "Fiber optic"
```

## 💻 Technical Details

### Machine Learning Pipeline

1. **Data Loading & EDA**
   - Dataset overview and statistical analysis
   - Missing value detection
   - Target variable distribution

2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables (Label Encoding)
   - Feature scaling (StandardScaler)

3. **Feature Engineering**
   - Create interaction features (e.g., TenureMonthlyCharges)
   - Domain-specific feature creation

4. **Model Training**
   - Train multiple algorithms:
     - Logistic Regression
     - Random Forest Classifier
     - Gradient Boosting Classifier
   - 5-fold cross-validation for model evaluation
   - Automatic best model selection based on ROC AUC score

5. **Model Evaluation**
   - Classification report (Precision, Recall, F1-Score)
   - ROC AUC score
   - Confusion matrix

6. **Model Persistence**
   - Save trained model and scaler using joblib
   - Load model for inference

### Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib & seaborn** - Data visualization
- **joblib** - Model serialization

## 📊 Model Performance

The model achieves:
- **ROC AUC Score**: ~0.75-0.85 (varies with data)
- **Precision**: ~70-80%
- **Recall**: ~65-75%

*Note: Actual performance will vary based on the dataset used.*

## 📖 Usage Examples

### Training Custom Model

```python
from train_model import ChurnPredictor
import pandas as pd

# Initialize predictor
predictor = ChurnPredictor()

# Load your own data
predictor.df = pd.read_csv('your_data.csv')

# Run the pipeline
predictor.exploratory_analysis()
predictor.preprocess_data()
predictor.feature_engineering()

X, y = predictor.prepare_features()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

predictor.train_model(X_train, y_train)
predictor.evaluate_model(X_test, y_test)
predictor.save_model()
```

### Making Predictions

```python
from predict import ChurnPredictor

# Load trained model
predictor = ChurnPredictor('models/churn_model.pkl')

# Predict for a customer
result = predictor.predict_single(
    tenure=24,
    monthly_charges=80.0,
    total_charges=1920.0,
    contract='Month-to-month',
    internet_service='Fiber optic'
)

print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Will Churn: {result['will_churn']}")
```

## 📝 Key Learnings & Insights

This project demonstrates:

1. **End-to-End ML Workflow**: Complete pipeline from raw data to predictions
2. **Model Comparison**: Systematic comparison of multiple algorithms
3. **Production Considerations**: Model persistence, error handling, and logging
4. **Code Quality**: Modular design, documentation, and best practices
5. **Real-World Application**: Solving a common business problem

## 🔧 Customization

### Using Your Own Data

1. Prepare your dataset with these columns:
   - `tenure`: Customer tenure in months
   - `MonthlyCharges`: Monthly charges
   - `TotalCharges`: Total charges
   - `Contract`: Contract type
   - `InternetService`: Internet service type
   - `Churn`: Target variable (Yes/No or 1/0)

2. Modify `train_model.py`:
   ```python
   # Replace the sample data generation with:
   predictor.load_data('path/to/your/data.csv')
   ```

3. Run training as usual:
   ```bash
   python train_model.py
   ```

## ❓ FAQ

**Q: Can I use this project for my portfolio?**  
A: Yes! This project is MIT licensed and perfect for portfolios.

**Q: What if I don't have my own data?**  
A: The project includes sample data generation, so you can run it immediately.

**Q: How do I deploy this model?**  
A: The prediction script (`predict.py`) is ready for deployment. You can wrap it in a Flask/FastAPI REST API.

**Q: Can I add more features?**  
A: Absolutely! Modify the `feature_engineering()` method in `train_model.py`.

## 💬 Contact

**Prakash Kumar**
- GitHub: [@prakashkmr48](https://github.com/prakashkmr48)
- Project Link: [https://github.com/prakashkmr48/Customer-Churn-Prediction-ML](https://github.com/prakashkmr48/Customer-Churn-Prediction-ML)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Acknowledgments

- Inspired by real-world telecom industry challenges
- Built with best practices for ML project structure
- Designed to be interview-friendly and easy to understand

---

**👍 If you find this project helpful, please give it a star!**
