# Heart Disease Prediction System

A professional web application for predicting heart disease risk using machine learning, built with Flask and trained on the UCI Heart Disease dataset.

## Features

- **User Authentication**: Patient registration and login system
- **Admin Dashboard**: View statistics and manage prediction records
- **ML-Based Prediction**: Ensemble model (XGBoost, Random Forest, MLP) for accurate predictions
- **Professional UI**: Modern, responsive design with Bootstrap 5
- **Risk Assessment**: Categorizes predictions into High, Medium, and Low risk levels

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (if models don't exist)
   ```bash
   python train.py
   ```
   This will create the model files in the `models/` directory:
   - `heart_model.pkl` - Trained ensemble model
   - `scaler.pkl` - Feature scaler
   - `imputer.pkl` - Missing value imputer

3. **Initialize Admin User** (optional)
   ```bash
   python init_admin.py
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the Application**
   - Open your browser and navigate to `http://localhost:5000`
   - Register a new patient account or login
   - Fill in the medical information form to get predictions

## Project Structure

```
heart_disease_pro/
├── app.py                 # Main Flask application
├── train.py              # Model training script
├── init_admin.py         # Admin user initialization
├── heart.csv             # UCI Heart Disease dataset
├── requirements.txt      # Python dependencies
├── records.db            # SQLite database (created automatically)
├── models/               # ML model files
│   ├── heart_model.pkl
│   ├── scaler.pkl
│   └── imputer.pkl
└── templates/            # HTML templates
    ├── layout.html
    ├── index.html
    ├── login.html
    ├── register.html
    ├── admin_login.html
    ├── admin_dashboard.html
    └── result.html
```

## Usage

### For Patients

1. Register a new account or login
2. Fill in the medical information form with:
   - Age, Sex, Chest Pain Type
   - Blood Pressure, Cholesterol
   - ECG results, Heart Rate
   - Exercise-related metrics
   - And other clinical parameters
3. Submit the form to get your heart disease risk prediction
4. View detailed results with risk level and probability

### For Administrators

1. Login with admin credentials
2. View dashboard statistics:
   - Total prediction records
   - High/Medium/Low risk counts
3. Browse recent prediction records
4. Monitor system usage

## Model Information

The prediction model uses an ensemble approach combining:
- **XGBoost Classifier**: Gradient boosting
- **Random Forest**: Ensemble of decision trees
- **MLP Classifier**: Neural network

The model is trained with:
- SMOTE for handling class imbalance
- StandardScaler for feature normalization
- SimpleImputer for missing value handling

## Database Schema

- **User**: Patient accounts (username, password)
- **Admin**: Administrator accounts
- **Record**: Prediction history (patient info, prediction results, timestamp)

## Security Notes

- Passwords are stored in plain text (for demo purposes). In production, use password hashing.
- Admin credentials should be kept secure.
- The application uses session-based authentication.

## Requirements

- Python 3.8+
- Flask 3.0.0
- scikit-learn 1.3.2
- pandas 2.0.3
- numpy 1.24.3
- xgboost 2.0.3
- imbalanced-learn 0.11.0

## License

This project is for educational purposes. The UCI Heart Disease dataset is publicly available.

## Disclaimer

This application is for educational and research purposes only. The predictions should not replace professional medical advice. Always consult with qualified healthcare professionals for accurate diagnosis and treatment.

