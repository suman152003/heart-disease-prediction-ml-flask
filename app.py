
import os
import re
import numpy as np
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
import json
import math
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------- CONFIG ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "records.db")
FLASK_SECRET = os.environ.get("FLASK_SECRET", "change_me")

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + DB_PATH
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ---------------- EMAIL CONFIGURATION ----------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'your-email@gmail.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'your-app-password')
mail = Mail(app)

# ---------------- DATABASE ----------------
db = SQLAlchemy(app)

# ---------------- MODELS ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_email = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    cp = db.Column(db.Integer, nullable=False)
    trestbps = db.Column(db.Integer, nullable=False)
    chol = db.Column(db.Integer, nullable=False)
    fbs = db.Column(db.Integer, nullable=False)
    restecg = db.Column(db.Integer, nullable=False)
    thalach = db.Column(db.Integer, nullable=False)
    exang = db.Column(db.Integer, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    slope = db.Column(db.Integer, nullable=False)
    ca = db.Column(db.Integer, nullable=False)
    thal = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    health_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- CREATE TABLES ---------------- 
with app.app_context():
    db.create_all()
    
    # Add health_score column if it doesn't exist (for existing databases)
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        if 'record' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('record')]
            if 'health_score' not in columns:
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE record ADD COLUMN health_score FLOAT'))
                    conn.commit()
                print("✓ Added health_score column to existing database")
    except Exception as e:
        # If table doesn't exist or column already exists, that's fine
        if 'duplicate column' not in str(e).lower() and 'no such table' not in str(e).lower():
            print(f"Database migration note: {e}")
    
    # Create default admin if it doesn't exist
    default_admin = Admin.query.filter_by(username="admin").first()
    if not default_admin:
        admin = Admin(
    username="admin",
    password=generate_password_hash("admin@123")
)
        db.session.add(admin)
        db.session.commit()
        print("Default admin account created: username='admin', password='admin@123'")

# ---------------- LOAD ML MODELS ---------------- 
MODEL_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.pkl")

model = None
scaler = None
imputer = None

def load_models():
    global model, scaler, imputer
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            imputer = joblib.load(IMPUTER_PATH)
            print("ML models loaded successfully")
        else:
            print("Warning: ML models not found. Please run train.py first.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

# ---------------- HELPERS ----------------
def calculate_health_score(features, probability):
    """Calculate a health score from 0-100 based on medical parameters"""
    score = 100.0
    
    # Age factor (optimal: 30-50)
    age = features[0]
    if age < 30:
        score -= 5
    elif age > 60:
        score -= (age - 60) * 0.5
    
    # Blood pressure (optimal: <120)
    bp = features[3]
    if bp > 140:
        score -= 15
    elif bp > 120:
        score -= 8
    
    # Cholesterol (optimal: <200)
    chol = features[4]
    if chol > 240:
        score -= 20
    elif chol > 200:
        score -= 10
    
    # Heart rate (optimal: 150-180)
    hr = features[7]
    if hr < 100:
        score -= 10
    elif hr > 200:
        score -= 5
    
    # Exercise angina
    if features[8] == 1:
        score -= 15
    
    # ST Depression
    if features[9] > 2.0:
        score -= 20
    elif features[9] > 1.0:
        score -= 10
    
    # Probability penalty
    score -= probability * 30
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))
 
def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search(r"[A-Za-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

# ---------------- ROUTES ----------------

# Home/Dashboard
@app.route("/")
def index():
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    elif session.get("username"):
        return render_template("index.html", username=session.get("username"))
    return redirect(url_for("login_page"))

# Patient Registration
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("Enter both username and password", "danger")
            return redirect(url_for("register"))

        if not is_valid_password(password):
            flash("Password must be at least 8 chars with letters, numbers, and symbols!", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for("register"))

        user = User(
    username=username,
    password=generate_password_hash(password)
)
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully! Login now.", "success")
        return redirect(url_for("login_page"))

    return render_template("register.html")

# Patient Login
@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["username"] = user.username
            flash(f"Logged in as {user.username}", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login_page"))

    return render_template("login.html")


# Admin Login
@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password, password):
            session["is_admin"] = True
            session["admin_username"] = admin.username
            flash(f"Admin logged in as {admin.username}", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid admin credentials", "danger")
            return redirect(url_for("admin_login"))
            
    return render_template("admin_login.html")

# Prediction Route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("username"):
        flash("Please login first", "warning")
        return redirect(url_for("login_page"))
    
    if request.method == "POST":
        if not model or not scaler or not imputer:
            flash("ML model not loaded. Please contact administrator.", "danger")
            return redirect(url_for("index"))
        
        try:
            # Get form data
            features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            input_data = []
            for feat in features:
                value = request.form.get(feat)
                if value is None or value == '':
                    flash(f"Please fill all fields. Missing: {feat}", "danger")
                    return redirect(url_for("index"))
                try:
                    input_data.append(float(value))
                except ValueError:
                    flash(f"Invalid value for {feat}", "danger")
                    return redirect(url_for("index"))
            
            # Prepare data for prediction
            input_array = np.array(input_data).reshape(1, -1)
            
            # Impute and scale
            input_imputed = imputer.transform(input_array)
            input_scaled = scaler.transform(input_imputed)
            
            # Predict
            probability = model.predict_proba(input_scaled)[0][1]
            prediction_binary = model.predict(input_scaled)[0]
            
            # Determine risk level
            if probability >= 0.7:
                risk_level = "High Risk"
                prediction_text = "Heart Disease Detected"
            elif probability >= 0.4:
                risk_level = "Medium Risk"
                prediction_text = "Moderate Risk of Heart Disease"
            else:
                risk_level = "Low Risk"
                prediction_text = "No Heart Disease Detected"
            
            # Calculate Health Score (0-100, higher is better)
            health_score = calculate_health_score(input_data, probability)
            
            # Save to database
            record = Record(
                patient_email=session.get("username"),
                age=int(input_data[0]),
                sex=int(input_data[1]),
                cp=int(input_data[2]),
                trestbps=int(input_data[3]),
                chol=int(input_data[4]),
                fbs=int(input_data[5]),
                restecg=int(input_data[6]),
                thalach=int(input_data[7]),
                exang=int(input_data[8]),
                oldpeak=float(input_data[9]),
                slope=int(input_data[10]),
                ca=int(input_data[11]),
                thal=int(input_data[12]),
                prediction=risk_level,
                probability=float(probability),
                health_score=float(health_score)
            )
            db.session.add(record)
            db.session.commit()
            
            # Get health recommendations
            temp_record = Record(
                patient_email=session.get("username"),
                age=int(input_data[0]),
                sex=int(input_data[1]),
                cp=int(input_data[2]),
                trestbps=int(input_data[3]),
                chol=int(input_data[4]),
                fbs=int(input_data[5]),
                restecg=int(input_data[6]),
                thalach=int(input_data[7]),
                exang=int(input_data[8]),
                oldpeak=float(input_data[9]),
                slope=int(input_data[10]),
                ca=int(input_data[11]),
                thal=int(input_data[12]),
                prediction=risk_level,
                probability=float(probability)
            )
            recommendations = get_health_recommendations(temp_record)
            
            return render_template("result.html", 
                                 prediction=prediction_text,
                                 risk_level=risk_level,
                                 probability=probability * 100,
                                 record_id=record.id,
                                 recommendations=recommendations,
                                 health_score=health_score)
        
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")
            return redirect(url_for("index"))
    
    return redirect(url_for("index"))

# Admin Dashboard
@app.route("/admin-dashboard")
def admin_dashboard():
    if not session.get("is_admin"):
        flash("Admin access required", "danger")
        return redirect(url_for("admin_login"))
    
    # Get statistics
    total_records = Record.query.count()
    high_risk = Record.query.filter(Record.probability >= 0.7).count()
    med_risk = Record.query.filter(Record.probability >= 0.4, Record.probability < 0.7).count()
    low_risk = Record.query.filter(Record.probability < 0.4).count()
    
    stats = {
        'total': total_records,
        'high': high_risk,
        'med': med_risk,
        'low': low_risk
    }
    
    # Get recent records
    records = Record.query.order_by(Record.timestamp.desc()).limit(50).all()
    
    return render_template("admin_dashboard.html", stats=stats, records=records)

# Generate PDF Report
@app.route("/report/<int:rec_id>")
def generate_pdf(rec_id):
    # Check if user is admin or the record owner
    record = Record.query.get_or_404(rec_id)
    
    if not session.get("is_admin") and session.get("username") != record.patient_email:
        flash("Access denied", "danger")
        return redirect(url_for("login_page"))
    
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#283593'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("Heart Disease Prediction Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report Information
    story.append(Paragraph("<b>Report Information</b>", heading_style))
    report_data = [
        ["Report ID:", str(record.id)],
        ["Patient ID:", record.patient_email],
        ["Date:", record.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
        ["Risk Level:", record.prediction],
        ["Probability:", f"{record.probability * 100:.2f}%"]
    ]
    
    report_table = Table(report_data, colWidths=[2*inch, 4*inch])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(report_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    story.append(Paragraph("<b>Patient Clinical Data</b>", heading_style))
    
    # Map feature names to readable labels
    feature_labels = {
        'age': 'Age',
        'sex': 'Sex',
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'restecg': 'Resting ECG Results',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'Slope of Peak Exercise ST Segment',
        'ca': 'Number of Major Vessels',
        'thal': 'Thalassemia'
    }
    
    # Map values to readable text
    def get_readable_value(field, value):
        if field == 'sex':
            return 'Male' if value == 1 else 'Female'
        elif field == 'cp':
            cp_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
            return cp_map.get(int(value), str(value))
        elif field == 'fbs':
            return 'Yes' if value == 1 else 'No'
        elif field == 'restecg':
            ecg_map = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
            return ecg_map.get(int(value), str(value))
        elif field == 'exang':
            return 'Yes' if value == 1 else 'No'
        elif field == 'slope':
            slope_map = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
            return slope_map.get(int(value), str(value))
        elif field == 'thal':
            thal_map = {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect'}
            return thal_map.get(int(value), str(value))
        else:
            return str(value)
    
    patient_data = [
        [feature_labels.get('age', 'Age'), get_readable_value('age', record.age)],
        [feature_labels.get('sex', 'Sex'), get_readable_value('sex', record.sex)],
        [feature_labels.get('cp', 'Chest Pain Type'), get_readable_value('cp', record.cp)],
        [feature_labels.get('trestbps', 'Resting BP'), f"{record.trestbps} mm Hg"],
        [feature_labels.get('chol', 'Cholesterol'), f"{record.chol} mg/dl"],
        [feature_labels.get('fbs', 'Fasting Blood Sugar'), get_readable_value('fbs', record.fbs)],
        [feature_labels.get('restecg', 'Resting ECG'), get_readable_value('restecg', record.restecg)],
        [feature_labels.get('thalach', 'Max Heart Rate'), f"{record.thalach} bpm"],
        [feature_labels.get('exang', 'Exercise Angina'), get_readable_value('exang', record.exang)],
        [feature_labels.get('oldpeak', 'ST Depression'), str(record.oldpeak)],
        [feature_labels.get('slope', 'Slope'), get_readable_value('slope', record.slope)],
        [feature_labels.get('ca', 'Major Vessels'), str(record.ca)],
        [feature_labels.get('thal', 'Thalassemia'), get_readable_value('thal', record.thal)]
    ]
    
    patient_table = Table(patient_data, colWidths=[3.5*inch, 2.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#fafafa')])
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Prediction Results
    story.append(Paragraph("<b>Prediction Results</b>", heading_style))
    
    # Determine color based on risk level
    if record.prediction == "High Risk":
        risk_color = colors.HexColor('#d32f2f')
        recommendation = "Please consult with a healthcare professional immediately for further evaluation and treatment."
    elif record.prediction == "Medium Risk":
        risk_color = colors.HexColor('#f57c00')
        recommendation = "Consider consulting with a healthcare professional for preventive care and regular monitoring."
    else:
        risk_color = colors.HexColor('#388e3c')
        recommendation = "Continue maintaining a healthy lifestyle and regular check-ups."
    
    result_data = [
        ["Risk Assessment:", record.prediction],
        ["Probability Score:", f"{record.probability * 100:.2f}%"],
        ["Recommendation:", recommendation]
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff3e0')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
        ('TEXTCOLOR', (1, 1), (1, 1), risk_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceBefore=20
    )
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "<i>This prediction is based on machine learning algorithms and should not replace professional medical advice. "
        "Always consult with qualified healthcare professionals for accurate diagnosis and treatment.</i>",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Create response
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'inline; filename=heart_prediction_report_{rec_id}.pdf'
    
    return response

# Patient Prediction History
@app.route("/my-history")
def patient_history():
    if not session.get("username"):
        flash("Please login first", "warning")
        return redirect(url_for("login_page"))
    
    username = session.get("username")
    records = Record.query.filter_by(patient_email=username).order_by(Record.timestamp.desc()).all()
    
    # Calculate statistics
    total_predictions = len(records)
    high_risk_count = sum(1 for r in records if r.probability >= 0.7)
    med_risk_count = sum(1 for r in records if r.probability >= 0.4 and r.probability < 0.7)
    low_risk_count = sum(1 for r in records if r.probability < 0.4)
    
    # Trend data for charts
    trend_data = []
    for r in reversed(records[-10:]):  # Last 10 predictions
        trend_data.append({
            'date': r.timestamp.strftime('%Y-%m-%d'),
            'probability': float(r.probability * 100),
            'risk': r.prediction
        })
    
    return render_template("patient_history.html", 
                         records=records,
                         total=total_predictions,
                         high=high_risk_count,
                         med=med_risk_count,
                         low=low_risk_count,
                         trend_data=json.dumps(trend_data))

# Feature Importance Analysis
@app.route("/feature-importance/<int:rec_id>")
def feature_importance(rec_id):
    if not session.get("username") and not session.get("is_admin"):
        flash("Access denied", "danger")
        return redirect(url_for("login_page"))
    
    record = Record.query.get_or_404(rec_id)
    
    # Check access
    if not session.get("is_admin") and session.get("username") != record.patient_email:
        flash("Access denied", "danger")
        return redirect(url_for("login_page"))
    
    if not model:
        flash("Model not loaded", "danger")
        return redirect(url_for("index"))
    
    # Get feature values
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    feature_values = [
        record.age, record.sex, record.cp, record.trestbps, record.chol,
        record.fbs, record.restecg, record.thalach, record.exang,
        record.oldpeak, record.slope, record.ca, record.thal
    ]
    
    # Calculate feature importance using SHAP-like approach (simplified)
    # For ensemble models, we'll use permutation importance approximation
    input_array = np.array(feature_values).reshape(1, -1)
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)
    
    base_prob = model.predict_proba(input_scaled)[0][1]
    
    importance_scores = {}
    for i, feat in enumerate(features):
        # Permute feature and see impact
        test_input = input_scaled.copy()
        test_input[0, i] = 0  # Set to mean (after scaling)
        perm_prob = model.predict_proba(test_input)[0][1]
        importance_scores[feat] = abs(base_prob - perm_prob)
    
    # Normalize and sort
    total_importance = sum(importance_scores.values())
    if total_importance > 0:
        importance_scores = {k: (v / total_importance) * 100 for k, v in importance_scores.items()}
    
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    return render_template("feature_importance.html",
                         record=record,
                         features=sorted_features,
                         feature_values=dict(zip(features, feature_values)))

# Health Recommendations
def get_health_recommendations(record):
    recommendations = []
    
    if record.probability >= 0.7:
        recommendations.append({
            'priority': 'high',
            'title': 'Immediate Medical Consultation',
            'description': 'Schedule an appointment with a cardiologist immediately for comprehensive evaluation.'
        })
    
    if record.chol > 240:
        recommendations.append({
            'priority': 'high',
            'title': 'Cholesterol Management',
            'description': 'Your cholesterol level is elevated. Consider dietary changes, exercise, and medication if prescribed.'
        })
    
    if record.trestbps > 140:
        recommendations.append({
            'priority': 'high',
            'title': 'Blood Pressure Control',
            'description': 'Your blood pressure is high. Monitor regularly, reduce sodium intake, and follow medical advice.'
        })
    
    if record.thalach < 120:
        recommendations.append({
            'priority': 'medium',
            'title': 'Improve Cardiovascular Fitness',
            'description': 'Low maximum heart rate suggests reduced fitness. Start with moderate exercise and gradually increase intensity.'
        })
    
    if record.exang == 1:
        recommendations.append({
            'priority': 'high',
            'title': 'Exercise-Induced Symptoms',
            'description': 'You experience chest pain during exercise. Avoid strenuous activities and consult a doctor before resuming.'
        })
    
    if record.oldpeak > 2.0:
        recommendations.append({
            'priority': 'high',
            'title': 'ST Depression Concern',
            'description': 'Elevated ST depression indicates potential heart stress. Medical evaluation is recommended.'
        })
    
    if record.age > 60:
        recommendations.append({
            'priority': 'medium',
            'title': 'Age-Related Risk',
            'description': 'Regular cardiovascular check-ups are important. Maintain a heart-healthy lifestyle.'
        })
    
    if record.fbs == 1:
        recommendations.append({
            'priority': 'medium',
            'title': 'Blood Sugar Management',
            'description': 'Elevated fasting blood sugar requires monitoring. Follow a diabetic-friendly diet if diagnosed.'
        })
    
    # General recommendations
    if record.probability < 0.4:
        recommendations.append({
            'priority': 'low',
            'title': 'Maintain Healthy Lifestyle',
            'description': 'Continue regular exercise, balanced diet, and annual health check-ups.'
        })
    else:
        recommendations.append({
            'priority': 'medium',
            'title': 'Lifestyle Modifications',
            'description': 'Adopt a heart-healthy diet (Mediterranean or DASH diet), regular exercise, stress management, and adequate sleep.'
        })
    
    return recommendations

# Enhanced Result Page with Recommendations
@app.route("/result/<int:rec_id>")
def view_result(rec_id):
    if not session.get("username") and not session.get("is_admin"):
        flash("Access denied", "danger")
        return redirect(url_for("login_page"))
    
    record = Record.query.get_or_404(rec_id)
    
    if not session.get("is_admin") and session.get("username") != record.patient_email:
        flash("Access denied", "danger")
        return redirect(url_for("login_page"))
    
    recommendations = get_health_recommendations(record)
    
    prediction_text = record.prediction
    
    return render_template("result.html",
                         prediction=prediction_text,
                         risk_level=record.prediction,
                         probability=record.probability * 100,
                         record_id=record.id,
                         recommendations=recommendations)

# Export Data (CSV/Excel)
@app.route("/export-data")
def export_data():
    if not session.get("is_admin"):
        flash("Admin access required", "danger")
        return redirect(url_for("admin_login"))
    
    format_type = request.args.get('format', 'csv')
    
    records = Record.query.order_by(Record.timestamp.desc()).all()
    
    data = []
    for r in records:
        data.append({
            'ID': r.id,
            'Patient': r.patient_email,
            'Age': r.age,
            'Sex': 'Male' if r.sex == 1 else 'Female',
            'Chest Pain': r.cp,
            'Blood Pressure': r.trestbps,
            'Cholesterol': r.chol,
            'Fasting Blood Sugar': 'Yes' if r.fbs == 1 else 'No',
            'Resting ECG': r.restecg,
            'Max Heart Rate': r.thalach,
            'Exercise Angina': 'Yes' if r.exang == 1 else 'No',
            'ST Depression': r.oldpeak,
            'Slope': r.slope,
            'Major Vessels': r.ca,
            'Thalassemia': r.thal,
            'Risk Level': r.prediction,
            'Probability': f"{r.probability * 100:.2f}%",
            'Date': r.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(data)
    
    if format_type == 'excel':
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            date_str = datetime.now().strftime('%Y%m%d')
            response.headers['Content-Disposition'] = f'attachment; filename=heart_predictions_{date_str}.xlsx'
            return response
        except ImportError:
            flash('Excel export requires the openpyxl package. Please contact the administrator.', 'danger')
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            flash(f'Excel export failed: {str(e)}', 'danger')
            return redirect(url_for('admin_dashboard'))
    else:
        try:
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            date_str = datetime.now().strftime('%Y%m%d')
            response.headers['Content-Disposition'] = f'attachment; filename=heart_predictions_{date_str}.csv'
            return response
        except Exception as e:
            flash(f'CSV export failed: {str(e)}', 'danger')
            return redirect(url_for('admin_dashboard'))

# Analytics API for Charts
@app.route("/api/analytics")
def analytics_api():
    if not session.get("is_admin"):
        return jsonify({"error": "Unauthorized"}), 403
    
    # Risk distribution
    total = Record.query.count()
    high = Record.query.filter(Record.probability >= 0.7).count()
    med = Record.query.filter(Record.probability >= 0.4, Record.probability < 0.7).count()
    low = Record.query.filter(Record.probability < 0.4).count()
    
    # Daily predictions (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    daily_data = {}
    records = Record.query.filter(Record.timestamp >= thirty_days_ago).all()
    for r in records:
        date_str = r.timestamp.strftime('%Y-%m-%d')
        daily_data[date_str] = daily_data.get(date_str, 0) + 1
    
    # Age distribution
    age_groups = {'<30': 0, '30-40': 0, '40-50': 0, '50-60': 0, '60-70': 0, '>70': 0}
    all_records = Record.query.all()
    for r in all_records:
        if r.age < 30:
            age_groups['<30'] += 1
        elif r.age < 40:
            age_groups['30-40'] += 1
        elif r.age < 50:
            age_groups['40-50'] += 1
        elif r.age < 60:
            age_groups['50-60'] += 1
        elif r.age < 70:
            age_groups['60-70'] += 1
        else:
            age_groups['>70'] += 1
    
    return jsonify({
        'risk_distribution': {
            'high': int(high), 
            'medium': int(med), 
            'low': int(low), 
            'total': int(total)
        },
        'daily_predictions': daily_data,
        'age_distribution': age_groups
    })

# Prediction Comparison Tool
@app.route("/compare-predictions")
def compare_predictions():
    if not session.get("username"):
        flash("Please login first", "warning")
        return redirect(url_for("login_page"))
    
    username = session.get("username")
    records = Record.query.filter_by(patient_email=username).order_by(Record.timestamp.desc()).limit(5).all()
    
    if len(records) < 2:
        flash("You need at least 2 predictions to compare", "info")
        return redirect(url_for("patient_history"))
    
    comparison_data = []
    for r in records:
        comparison_data.append({
            'id': r.id,
            'date': r.timestamp.strftime('%Y-%m-%d'),
            'probability': float(r.probability * 100),
            'health_score': float(r.health_score) if r.health_score else 0,
            'risk_level': r.prediction
        })
    
    return render_template("compare_predictions.html", comparisons=comparison_data)

# Advanced Analytics Dashboard
@app.route("/advanced-analytics")
def advanced_analytics():
    if not session.get("is_admin"):
        flash("Admin access required", "danger")
        return redirect(url_for("admin_login"))
    
    all_records = Record.query.all()
    if not all_records:
        flash("No data available", "info")
        return redirect(url_for("admin_dashboard"))
    
    high_risk_scores = [r.health_score for r in all_records if r.probability >= 0.7 and r.health_score]
    med_risk_scores = [r.health_score for r in all_records if r.probability >= 0.4 and r.probability < 0.7 and r.health_score]
    low_risk_scores = [r.health_score for r in all_records if r.probability < 0.4 and r.health_score]
    
    analytics = {
        'total_predictions': len(all_records),
        'avg_health_score': sum(r.health_score for r in all_records if r.health_score) / len([r for r in all_records if r.health_score]) if any(r.health_score for r in all_records) else 0,
        'high_risk_avg_score': sum(high_risk_scores) / len(high_risk_scores) if high_risk_scores else 0,
        'med_risk_avg_score': sum(med_risk_scores) / len(med_risk_scores) if med_risk_scores else 0,
        'low_risk_avg_score': sum(low_risk_scores) / len(low_risk_scores) if low_risk_scores else 0,
        'avg_age': sum(r.age for r in all_records) / len(all_records),
        'avg_chol': sum(r.chol for r in all_records) / len(all_records),
        'avg_bp': sum(r.trestbps for r in all_records) / len(all_records)
    }
    
    return render_template("advanced_analytics.html", analytics=analytics)

# API Documentation

# Logout
@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("is_admin", None)
    session.pop("admin_username", None)
    flash("Logged out successfully", "info")
    return redirect(url_for("login_page"))

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
