from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
from model import predict_heart_disease

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Application Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/guest')
def guest_access():
    session['guest_mode'] = True
    return redirect(url_for('home'))

@app.route('/guest-clear')
def guest_clear():
    session.pop('guest_mode', None)
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists', 'error')
            return redirect(url_for('register'))

        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password, method='pbkdf2:sha256')
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Please check your login details and try again.', 'error')
            
    return render_template('login.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not email or not new_password or not confirm_password:
            flash('All fields are required.', 'error')
            return redirect(url_for('reset_password'))
        
        if new_password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('reset_password'))
        
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('No account found with that email address.', 'error')
            return redirect(url_for('reset_password'))
        
        user.password_hash = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.session.commit()
        
        flash('Password reset successfully! Please log in with your new password.', 'success')
        return redirect(url_for('home'))
    
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.date.asc()).all()
    latest_prediction = predictions[-1] if predictions else None
    
    dates = [p.date.strftime('%Y-%m-%d %H:%M') for p in predictions]
    probs = [p.probability for p in predictions]
    
    return render_template('dashboard.html', 
                           latest=latest_prediction, 
                           dates=dates, 
                           probs=probs,
                           has_data=len(predictions) > 0)

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.date.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/prediction')
def prediction_form():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age_raw = request.form['Age']
        sex_raw = request.form['Sex']
        cp_raw = request.form['ChestPainType']
        bp_raw = request.form['RestingBP']
        chol_raw = request.form['Cholesterol']
        fbs_raw = request.form['FastingBS']
        ecg_raw = request.form['RestingECG']
        maxhr_raw = request.form['MaxHR']
        angina_raw = request.form['ExerciseAngina']
        oldpeak_raw = request.form['Oldpeak']
        slope_raw = request.form['ST_Slope']

        sex_map = {"1": "M", "0": "F"}
        chest_pain_map = {"No pain": "NAP", "Mild discomfort": "ATA", "Chest pain during activity": "TA", "Severe pain even at rest": "ASY"}
        resting_bp_map = {"No": 110.0, "Sometimes": 130.0, "Yes": 150.0}
        cholesterol_map = {"No": 180.0, "Not sure": 210.0, "Yes": 250.0}
        fasting_bs_map = {"No": 0, "Yes": 1}
        resting_ecg_map = {"No": "Normal", "Not sure": "ST", "Yes": "LVH"}
        max_hr_map = {"No": 170.0, "Sometimes": 140.0, "Often": 110.0}
        exercise_angina_map = {"No": "N", "Yes": "Y"}
        oldpeak_map = {"No": 0.2, "Sometimes": 1.0, "Often": 2.5}
        st_slope_map = {"No": "Up", "Not sure": "Flat", "Yes": "Down"}

        age = float(age_raw)
        sex = sex_map[sex_raw]
        chest_pain_type = chest_pain_map[cp_raw]
        resting_bp = resting_bp_map[bp_raw]
        cholesterol = cholesterol_map[chol_raw]
        fasting_bs = fasting_bs_map[fbs_raw]
        resting_ecg = resting_ecg_map[ecg_raw]
        max_hr = max_hr_map[maxhr_raw]
        exercise_angina = exercise_angina_map[angina_raw]
        oldpeak = oldpeak_map[oldpeak_raw]
        st_slope = st_slope_map[slope_raw]

        user_inputs = [
            age, sex, chest_pain_type, fasting_bs, resting_bp, cholesterol,
            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
        ]
        
        result = predict_heart_disease(*user_inputs)
        
        # --- EXPLAINABLE AI (Feature Importance Calculation) ---
        baseline = [40.0, 'F', 'NAP', 0, 110.0, 180.0, 'Normal', 170.0, 'N', 0.2, 'Up']
        feature_names = ["Age", "Gender", "Chest Pain", "Diabetes Proxy", "Blood Pressure", "Cholesterol", "ECG", "Max HR Target", "Physical Inactivity", "Fatigue Output", "Heart Rhythm Slope"]
        
        importances = []
        original_prob = result['probability_yes']
        
        for i in range(len(user_inputs)):
            if user_inputs[i] != baseline[i]:
                temp_inputs = list(user_inputs)
                temp_inputs[i] = baseline[i]
                new_prob = predict_heart_disease(*temp_inputs)['probability_yes']
                
                drop = original_prob - new_prob
                if drop > 0:
                    importances.append({"feature": feature_names[i], "drop": drop})
                    
        total_drop = sum([x['drop'] for x in importances])
        top_factors = []
        if total_drop > 0:
            for item in importances:
                item['percent'] = round((item['drop'] / total_drop) * 100)
            
            importances = sorted(importances, key=lambda x: x['percent'], reverse=True)
            top_factors = [x for x in importances if x['percent'] > 0][:4]
            
        result['top_factors'] = top_factors
        
        # --- SAVE PREDICTION IF AUTHENTICATED ---
        if current_user.is_authenticated:
            new_pred = Prediction(
                user_id=current_user.id,
                probability=result['probability_yes'],
                risk_level=result['risk_level']
            )
            db.session.add(new_pred)
            db.session.commit()

        return render_template('result.html', result=result)
        
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
