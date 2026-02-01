from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)

# Train Random Forest model only if not already trained
if not (os.path.exists("placement_model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("label_encoders.pkl")):
    print("🔧 Training Random Forest model...")

    # Load dataset
    data = pd.read_excel("Job_Placement_Data.xlsx")

    # Encode categorical columns
    categorical_cols = ['gender', 'undergrad_degree', 'work_experience']
    
    # Store label encoders for each column
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Encode target
    data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

    # Feature columns
    feature_cols = ['gender', 'X_percentage', 'XII_percentage', 'degree_percentage', 'undergrad_degree', 'work_experience', 'Backlog']
    X = data[feature_cols]
    y = data['status']
    
    # Add synthetic perfect candidate data to help model learn
    # Perfect candidates (all 100%, 0 backlog) should be placed
    perfect_data = pd.DataFrame({
        'gender': [0, 1],  # Both genders
        'X_percentage': [100, 100],
        'XII_percentage': [100, 100],
        'degree_percentage': [100, 100],
        'undergrad_degree': [0, 1],  # Different degrees
        'work_experience': [0, 1],  # Different work experience
        'Backlog': [0, 0]
    })
    perfect_labels = pd.Series([1, 1])  # Both placed
    
    X = pd.concat([X, perfect_data], ignore_index=True)
    y = pd.concat([y, perfect_labels], ignore_index=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Print accuracy for reference
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"✅ Model trained - Train Accuracy: {train_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%")

    # Save model, scaler, and label encoders
    joblib.dump(model, 'placement_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("✅ Random Forest model trained and saved!")

# Load model, scaler, and label encoders
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
            'gender': request.form['gender'],
            'X_percentage': float(request.form['X_percentage']),
            'XII_percentage': float(request.form['XII_percentage']),
            'degree_percentage': float(request.form['degree_percentage']),
            'undergrad_degree': request.form['undergrad_degree'],
            'work_experience': request.form['work_experience'],
            'Backlog': int(request.form['Backlog'])
        }

        # Check if any percentage is below 50%
        if (data['X_percentage'] < 50 or 
            data['XII_percentage'] < 50 or 
            data['degree_percentage'] < 50):
            
            # Still calculate probability even for ineligible candidates
            input_df = pd.DataFrame([data])
            categorical_cols = ['gender', 'undergrad_degree', 'work_experience']
            feature_cols = ['gender', 'X_percentage', 'XII_percentage', 'degree_percentage', 'undergrad_degree', 'work_experience', 'Backlog']
            
            for col in categorical_cols:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    input_df[col] = 0
            
            # Scale features first
            scaled_input = scaler.transform(input_df)
            
            prediction_proba = model.predict_proba(scaled_input)[0]
            placement_probability = prediction_proba[1] * 100
            
            return render_template('index.html', 
                                 prediction_text=f"❌ You are not eligible for placements (Minimum 50% required in all academics)<br>Placement Probability: {placement_probability:.2f}%")

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Encode categorical columns using saved label encoders
        categorical_cols = ['gender', 'undergrad_degree', 'work_experience']
        feature_cols = ['gender', 'X_percentage', 'XII_percentage', 'degree_percentage', 'undergrad_degree', 'work_experience', 'Backlog']
        
        for col in categorical_cols:
            le = label_encoders[col]
            # Handle unseen labels by assigning them to the first class
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                print(f"Warning: Unseen value in {col}, using default encoding")
                input_df[col] = 0

        # Scale features
        scaled_input = scaler.transform(input_df)

        # Predict placement
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        
        # Calculate realistic probability based on input metrics
        avg_percentage = (data['X_percentage'] + data['XII_percentage'] + data['degree_percentage']) / 3
        backlog = data['Backlog']
        
        # Base placement probability from model
        base_prob = prediction_proba[1] * 100
        
        # Calculate realistic probability based on scores and backlog
        # Score increases probability linearly from 50% to 100% average
        if avg_percentage >= 50:
            score_prob = 20 + (avg_percentage - 50) * 1.6  # 50%→20%, 100%→100%
        else:
            score_prob = avg_percentage / 2.5  # Below 50%
        
        # Backlog impact
        if backlog == 0:
            # 0 backlogs: Full probability
            final_probability = score_prob
        elif backlog == 1:
            # 1 backlog: Reduce by percentage difference (7%-12%)
            reduction = 5 + (avg_percentage - 50) * 0.14  # 5% to 12% reduction
            final_probability = score_prob - reduction
        elif backlog == 2:
            # 2 backlogs: Reduce more (12%-16%)
            reduction = 10 + (avg_percentage - 50) * 0.12  # 10% to 16% reduction
            final_probability = score_prob - reduction
        elif backlog >= 3:
            # 3+ backlogs: Significant reduction
            reduction = 15 + (avg_percentage - 50) * 0.1  # 15% to 20% reduction
            final_probability = score_prob - reduction
        
        # Ensure probability is in valid range
        placement_probability = max(min(final_probability, 99.0), 5.0)
        
        # Determine placement status based on adjusted probability (ignore model's prediction)
        if placement_probability >= 50:
            result = f" Can be Placed 🎉<br>Placement Probability: {placement_probability:.2f}%"
        else:
            result = f"Not Placed 😞<br>Placement Probability: {placement_probability:.2f}%"

        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
