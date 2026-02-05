import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import ast
from ollama import Client

# -------------------------------------------------------------
# CONFIG & PATHS
# -------------------------------------------------------------
# Ensure these match your actual file structure
MODEL_PATH = "medical_rf_model.pkl"
LE_PATH = "label_encoder.pkl"
FEATURES_PATH = "features.pkl"
SCALER_PATH = "scaler.pkl"
MAX_WEIGHT_PATH = "max_weight.pkl"

# CSV Data for descriptions and weights
SYMPTOM_DESC_PATH = "data/symptom_Description.csv"
SYMPTOM_PREC_PATH = "data/symptom_precaution.csv"
SYMPTOM_SEVERITY_PATH = "data/Symptom-severity.csv" # REQUIRED for accurate weighting

# OLLAMA SETTINGS
OLLAMA_API_KEY = "821e286afabb4ab49533fa6c3aeec4c3.Vo7YFxfEOdhfQ16wFpacGEGm" 
OLLAMA_MODEL = "gpt-oss:120b" # Or "llama3", "mistral", etc.

# -------------------------------------------------------------
# 1. LOAD ASSETS
# -------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        features = joblib.load(FEATURES_PATH)
        scaler = joblib.load(SCALER_PATH)
        max_weight = joblib.load(MAX_WEIGHT_PATH)
        
        # Load CSVs
        desc = pd.read_csv(SYMPTOM_DESC_PATH)
        prec = pd.read_csv(SYMPTOM_PREC_PATH)
        
        # Load Severity (Optional but recommended)
        try:
            severity_df = pd.read_csv(SYMPTOM_SEVERITY_PATH)
            # Create a dictionary: {'itching': 1, 'vomiting': 5}
            severity_map = dict(zip(severity_df.iloc[:,0].str.replace('_',' '), severity_df.iloc[:,1]))
        except FileNotFoundError:
            st.warning("⚠️ 'Symptom-severity.csv' not found. Using default weights.")
            severity_map = {}

        # Cleanup strings
        desc["Disease"] = desc["Disease"].astype(str).str.strip()
        prec["Disease"] = prec["Disease"].astype(str).str.strip()
        
        return model, le, features, scaler, max_weight, desc, prec, severity_map
        
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None, None, None, None, None

# Load everything
model, le, feature_list, scaler, max_weight, desc_df, prec_df, severity_map = load_assets()

# -------------------------------------------------------------
# 2. OLLAMA CLIENT
# -------------------------------------------------------------
# Initialize Client (Adjust URL if running locally vs remotely)
client = Client(host="https://ollama.com", headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY})

def extract_symptoms_from_text(user_text):
    """
    Uses Ollama to find symptoms in the text that match our known feature list.
    """
    # Filter feature list to only include symptoms (exclude Age, BMI, etc.)
    known_symptoms = [f for f in feature_list if f not in ['Age', 'Systolic_BP', 'Glucose', 'BMI']]
    
    messages = [
        {
            "role": "system", 
            "content": "You are a medical assistant. Extract symptoms from the text that match the provided list exactly."
        },
        {
            "role": "user",
            "content": f"""
            Extract symptoms from this text: "{user_text}"
            
            Match them to this allowed list ONLY:
            {known_symptoms}
            
            Return ONLY a Python list of strings (e.g. ['headache', 'vomiting']).
            If no matches, return [].
            """
        }
    ]

    try:
        response = client.chat(model=OLLAMA_MODEL, messages=messages)
        content = response['message']['content']
        
        # specific cleanup for LLM response
        start = content.find('[')
        end = content.find(']') + 1
        if start != -1 and end != -1:
            clean_list = ast.literal_eval(content[start:end])
            return [s for s in clean_list if s in known_symptoms]
        return []
    except Exception as e:
        st.error(f"Ollama Error: {e}")
        return []

# -------------------------------------------------------------
# 3. PREDICTION LOGIC (THE BRAIN)
# -------------------------------------------------------------
def predict_disease(user_vitals, user_symptoms):
    """
    1. Scale Vitals (Age, BP, etc.) using the saved Scaler.
    2. Weight Symptoms using the Severity Map.
    3. Normalize Symptoms using max_weight.
    4. Combine into a single vector.
    5. Predict.
    """
    
    # A. Create a DataFrame with all zeros for all features
    input_data = pd.DataFrame(0, index=[0], columns=feature_list)
    
    # B. Fill Vitals & Normalize
    # We must match the order the scaler expects: ['Age', 'Systolic_BP', 'Glucose', 'BMI']
    vitals_array = np.array([[
        user_vitals['Age'], 
        user_vitals['Systolic_BP'], 
        user_vitals['Glucose'], 
        user_vitals['BMI']
    ]])
    
    vitals_scaled = scaler.transform(vitals_array)
    
    # Update the dataframe
    input_data['Age'] = vitals_scaled[0][0]
    input_data['Systolic_BP'] = vitals_scaled[0][1]
    input_data['Glucose'] = vitals_scaled[0][2]
    input_data['BMI'] = vitals_scaled[0][3]
    
    # C. Fill Symptoms (Weighted & Normalized)
    for symptom in user_symptoms:
        if symptom in feature_list:
            # Get weight (default to 3 if unknown)
            weight = severity_map.get(symptom, 3) 
            # Normalize (Weight / Global Max)
            normalized_score = weight / max_weight
            input_data[symptom] = normalized_score
            
    # D. Predict
    probabilities = model.predict_proba(input_data)[0]
    
    # Get Top 3
    top_indices = np.argsort(probabilities)[-3:][::-1]
    results = []
    
    for idx in top_indices:
        disease_name = le.classes_[idx]
        prob = probabilities[idx]
        
        # Get Info
        desc = desc_df.loc[desc_df['Disease'] == disease_name, 'Description'].values
        desc = desc[0] if len(desc) > 0 else "No description available."
        
        prec = prec_df.loc[prec_df['Disease'] == disease_name].values.flatten().tolist()[1:]
        prec = [p for p in prec if pd.notna(p)]
        
        results.append({
            "Disease": disease_name,
            "Probability": prob,
            "Description": desc,
            "Precautions": prec
        })
        
    return results

# -------------------------------------------------------------
# 4. STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="AI Medical Diagnostics", layout="wide", page_icon="🩺")

# -- Header --
st.title("🩺 AI Medical Diagnostics (Enhanced)")
st.markdown("---")

# -- Sidebar for Vitals (New Requirement) --
with st.sidebar:
    st.header("1. Enter Vitals")
    st.info("These metrics are crucial for accurate prediction.")
    
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    bp = st.number_input("Systolic BP (High number)", min_value=70, max_value=250, value=120)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=500, value=90)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, format="%.1f")
    
    user_vitals = {"Age": age, "Systolic_BP": bp, "Glucose": glucose, "BMI": bmi}

# -- Main Section for Symptoms --
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("2. Describe Your Symptoms")
    user_text = st.text_area("How are you feeling?", height=150, placeholder="Example: I have a severe headache, nausea, and stiff neck...")
    
    analyze_btn = st.button("🔍 Analyze & Diagnose", type="primary")

# -- Results Section --
if analyze_btn:
    if not user_text:
        st.warning("Please describe your symptoms first.")
        st.stop()
        
    with st.spinner("Extracting symptoms using AI..."):
        # 1. Extract
        extracted_symptoms = extract_symptoms_from_text(user_text)
        
    if not extracted_symptoms:
        st.error("No recognizable symptoms found. Please try again with simpler terms.")
    else:
        # Show what was found
        st.success(f"Symptoms Identified: {', '.join(extracted_symptoms)}")
        
        with st.spinner("Analyzing Vitals & Predicting Disease..."):
            # 2. Predict
            predictions = predict_disease(user_vitals, extracted_symptoms)
            
        # 3. Display
        st.markdown("### 🏥 Diagnostic Results")
        
        for i, res in enumerate(predictions):
            # Dynamic color for probability
            color = "red" if res['Probability'] > 0.5 else "orange" if res['Probability'] > 0.2 else "blue"
            
            with st.expander(f"{i+1}. {res['Disease']} ({res['Probability']*100:.1f}%)", expanded=(i==0)):
                st.markdown(f"**Confidence:** :{color}[{res['Probability']*100:.2f}%]")
                st.markdown(f"**Description:** {res['Description']}")
                
                st.markdown("**Recommended Actions:**")
                for p in res['Precautions']:
                    st.markdown(f"- {p}")
