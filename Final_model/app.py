# app.py
# -------------------------------------------------------------
# Streamlit App for Medical Symptom → Disease Prediction
# Using already-trained ExtraTrees model and Ollama symptom extraction
# -------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import ast
import os
from ollama import Client

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
MODEL_PATH = "Final_model/Models/model_extratrees.pkl"
LE_PATH = "Final_model/Models/label_encoder.pkl"
FEATURES_PATH = "Final_model/Models/features.pkl"
SYMPTOM_DESC_PATH = "Final_model/data/symptom_Description.csv"
SYMPTOM_PREC_PATH = "Final_model/data/symptom_precaution.csv"

OLLAMA_API_KEY = "821e286afabb4ab49533fa6c3aeec4c3.Vo7YFxfEOdhfQ16wFpacGEGm"    # <---- replace with your real key
OLLAMA_MODEL = "gpt-oss:120b"

# -------------------------------------------------------------
# LOAD MODEL & ASSETS
# -------------------------------------------------------------

@st.cache_resource
def load_assets():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(LE_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        feature_list = pickle.load(f)

    # Load description / precautions CSV files
    desc = pd.read_csv(SYMPTOM_DESC_PATH)
    prec = pd.read_csv(SYMPTOM_PREC_PATH)

    desc["Disease"] = desc["Disease"].astype(str).str.strip()
    prec["Disease"] = prec["Disease"].astype(str).str.strip()

    return model, label_encoder, feature_list, desc, prec


model, le, feature_list, desc_df, prec_df = load_assets()


# -------------------------------------------------------------
# OLLAMA CLIENT INITIALIZATION
# -------------------------------------------------------------
client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY}
)

# -------------------------------------------------------------
# OLLAMA SYMPTOM EXTRACTION FUNCTION
# -------------------------------------------------------------
def extract_symptoms_from_text(user_text):
    """
    Sends user text → Ollama → extracts symptoms only from allowed list.
    Returns a clean Python list of symptoms.
    """

    allowed_json = json.dumps(feature_list)

    messages = [
        {
            "role": "user",
            "content": f"""
You are a strict medical symptom extractor.
Extract only symptoms mentioned in the user's text and return a Python list.
Do NOT include anything not in this list:

Allowed symptoms: {allowed_json}

User text:
\"\"\"{user_text}\"\"\"

Return ONLY a Python list. Nothing else.
"""
        }
    ]

    fragments = []
    try:
        for part in client.chat(OLLAMA_MODEL, messages=messages, stream=True):
            fragments.append(part["message"]["content"])
        full_output = "".join(fragments)
    except Exception as e:
        st.error(f"Ollama request failed: {e}")
        return []

    # Parse output to Python list safely
    try:
        final_list = ast.literal_eval(full_output)
    except Exception:
        final_list = []

    # Final clean filter → ensure only allowed symptoms
    final_list = [s for s in final_list if s in feature_list]

    return final_list


# -------------------------------------------------------------
# SYMPTOMS → VECTOR
# -------------------------------------------------------------
def symptoms_to_vector(symptom_list):
    vector = np.zeros((1, len(feature_list)), dtype=np.int8)
    index_map = {s: i for i, s in enumerate(feature_list)}

    for s in symptom_list:
        if s in index_map:
            vector[0, index_map[s]] = 1

    return vector


# -------------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------------
def predict_diseases(symptom_list, top_k=3):

    X = symptoms_to_vector(symptom_list)
    proba = model.predict_proba(X)[0]

    # Top K predicted indexes
    idx = np.argsort(proba)[-top_k:][::-1]

    results = []
    for i in idx:
        disease = le.classes_[i]
        prob = float(proba[i])

        # Fetch description
        description = None
        if disease in desc_df["Disease"].values:
            description = desc_df.loc[desc_df["Disease"] == disease].iloc[0, 1]

        # Fetch precautions
        precautions = []
        if disease in prec_df["Disease"].values:
            row = prec_df.loc[prec_df["Disease"] == disease].iloc[0, 1:]
            precautions = [p for p in row if pd.notna(p) and str(p).strip()]

        results.append({
            "disease": disease,
            "probability": prob,
            "description": description,
            "precautions": precautions
        })

    return results


# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="AI Medical Diagnosis", layout="wide")
st.title("🩺 AI Medical Diagnosis from Symptoms")

st.markdown("""
Enter a natural-language description of how you feel.
The system will extract symptoms → generate a medical-like prediction  
(Top 3 possible diseases), along with descriptions and precautions.
""")

user_input = st.text_area("Describe how you feel:", height=150)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter your symptoms description.")
        st.stop()

    st.info("Extracting symptoms using the AI model...")
    extracted = extract_symptoms_from_text(user_input)

    st.subheader("🧩 Extracted Symptoms")
    st.write(extracted)

    if not extracted:
        st.error("No valid symptoms detected. Try expanding your description.")
        st.stop()

    st.info("Predicting possible diseases...")
    results = predict_diseases(extracted, top_k=3)

    st.subheader("📌 Top 3 Possible Diagnoses")
    for r in results:
        st.markdown(f"### 🦠 **{r['disease']}**  — Probability: **{r['probability']:.2f}**")

        if r["description"]:
            st.markdown(f"**Description:** {r['description']}")

        if r["precautions"]:
            st.markdown("**Recommended Precautions:**")
            for p in r["precautions"]:
                st.markdown(f"- {p}")

        st.markdown("---")
