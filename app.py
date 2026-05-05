import streamlit as st
import pickle
import pdfplumber
import re
import pandas as pd
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import numpy as np

# ---------------- PAGE ----------------
st.set_page_config(page_title="Hospital AI System", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {background:#f1f5f9;color:#0f172a;}
section[data-testid="stSidebar"] {background:#0f172a;}
section[data-testid="stSidebar"] * {color:white !important;}
.card {background:white;padding:18px;border-radius:12px;
box-shadow:0 2px 10px rgba(0,0,0,0.06);margin-bottom:15px;}
.high {color:#16a34a;font-weight:bold;font-size:22px;}
.low {color:#dc2626;font-weight:bold;font-size:22px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h2>🏥 Heart Disease Prediction System</h2>
<p>AI-powered clinical decision support</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Patient Input")
option = st.sidebar.radio("Select Mode", ["Manual Entry", "Upload Report"])

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# Load scaler if exists
scaler = None
if os.path.exists("scaler.pkl"):
    scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- FUNCTIONS ----------------

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_features(text):
    text = text.lower()
    age = re.search(r'age[:\s]*(\d+)', text)
    chol = re.search(r'cholesterol[:\s]*(\d+)', text)
    bp = re.search(r'bp[:\s]*(\d+)', text)

    return [
        int(age.group(1)) if age else 50,
        int(chol.group(1)) if chol else 200,
        int(bp.group(1)) if bp else 120
    ]

def show_kpi(features):
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><h4>Age</h4><h2>{features[0]}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h4>Cholesterol</h4><h2>{features[1]}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h4>Blood Pressure</h4><h2>{features[2]}</h2></div>", unsafe_allow_html=True)

def show_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={'suffix': "%"},
        title={'text': "Risk Probability"},
        gauge={
            'axis': {'range': [0,100]},
            'steps': [
                {'range':[0,40],'color':"#22c55e"},
                {'range':[40,70],'color':"#facc15"},
                {'range':[70,100],'color':"#ef4444"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def show_graph(features, prob):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Age", "Chol", "BP"],
        y=features,
        name="Patient Data"
    ))

    fig.add_trace(go.Scatter(
        x=["Age", "Chol", "BP"],
        y=[prob/2, prob/1.5, prob],
        mode='lines+markers',
        name="Risk Trend"
    ))

    st.plotly_chart(fig, use_container_width=True)

def generate_pdf(features, result, prob):
    file = "report.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Heart Disease Report", styles["Title"]))
    content.append(Spacer(1,10))
    content.append(Paragraph(f"Age: {features[0]}", styles["Normal"]))
    content.append(Paragraph(f"Cholesterol: {features[1]}", styles["Normal"]))
    content.append(Paragraph(f"BP: {features[2]}", styles["Normal"]))
    content.append(Spacer(1,10))
    content.append(Paragraph(f"Result: {result}", styles["Normal"]))
    content.append(Paragraph(f"Risk: {round(prob,2)}%", styles["Normal"]))

    doc.build(content)
    return file

# ---------------- PREDICTION ----------------
def predict(features):

    raw_features = np.array(features).reshape(1, -1)   # for UI

    scaled_features = raw_features.copy()

    # scale ONLY for model
    if scaler:
        scaled_features = scaler.transform(raw_features)

    prediction = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1] * 100

    st.write("Prediction:", prediction)
    st.write("Probability:", prob)

    # ✅ show RAW values (NO decimals)
    show_kpi(raw_features[0])

    if prediction == 1:
        result = "HIGH RISK"
        st.markdown(f"<div class='high'>⚠️ HIGH RISK ({round(prob,2)}%)</div>", unsafe_allow_html=True)
    else:
        result = "LOW RISK"
        st.markdown(f"<div class='low'>✅ LOW RISK ({round(prob,2)}%)</div>", unsafe_allow_html=True)

    show_gauge(prob)
    show_graph(raw_features[0], prob)

    pdf = generate_pdf(raw_features[0], result, prob)
    with open(pdf, "rb") as f:
        st.download_button("📄 Download Report", f)
# ---------------- MAIN ----------------
if option == "Manual Entry":

    st.markdown("<div class='card'><h3>Enter Patient Details</h3></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age",1,100)
    chol = col2.number_input("Cholesterol",50,400)
    bp = col3.number_input("Blood Pressure",40,200)

    if st.button("Run Diagnosis"):
        predict([age,chol,bp])

else:
    st.markdown("<div class='card'><h3>Upload Medical Report</h3></div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file:
        text = extract_pdf(file)
        features = extract_features(text)
        predict(features)

# ---------------- HISTORY ----------------
st.markdown("---")
if st.button("View History"):
    try:
        st.dataframe(pd.read_csv("history.csv"))
    except:
        st.warning("No records found")