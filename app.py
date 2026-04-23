import streamlit as st
import pickle
import pdfplumber
import re
import pandas as pd
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# ---------------- PAGE ----------------
st.set_page_config(page_title="Hospital AI System", layout="wide")

# ---------------- HOSPITAL UI ----------------
st.markdown("""
<style>

/*  Clean hospital background */
.stApp {
    background: #f1f5f9;
    color: #0f172a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Header */
.header {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Cards */
.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

/* KPI */
.kpi {
    text-align: center;
    border-left: 5px solid #3b82f6;
}

/* Buttons */
.stButton>button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
}

/* Risk colors */
.high {
    color: #dc2626;
    font-weight: bold;
    font-size: 20px;
}
.low {
    color: #16a34a;
    font-weight: bold;
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
<h2>🏥 Heart Disease Prediction System</h2>
<p>AI-powered clinical decision support</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Patient Input")
option = st.sidebar.radio("Select Mode", ["Manual Entry", "Upload Report"])

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

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

    col1.markdown(f"<div class='card kpi'><h4>Age</h4><h2>{features[0]}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card kpi'><h4>Cholesterol</h4><h2>{features[1]}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card kpi'><h4>Blood Pressure</h4><h2>{features[2]}</h2></div>", unsafe_allow_html=True)


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
    fig.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# 🔥 NEW GRAPH FUNCTION
def show_result_graph(features, prob):
    age, chol, bp = features

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Age", "Cholesterol", "BP"],
        y=[age, chol, bp],
        name="Patient Data"
    ))

    fig.add_trace(go.Scatter(
        x=["Age", "Cholesterol", "BP"],
        y=[prob/2, prob/1.5, prob],
        mode='lines+markers',
        name="Risk Trend"
    ))

    fig.update_layout(
        title="📊 Patient Health Analysis",
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)


def generate_pdf(features, result, prob):
    file="report.pdf"
    doc=SimpleDocTemplate(file)
    styles=getSampleStyleSheet()

    content=[]
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


def predict(features):
    pred = model.predict([features])
    prob = model.predict_proba([features])[0][1]*100

    st.markdown("<div class='card'><h3>Patient Summary</h3></div>", unsafe_allow_html=True)
    show_kpi(features)

    st.markdown("<div class='card'><h3>Diagnosis</h3></div>", unsafe_allow_html=True)

    if pred[0] == 1:
        st.markdown(f"<div class='high'>⚠️ High Risk ({round(prob,2)}%)</div>", unsafe_allow_html=True)
        result="HIGH RISK"
    else:
        st.markdown(f"<div class='low'>✅ Low Risk ({round(prob,2)}%)</div>", unsafe_allow_html=True)
        result="LOW RISK"

    show_gauge(prob)

    #  GRAPH ADDED HERE
    show_result_graph(features, prob)

    st.markdown("<div class='card'><h3>Download Report</h3></div>", unsafe_allow_html=True)
    pdf=generate_pdf(features,result,prob)
    with open(pdf,"rb") as f:
        st.download_button("Download Medical Report", f)


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


# ---------------- CHAT ----------------
st.markdown("---")
st.subheader("🩺 AI Assistant")

q = st.text_input("Ask medical question")
if st.button("Get Advice"):
    st.info("Consult a doctor for accurate diagnosis. Maintain healthy lifestyle.")


# ---------------- HISTORY ----------------
st.markdown("---")
if st.button("View Patient History"):
    try:
        st.dataframe(pd.read_csv("history.csv"))
    except:
        st.warning("No records found")