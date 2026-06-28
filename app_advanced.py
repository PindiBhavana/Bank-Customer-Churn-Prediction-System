import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App Header
st.markdown('<h1 style="color: #1e293b;">🏦 Customer Churn Intelligence</h1>', unsafe_allow_html=True)
st.markdown('### Predictive insights and real-time risk assessment for Bank Customers.', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e1:
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            st.error(f"⚠️ Model loading failed. Ensure 'model.pkl' is in the same directory.")
            return None

best_model = load_model()

if best_model:
    st.markdown("---")
    st.markdown("### 1. Build Customer Profile")
    st.markdown("Configure the customer's demographics and banking attributes below:")
    
    # Main Tabs for Input tailored for Bank Churn Data
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🏦 Bank Details", "💳 Account Usage"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", min_value=18, max_value=100, value=40)
        with col2:
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
        with col2:
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        with col2:
            has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active = st.selectbox("Is Active Member?", ["Yes", "No"])

    # Map inputs directly to the 11 columns the model expects
    # This avoids get_dummies() failing on single rows
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [1 if has_crcard == 'Yes' else 0],
        'IsActiveMember': [1 if is_active == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary],
        'Geography_Germany': [1 if geography == 'Germany' else 0],
        'Geography_Spain': [1 if geography == 'Spain' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0]
    })

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_prediction = st.button("🔮 Run Risk Assessment", use_container_width=True)

    if run_prediction:
        st.markdown("### 2. Real-Time Predictive Dashboard")
        
        try:
            # Check if model supports predict_proba
            if hasattr(best_model, 'predict_proba'):
                probs = best_model.predict_proba(input_data)[0]
                churn_prob = probs[1] * 100
            else:
                pred = best_model.predict(input_data)[0]
                churn_prob = 90.0 if pred == 1 else 10.0 # Fallback mock probability if not supported

            col_res1, col_res2 = st.columns([1.5, 2], gap="large")
            
            with col_res1:
                with st.container():
                    st.markdown("#### Real-time Risk Score", unsafe_allow_html=True)
                    
                    # Plotly Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = churn_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probability of Churn (%)"},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#1e293b"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': '#10b981'},
                                {'range': [40, 70], 'color': '#f59e0b'},
                                {'range': [70, 100], 'color': '#ef4444'}],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': churn_prob}
                        }
                    ))
                    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1e293b"})
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
            with col_res2:
                with st.container():
                    st.markdown("#### Status & Recommendation", unsafe_allow_html=True)
                    
                    # Recommendation Logic
                    if churn_prob < 40.0:
                        text_color = "#10b981"
                        status = "STABLE"
                        rec = "Customer is satisfied. Continue standard engagement."
                        pred_statement = "The customer is <b>NOT EXPECTED</b> to churn."
                    elif churn_prob <= 70.0:
                        text_color = "#d97706"
                        status = "MONITOR"
                        rec = "Offer proactive support, check-ins, or targeted bank services."
                        pred_statement = "The customer is <b>AT RISK</b> of churning."
                    else:
                        text_color = "#dc2626"
                        status = "CRITICAL"
                        rec = "Immediate retention action required! Consider fee waivers or high-yield offers."
                        pred_statement = "The customer <b>WILL LIKELY CHURN</b>."
                        
                    st.markdown(f"<h2 style='color: {text_color};'>{status}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div style='padding: 15px; border-left: 5px solid {text_color}; background-color: rgba(0,0,0,0.03); margin-bottom: 20px; font-size: 1.2rem;'>{pred_statement}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Action Plan:** {rec}")
                    
                    st.markdown("---")
                    st.markdown("#### Input Summary")
                    st.dataframe(input_data.T.rename(columns={0: "Value"}), height=250)
                    
        except Exception as pred_err:
            st.error(f"An error occurred during prediction: {pred_err}")
else:
    st.warning("Model not found. Please place 'model.pkl' in the directory.")
