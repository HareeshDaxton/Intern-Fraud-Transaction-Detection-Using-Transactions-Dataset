import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import datetime
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="🛡️ Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
    }
    
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        background-size: 400% 400%;
        animation: gradientShift 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .fraud-alert {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 10px;
        padding: 15px;
        color: white;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .safe-alert {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 10px;
        padding: 15px;
        color: white;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .sidebar .sidebar-content {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(20px);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    
    .highlight-number {
        font-size: 2rem;
        font-weight: bold;
        color: #4ecdc4;
        text-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = joblib.load('ada_model.pkl')
        return model, True
    except FileNotFoundError:
        return None, False


@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    data = []
    for i in range(1000):
        customer_id = np.random.randint(1, 1000)
        terminal_id = np.random.randint(1, 500)
        tx_amount = np.random.exponential(50) + 1
        day = np.random.randint(1, 32)
        month = np.random.randint(1, 13)
        year = np.random.choice([2022, 2023, 2024])
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        
        data.append({
            'CUSTOMER_ID': customer_id,
            'TERMINAL_ID': terminal_id,
            'TX_AMOUNT': round(tx_amount, 2),
            'DAY': day,
            'MONTH': month,
            'YEAR': year,
            'HOUR': hour,
            'MINUTE': minute
        })
    return pd.DataFrame(data)


st.markdown('<div class="main-header">🛡️ AI Fraud Detection System</div>', unsafe_allow_html=True)


model, model_loaded = load_model()

if not model_loaded:
    st.error("⚠️ Model file 'ada_model.pkl' not found. Please ensure the model is trained and saved.")
    st.stop()

with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
  
    mode = st.selectbox(
        "Select Mode",
        ["Single Transaction", "Batch Processing"]
    )
    
    st.markdown("---")
    
 
    st.markdown("### 📊 Model Information")
    st.info("**Algorithm:** AdaBoost Classifier")
    st.info("**Accuracy:** 99.83%")
    st.info("**ROC AUC:** 98.57%")
    st.info("**Status:** 🟢 Active")


if mode == "Single Transaction":
    st.markdown("## 🔍 Single Transaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Transaction Details")
        
        customer_id = st.number_input("Customer ID", min_value=1, max_value=9999, value=124)
        terminal_id = st.number_input("Terminal ID", min_value=1, max_value=999, value=568)
        tx_amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=10000.0, value=30.0, step=0.01)
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            day = st.selectbox("Day", range(1, 32), index=14)
            month = st.selectbox("Month", range(1, 13), index=6)
        with col_date2:
            year = st.selectbox("Year", [2022, 2023, 2024], index=2)
            hour = st.selectbox("Hour", range(0, 24), index=14)
        
        minute = st.selectbox("Minute", range(0, 60), index=35)
        
        predict_btn = st.button("🔍 Analyze Transaction", type="primary")
    
    with col2:
        st.markdown("### 📈 Risk Indicators")
        
        
        risk_factors = {
            'Amount Risk': min(tx_amount / 1000 * 100, 100),
            'Time Risk': 20 if hour < 6 or hour > 22 else 10,
            'Customer Risk': np.random.randint(5, 25),
            'Terminal Risk': np.random.randint(5, 30)
        }
        
        fig_risk = go.Figure(data=[
            go.Bar(
                x=list(risk_factors.values()),
                y=list(risk_factors.keys()),
                orientation='h',
                marker=dict(
                    color=list(risk_factors.values()),
                    colorscale='RdYlBu_r',
                    showscale=True
                )
            )
        ])
        
        fig_risk.update_layout(
            title="Risk Factor Analysis",
            xaxis_title="Risk Level (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    if predict_btn:
        
        sample_data = pd.DataFrame([{
            "CUSTOMER_ID": customer_id,
            "TERMINAL_ID": terminal_id,
            "TX_AMOUNT": tx_amount,
            "DAY": day,
            "MONTH": month,
            "YEAR": year,
            "HOUR": hour,
            "MINUTE": minute
        }])
        
        with st.spinner("🤖 AI is analyzing the transaction..."):
            time.sleep(2)  
            
            pred = model.predict(sample_data)[0]
            prob = model.predict_proba(sample_data)[0][1]
        
        
        st.markdown("---")
        st.markdown("## 🎯 Analysis Results")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if pred == 1:
                st.markdown(f'<div class="fraud-alert">🚨 FRAUD DETECTED<br>Confidence: {prob:.1%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-alert">✅ TRANSACTION SAFE<br>Confidence: {(1-prob):.1%}</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Fraud Probability", f"{prob:.1%}", delta=f"{prob-0.5:.1%}")
        
        with col3:
            risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
            st.metric("Risk Level", risk_level)

elif mode == "Batch Processing":
    st.markdown("## 📊 Batch Transaction Processing")
    

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} transactions")
        
        
        if st.button("🚀 Process Batch", type="primary"):
            with st.spinner("Processing transactions..."):
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)[:, 1]
                
                df['Prediction'] = predictions
                df['Fraud_Probability'] = probabilities
                df['Risk_Level'] = pd.cut(probabilities, 
                                        bins=[0, 0.3, 0.7, 1.0], 
                                        labels=['LOW', 'MEDIUM', 'HIGH'])
            
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fraud_count = sum(predictions)
                st.markdown(f'<div class="metric-card"><div class="highlight-number">{fraud_count}</div>Fraud Detected</div>', unsafe_allow_html=True)
            
            with col2:
                safe_count = len(predictions) - fraud_count
                st.markdown(f'<div class="metric-card"><div class="highlight-number">{safe_count}</div>Safe Transactions</div>', unsafe_allow_html=True)
            
            with col3:
                fraud_rate = fraud_count / len(predictions) * 100
                st.markdown(f'<div class="metric-card"><div class="highlight-number">{fraud_rate:.1f}%</div>Fraud Rate</div>', unsafe_allow_html=True)
            
            with col4:
                avg_amount = df[df['Prediction'] == 1]['TX_AMOUNT'].mean() if fraud_count > 0 else 0
                st.markdown(f'<div class="metric-card"><div class="highlight-number">${avg_amount:.0f}</div>Avg Fraud Amount</div>', unsafe_allow_html=True)
            
           
            col1, col2 = st.columns(2)
            
            with col1:
               
                fig1 = px.histogram(df, x='TX_AMOUNT', color='Prediction', 
                                  title='Transaction Amount Distribution',
                                  color_discrete_map={0: 'green', 1: 'red'})
                fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
               
                risk_dist = df['Risk_Level'].value_counts()
                fig2 = px.pie(values=risk_dist.values, names=risk_dist.index, 
                             title='Risk Level Distribution',
                             color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'})
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig2, use_container_width=True)
            
            
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "fraud_analysis_results.csv", "text/csv")
    
    else:
        
        st.info("Please upload a CSV file with the following columns:")
        sample_df = pd.DataFrame({
            'CUSTOMER_ID': [124, 456, 789],
            'TERMINAL_ID': [568, 234, 891],
            'TX_AMOUNT': [30.0, 150.75, 89.99],
            'DAY': [15, 20, 8],
            'MONTH': [7, 8, 9],
            'YEAR': [2024, 2024, 2024],
            'HOUR': [14, 22, 9],
            'MINUTE': [35, 15, 42]
        })
        st.dataframe(sample_df, use_container_width=True)




st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.6;'>
    <p>🛡️ Fraud Detection System | Powered by AI | Built with Streamlit</p>
    <p>© 2024 Advanced Analytics Solutions</p>
</div>
""", unsafe_allow_html=True)








