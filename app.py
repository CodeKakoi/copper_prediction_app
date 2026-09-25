# app.py - Complete Copper Prediction App

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title="Copper Prediction Tool",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Oyu Tolgoi Copper Concentration Prediction Tool")

st.markdown("""
This tool uses a **Random Forest machine learning model** to predict 
copper (Cu) concentration based on spatial coordinates and geochemical measurements.
""")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("copper_prediction_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found: 'copper_prediction_model.pkl'")
        st.stop()

rf = load_model()

# ========== SINGLE PREDICTION ==========
st.header("📊 Single Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Spatial Coordinates")
    x = st.number_input("X coordinate", value=100.0, step=10.0)
    y = st.number_input("Y coordinate", value=50.0, step=10.0)
    z = st.number_input("Z coordinate (depth)", value=500.0, step=50.0)

with col2:
    st.subheader("Geochemical Elements")
    mo = st.number_input("Molybdenum (Mo)", value=0.5, step=0.1)
    cp = st.number_input("Chalcopyrite (Cp)", value=0.2, step=0.05)

with col3:
    st.subheader("Other Measurements")
    py = st.number_input("Pyrite (Py)", value=0.1, step=0.05)
    ts = st.number_input("Temperature/Stress (TS)", value=25.0, step=5.0)

if st.button("🔮 Predict Cu Concentration", type="primary", use_container_width=True):
    features = ["X", "Y", "Z", "Mo", "Cp", "Py", "TS"]
    
    input_data = pd.DataFrame({
        "X": [x],
        "Y": [y],
        "Z": [z],
        "Mo": [mo],
        "Cp": [cp],
        "Py": [py],
        "TS": [ts]
    })
    
    try:
        prediction = rf.predict(input_data[features])[0]
        
        st.success(f"✓ Prediction successful!", icon="✅")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Cu Concentration", value=f"{prediction:.4f}")
        
        with col2:
            st.info(f"""
            **Input Summary:**
            - Location: X={x}, Y={y}, Z={z}
            - Elements: Mo={mo}, Cp={cp}, Py={py}, TS={ts}
            """)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ========== BATCH PREDICTION ==========
st.header("📁 Batch Prediction (Upload CSV)")

st.write("Upload a CSV file with columns: X, Y, Z, Mo, Cp, Py, TS")

st.text("Example CSV format:")
example_csv = """X,Y,Z,Mo,Cp,Py,TS
100,50,500,0.5,0.2,0.1,25
150,75,600,0.6,0.3,0.15,30
200,100,700,0.7,0.4,0.2,35"""
st.code(example_csv, language="csv")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(df)} rows")
        
        features = ["X", "Y", "Z", "Mo", "Cp", "Py", "TS"]
        
        missing_cols = [col for col in features if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        else:
            predictions = rf.predict(df[features])
            df["Predicted_Cu"] = predictions
            
            st.subheader("📊 Results")
            st.dataframe(df, use_container_width=True)
            
            st.subheader("📈 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{predictions.mean():.4f}")
            with col2:
                st.metric("Min", f"{predictions.min():.4f}")
            with col3:
                st.metric("Max", f"{predictions.max():.4f}")
            with col4:
                st.metric("Std Dev", f"{predictions.std():.4f}")
            
            csv_result = df.to_csv(index=False)
            st.download_button(
                label="📥 Download predictions as CSV",
                data=csv_result,
                file_name="copper_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit & scikit-learn")