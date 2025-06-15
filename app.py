import streamlit as st
import pandas as pd
import joblib

# ========== CONFIGURATION ==========
st.set_page_config(page_title="Temperature Prediction Model", page_icon="🌤️")

# ========== LOAD MODEL ==========
model_path = "temperature_predictor_v3.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"❌ Model file '{model_path}' not found in the current directory.")
    st.stop()

# ========== PAGE HEADER ==========
st.title("🌤️ Smart Weather Temperature Predictor")
st.markdown("Use this app to get predictions for **Minimum**, **Maximum**, and **Average** temperatures based on your selected inputs.")

# ========== USER INPUT ==========
st.header("🔧 Input Weather Details")
city = st.selectbox("📍 Select City", ["Islamabad", "Lahore", "Quetta"])
year = st.number_input("📅 Year", min_value=2000, max_value=2100, value=2025)
month = st.number_input("📆 Month", min_value=1, max_value=12, value=6)
day = st.number_input("📆 Day", min_value=1, max_value=31, value=15)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
dew_point = st.slider("🌫️ Dew Point (°C)", -20, 40, 10)
pressure = st.slider("🌡️ Pressure (hPa)", 900, 1100, 1010)
wspd = st.slider("🌬️ Wind Speed (km/h)", 0, 100, 10)

# ========== PREDICTION ==========
if st.button("🚀 Predict Temperature"):
    input_df = pd.DataFrame([{
        "day": day,
        "month": month,
        "year": year,
        "humidity": humidity,
        "wspd": wspd,
        "pressure": pressure,
        "dew_point": dew_point,
        "city": city
    }])

    try:
        prediction = model.predict(input_df)
        tmin, tmax, tavg = prediction[0]

        st.header("📈 Predicted Temperatures")
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Tmin", f"{tmin:.2f} °C", help="Minimum Temperature")
        col2.metric("🔥 Tmax", f"{tmax:.2f} °C", help="Maximum Temperature")
        col3.metric("🌤️ Tavg", f"{tavg:.2f} °C", help="Average Temperature")

        st.success("✅ Prediction complete!")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
