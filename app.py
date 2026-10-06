import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Solar Power Prediction", page_icon="☀️", layout="centered")

# Training data ki hadein (Colab Cell 26 ke output se)
IRR_MAX = 1.22
TEMP_MIN, TEMP_MAX = 20.4, 39.2
INVERTERS_PER_PLANT = 22


@st.cache_resource
def load_model():
    bundle = joblib.load("solar_model.joblib")
    return bundle["model"], bundle["features"]


model, features = load_model()


def predict_power(irradiation, ambient_temp, is_plant2):
    """Aik inverter ki AC power (kW). Dhoop 0 ho to power 0."""
    irradiation = np.atleast_1d(irradiation).astype(float)
    X = pd.DataFrame(
        {
            "IRRADIATION": irradiation,
            "AMBIENT_TEMPERATURE": np.full_like(irradiation, ambient_temp),
            "IS_PLANT2": np.full_like(irradiation, is_plant2),
        }
    )[features]
    pred = np.clip(model.predict(X), 0, None)
    pred[irradiation <= 0] = 0.0
    return pred


st.title("☀️ Solar Power Generation Prediction")
st.write(
    "Dhoop ki shiddat (irradiation) aur hawa ke temperature se solar plant "
    "ke aik inverter ki AC power ka andaza. Model: XGBoost."
)

plant = st.selectbox("Solar plant", ["Plant 1", "Plant 2"])
irradiation = st.slider(
    "Irradiation (kW/m²)", min_value=0.0, max_value=IRR_MAX, value=0.6, step=0.01
)
ambient = st.slider(
    "Ambient temperature (°C)",
    min_value=float(TEMP_MIN),
    max_value=float(TEMP_MAX),
    value=28.0,
    step=0.1,
)

is_plant2 = 1 if plant == "Plant 2" else 0
per_inverter = float(predict_power(irradiation, ambient, is_plant2)[0])
whole_plant = per_inverter * INVERTERS_PER_PLANT

col1, col2 = st.columns(2)
col1.metric("Aik inverter ki AC power", f"{per_inverter:,.0f} kW")
col2.metric(f"Poora plant (~{INVERTERS_PER_PLANT} inverters)", f"{whole_plant:,.0f} kW")

if irradiation == 0:
    st.info("Dhoop 0 hai (raat), is liye power 0 hai.")

# Power vs irradiation curve (chuni hui temperature par)
irr_grid = np.linspace(0, IRR_MAX, 60)
curve = pd.DataFrame(
    {"Irradiation (kW/m²)": irr_grid, "AC power per inverter (kW)": predict_power(irr_grid, ambient, is_plant2)}
).set_index("Irradiation (kW/m²)")
st.subheader(f"Power vs Irradiation ({ambient:.1f} °C par)")
st.line_chart(curve)

with st.expander("Physics aur model ke baare mein"):
    st.markdown(
        """
- Solar power dhoop ki shiddat ke taqriban **proportional** hoti hai, isi liye `IRRADIATION`
  model ka sab se important feature hai.
- Garmi mein panel ki efficiency kam hoti hai, is liye hawa ka temperature bhi feature hai.
- Model ne **15 May se 17 June 2020** ka data dekha (India ke 2 solar plants, har 15 minute).
- Training mein sirf din ke rows aur theek kaam karte hue inverters shamil the. Is liye ye andaza
  **normal chalte hue inverter** ka hai, kharab inverter ka nahi.
- "Poora plant" wala number sirf andaza hai: aik inverter ki power ko 22 se guna kiya gaya hai.
- Ye weather se power ka **estimation** hai, kal ki forecast nahi. Sliders ki limits wahi hain
  jo training data mein thi.
"""
    )
