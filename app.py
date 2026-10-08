import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Solar Power Dashboard", page_icon="☀️", layout="wide")

# ---------------- Constants (Colab Cell 26 ke output se) ----------------
IRR_MAX = 1.22
TEMP_MIN, TEMP_MAX = 20.4, 39.2
INVERTERS_PER_PLANT = 22
MIN_EXPECTED_KW = 100.0  # isse kam expected power par fault judge nahi karte
WARN_RATIO, FAULT_RATIO = 0.8, 0.5  # asli/expected power ki hadein (heuristic)
NO_SUN = "Dhoop kam (judge nahi)"

CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Multan": (30.1575, 71.5249),
    "Faisalabad": (31.4504, 73.1350),
    "Peshawar": (34.0151, 71.5805),
    "Quetta": (30.1798, 66.9750),
    "Hyderabad": (25.3960, 68.3578),
    "Bahawalnagar": (29.9987, 73.2536),
}


@st.cache_resource
def load_model():
    bundle = joblib.load("solar_model.joblib")
    return bundle["model"], bundle["features"]


model, features = load_model()


# ---------------- Core functions ----------------
def predict_power(irradiation, ambient, is_plant2):
    """Aik inverter ki AC power (kW). Temperature training range mein clip hota hai,
    aur dhoop 0 ho to power 0."""
    irr, amb, plant = np.broadcast_arrays(
        np.atleast_1d(np.asarray(irradiation, dtype=float)),
        np.atleast_1d(np.asarray(ambient, dtype=float)),
        np.atleast_1d(np.asarray(is_plant2, dtype=float)),
    )
    irr, amb, plant = irr.copy(), amb.copy(), plant.copy()
    X = pd.DataFrame(
        {
            "IRRADIATION": np.clip(irr, 0, None),
            "AMBIENT_TEMPERATURE": np.clip(amb, TEMP_MIN, TEMP_MAX),
            "IS_PLANT2": plant,
        }
    )[features]
    pred = np.clip(model.predict(X), 0, None)
    pred[irr <= 0] = 0.0
    return pred


def stc_reference(is_plant2):
    """Reference: 1.0 kW/m2 dhoop aur 25 C par aik inverter ki power."""
    return float(predict_power(1.0, 25.0, is_plant2)[0])


def classify(expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ratio = actual / np.maximum(expected, 1e-9)
    status = np.full(len(expected), NO_SUN, dtype=object)
    ok = expected >= MIN_EXPECTED_KW
    status[ok & (ratio >= WARN_RATIO)] = "Normal"
    status[ok & (ratio < WARN_RATIO) & (ratio >= FAULT_RATIO)] = "Warning"
    status[ok & (ratio < FAULT_RATIO)] = "Fault"
    ratio = np.where(ok, ratio, np.nan)
    return ratio, status


def analyze(df, default_is_plant2, fallback_interval_min):
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    need = ["IRRADIATION", "AMBIENT_TEMPERATURE", "AC_POWER"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError("CSV mein ye columns nahi mile: " + ", ".join(missing))
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=need).reset_index(drop=True)
    if d.empty:
        raise ValueError("CSV mein koi valid row nahi mili.")

    if "PLANT_ID" in d.columns:
        is2 = (pd.to_numeric(d["PLANT_ID"], errors="coerce") == 4136001).astype(int).values
    else:
        is2 = np.full(len(d), int(default_is_plant2))

    interval = float(fallback_interval_min)
    if "DATE_TIME" in d.columns:
        d["DATE_TIME"] = pd.to_datetime(d["DATE_TIME"], errors="coerce")
        ts = d["DATE_TIME"].dropna().drop_duplicates().sort_values()
        if len(ts) > 1:
            diffs = ts.diff().dropna().dt.total_seconds() / 60.0
            diffs = diffs[diffs > 0]
            if len(diffs):
                interval = float(diffs.median())

    d["EXPECTED_KW"] = predict_power(
        d["IRRADIATION"].values, d["AMBIENT_TEMPERATURE"].values, is2
    )
    ratio, status = classify(d["EXPECTED_KW"].values, d["AC_POWER"].values)
    d["RATIO"] = ratio
    d["STATUS"] = status
    flagged = d["STATUS"].isin(["Warning", "Fault"])
    d["LOSS_KWH"] = np.where(
        flagged, np.clip(d["EXPECTED_KW"] - d["AC_POWER"], 0, None) * interval / 60.0, 0.0
    )
    return d, interval


def make_demo(is_plant2):
    """Nakli (demo) data: 4 inverters, aik din. Sirf dikhane ke liye."""
    rng = np.random.default_rng(7)
    times = pd.date_range("2020-06-13 05:00", "2020-06-13 19:00", freq="15min")
    h = times.hour.values + times.minute.values / 60.0
    irr = np.clip(0.95 * np.sin(np.pi * (h - 6) / 13), 0, None)
    amb = 24 + 8 * np.clip(np.sin(np.pi * (h - 6) / 13), 0, None)
    expected = predict_power(irr, amb, is_plant2)
    rows = []
    for inv in ["DEMO-INV-01", "DEMO-INV-02", "DEMO-INV-03", "DEMO-INV-04"]:
        noise = rng.normal(1.0, 0.03, len(times))
        factor = np.ones(len(times))
        if inv == "DEMO-INV-02":
            factor *= 0.7  # kam performance
        if inv == "DEMO-INV-03":
            factor[h >= 11] = 0.25  # dopahar se fault
        actual = np.clip(expected * factor * noise, 0, None)
        rows.append(
            pd.DataFrame(
                {
                    "DATE_TIME": times,
                    "SOURCE_KEY": inv,
                    "IRRADIATION": irr,
                    "AMBIENT_TEMPERATURE": amb,
                    "AC_POWER": actual,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast(lat, lon):
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "Asia/Karachi",
            "forecast_days": 3,
        },
        timeout=15,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(h["time"]),
            "temp_c": h["temperature_2m"],
            "ghi_wm2": h["shortwave_radiation"],
        }
    )


# ---------------- Sidebar ----------------
st.sidebar.title("☀️ Solar Dashboard")
plant_name = st.sidebar.selectbox("Solar plant (training data)", ["Plant 1", "Plant 2"])
is_plant2 = 1 if plant_name == "Plant 2" else 0
st.sidebar.caption(
    "Model: XGBoost. Data: India ke 2 solar plants, 15 May se 17 June 2020, "
    "har 15 minute. Features: irradiation, ambient temperature, plant."
)

st.title("☀️ Solar Power Prediction, Forecast aur Fault Check")

tab_pred, tab_fc, tab_fault, tab_phys = st.tabs(
    ["🔆 Predict", "📅 Forecast", "🛠 Fault check", "📊 Physics aur model"]
)

# ---------------- Tab 1: Predict ----------------
with tab_pred:
    st.write(
        "Dhoop ki shiddat (irradiation) aur hawa ke temperature se solar plant "
        "ke aik inverter ki AC power ka andaza."
    )
    irradiation = st.slider(
        "Irradiation (kW/m²)", 0.0, IRR_MAX, 0.6, 0.01, key="p_irr"
    )
    ambient = st.slider(
        "Ambient temperature (°C)", float(TEMP_MIN), float(TEMP_MAX), 28.0, 0.1, key="p_temp"
    )
    per_inv = float(predict_power(irradiation, ambient, is_plant2)[0])
    c1, c2 = st.columns(2)
    c1.metric("Aik inverter ki AC power", f"{per_inv:,.0f} kW")
    c2.metric(
        f"Poora plant (~{INVERTERS_PER_PLANT} inverters)",
        f"{per_inv * INVERTERS_PER_PLANT:,.0f} kW",
    )
    if irradiation == 0:
        st.info("Dhoop 0 hai (raat), is liye power 0 hai.")
    grid = np.linspace(0, IRR_MAX, 60)
    curve = pd.DataFrame(
        {
            "Irradiation (kW/m²)": grid,
            "AC power per inverter (kW)": predict_power(grid, ambient, is_plant2),
        }
    ).set_index("Irradiation (kW/m²)")
    st.subheader(f"Power vs Irradiation ({ambient:.1f} °C par)")
    st.line_chart(curve)

# ---------------- Tab 2: Forecast ----------------
with tab_fc:
    st.subheader("📅 Solar generation forecast")
    st.write(
        "Free weather API (Open-Meteo) se dhoop aur temperature le kar aap ke "
        "solar system ki andazi generation."
    )
    c1, c2 = st.columns(2)
    city = c1.selectbox("Shehr", list(CITIES.keys()) + ["Custom (lat/lon)"], key="fc_city")
    if city == "Custom (lat/lon)":
        lat = c2.number_input("Latitude", value=31.5204, format="%.4f", key="fc_lat")
        lon = c2.number_input("Longitude", value=74.3587, format="%.4f", key="fc_lon")
    else:
        lat, lon = CITIES[city]
    c3, c4, c5 = st.columns(3)
    kwp = c3.number_input(
        "Aap ke solar system ka size (kW)", min_value=0.5, value=5.0, step=0.5, key="fc_kwp"
    )
    tariff = c4.number_input(
        "Bijli ka nirkh (PKR/unit)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Apne bill se nirkh daalo, taake bachat ka andaza nikal sake.",
        key="fc_tariff",
    )
    today = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None).normalize()
    day_opts = {
        "Aaj": today,
        "Kal": today + pd.Timedelta(days=1),
        "Parson": today + pd.Timedelta(days=2),
    }
    day_label = c5.radio("Din", list(day_opts.keys()), horizontal=True, index=1, key="fc_day")

    try:
        wx = fetch_forecast(round(float(lat), 4), round(float(lon), 4))
    except Exception:
        wx = None
        st.error("Mausam ka data nahi mil saka (internet ya API ka masla). Thori der baad dobara try karo.")

    if wx is not None:
        day = wx[wx["time"].dt.date == day_opts[day_label].date()].copy()
        if day.empty:
            st.warning("Is din ka data nahi mila.")
        else:
            irr = np.clip(day["ghi_wm2"].fillna(0).values / 1000.0, 0, None)
            temp = day["temp_c"].ffill().bfill().fillna(25.0).values
            pred = predict_power(irr, temp, is_plant2)
            factor = pred / stc_reference(is_plant2)
            kw = kwp * factor
            kwh = float(kw.sum())  # har row 1 ghante ki hai
            peak_i = int(np.argmax(kw))
            out_of_range = int((((temp < TEMP_MIN) | (temp > TEMP_MAX)) & (irr > 0)).sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total generation", f"{kwh:,.1f} kWh")
            m2.metric("Peak power", f"{kw[peak_i]:,.2f} kW")
            m3.metric("Peak waqt", day["time"].iloc[peak_i].strftime("%H:%M"))
            if tariff > 0:
                m4.metric("Bijli ki bachat (andaza)", f"PKR {kwh * tariff:,.0f}")
            else:
                m4.metric("Bijli ki bachat (andaza)", "nirkh daalo")

            labels = day["time"].dt.strftime("%H:%M").values
            left, right = st.columns(2)
            left.markdown("**Ghante ke hisaab se power (kW)**")
            left.area_chart(pd.DataFrame({"Power (kW)": kw}, index=labels))
            right.markdown("**Mausam: dhoop (kW/m²) aur temperature (°C)**")
            right.line_chart(
                pd.DataFrame({"Irradiation (kW/m²)": irr, "Temperature (°C)": temp}, index=labels)
            )

            if out_of_range:
                st.warning(
                    f"{out_of_range} ghanton mein temperature model ki training range "
                    f"({TEMP_MIN}-{TEMP_MAX} °C) se bahar hai. Un ghanton ka andaza kam bharosemand hai."
                )

            table = pd.DataFrame(
                {
                    "Waqt": labels,
                    "Irradiation (kW/m²)": irr.round(3),
                    "Temp (°C)": np.round(temp, 1),
                    "Power (kW)": kw.round(3),
                }
            )
            table = table[table["Irradiation (kW/m²)"] > 0]
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button(
                "Forecast CSV download karo",
                table.to_csv(index=False).encode("utf-8"),
                file_name="solar_forecast.csv",
                mime="text/csv",
                key="fc_dl",
            )

    with st.expander("Ye andaza kitna sahi hai? (limitations)"):
        st.markdown(
            f"""
- Model ne India ke 2 plants ka data dekha. Aap ke system ki power us se alag ho sakti hai,
  is liye generation **(aap ka size) x (plant ka relative output)** se nikali gayi hai.
  Relative output ka reference 1.0 kW/m² dhoop aur 25 °C par plant {plant_name} ki power hai.
- Weather API ki dhoop (GHI) aur plant ke sensor ki dhoop mein farq hota hai, aur panel ka tilt,
  dhool, saya aur wiring ke losses isme shamil nahi.
- Ye aik andaza hai, kisi installer ke hisaab ka badal nahi.
"""
        )

# ---------------- Tab 3: Fault check ----------------
with tab_fault:
    st.subheader("🛠 Inverter fault check")
    st.write(
        "Asli napi hui power ko model ke andaze se compare karte hain. Jo inverter dhoop "
        "mein bhi andaze se bohat kam power de, usay mark kar dete hain."
    )
    st.caption(
        f"Rule (heuristic): asli/expected >= {WARN_RATIO:.0%} Normal, "
        f"{FAULT_RATIO:.0%} se {WARN_RATIO:.0%} Warning, {FAULT_RATIO:.0%} se kam Fault. "
        f"Expected power {MIN_EXPECTED_KW:.0f} kW se kam ho to judge nahi karte."
    )
    mode = st.radio(
        "Kya check karna hai?",
        ["Aik reading", "CSV file (bohat saari readings)"],
        horizontal=True,
        key="f_mode",
    )

    if mode == "Aik reading":
        c1, c2, c3 = st.columns(3)
        irr_m = c1.slider("Irradiation (kW/m²)", 0.0, IRR_MAX, 0.8, 0.01, key="f_irr")
        amb_m = c2.slider(
            "Ambient temp (°C)", float(TEMP_MIN), float(TEMP_MAX), 30.0, 0.1, key="f_amb"
        )
        act = c3.number_input(
            "Inverter ki napi hui AC power (kW)", min_value=0.0, value=300.0, step=10.0, key="f_act"
        )
        exp_kw = predict_power(irr_m, amb_m, is_plant2)
        ratio, status = classify(exp_kw, np.array([act]))
        k1, k2, k3 = st.columns(3)
        k1.metric("Expected power", f"{exp_kw[0]:,.0f} kW")
        k2.metric("Asli power", f"{act:,.0f} kW")
        k3.metric("Asli / expected", "-" if np.isnan(ratio[0]) else f"{ratio[0]:.0%}")
        s = status[0]
        if s == "Normal":
            st.success("✅ Normal: inverter apni expected power ke qareeb chal raha hai.")
        elif s == "Warning":
            st.warning("⚠️ Warning: power expected se kam hai. Dhool, saya ya partial fault ho sakta hai.")
        elif s == "Fault":
            st.error("🚨 Fault: power expected se aadhi se bhi kam hai. Inverter ya panel string check karo.")
        else:
            st.info("Dhoop kam hai, is liye is reading se faisla nahi ho sakta.")
    else:
        use_demo = st.checkbox(
            "Demo data use karo (nakli data, sirf dikhane ke liye)", key="f_demo"
        )
        up = None if use_demo else st.file_uploader("CSV upload karo", type="csv", key="f_up")
        fallback = st.number_input(
            "Agar DATE_TIME na ho to do readings ke darmiyan minute",
            min_value=1,
            max_value=120,
            value=15,
            key="f_int",
        )
        data = None
        if use_demo:
            data = make_demo(is_plant2)
            st.caption("Ye nakli data hai: DEMO-INV-02 kam performance, DEMO-INV-03 dopahar se fault.")
        elif up is not None:
            try:
                data = pd.read_csv(up)
            except Exception:
                st.error("CSV file parh nahi saka. File ka format check karo.")

        if data is None:
            st.info(
                "Zaroori columns: IRRADIATION (kW/m²), AMBIENT_TEMPERATURE (°C), AC_POWER (kW). "
                "Optional: DATE_TIME, SOURCE_KEY (inverter ka naam), PLANT_ID (4135001 ya 4136001)."
            )
        else:
            try:
                res, interval = analyze(data, is_plant2, fallback)
            except ValueError as e:
                st.error(str(e))
            else:
                judged = res[res["STATUS"] != NO_SUN]
                st.caption(f"{len(res):,} rows, readings ke darmiyan {interval:g} minute.")
                if judged.empty:
                    st.info("Kisi row mein itni dhoop nahi thi ke faisla ho sake.")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Judge ki gayi rows", f"{len(judged):,}")
                    m2.metric("Fault", f"{(judged['STATUS'] == 'Fault').mean():.1%}")
                    m3.metric("Warning", f"{(judged['STATUS'] == 'Warning').mean():.1%}")
                    m4.metric("Zaya energy (andaza)", f"{res['LOSS_KWH'].sum():,.0f} kWh")

                    if "SOURCE_KEY" in res.columns:
                        g = (
                            judged.groupby("SOURCE_KEY")
                            .agg(
                                rows=("STATUS", "size"),
                                fault_pct=("STATUS", lambda s: 100 * (s == "Fault").mean()),
                                avg_ratio=("RATIO", "mean"),
                                loss_kwh=("LOSS_KWH", "sum"),
                            )
                            .round(2)
                            .sort_values("loss_kwh", ascending=False)
                        )
                        st.markdown("**Sab se zyada nuqsan wale inverters**")
                        st.dataframe(g.head(10), use_container_width=True)

                    if "DATE_TIME" in res.columns and res["DATE_TIME"].notna().any():
                        tl = (
                            res.dropna(subset=["DATE_TIME"])
                            .groupby("DATE_TIME")[["EXPECTED_KW", "AC_POWER"]]
                            .mean()
                        )
                        st.markdown("**Expected vs asli power (sab inverters ka average)**")
                        st.line_chart(tl)

                st.dataframe(res.head(500), use_container_width=True)
                st.download_button(
                    "Nateeje CSV download karo",
                    res.to_csv(index=False).encode("utf-8"),
                    file_name="fault_check_results.csv",
                    mime="text/csv",
                    key="f_dl",
                )

# ---------------- Tab 4: Physics aur model ----------------
with tab_phys:
    st.subheader("📊 Model physics kya seekha?")

    st.markdown("**1. Power vs dhoop, alag alag temperature par**")
    grid = np.linspace(0, IRR_MAX, 60)
    temps = [22, 30, 38]
    st.line_chart(
        pd.DataFrame(
            {f"{t} °C": predict_power(grid, t, is_plant2) for t in temps},
            index=pd.Index(grid.round(3), name="Irradiation (kW/m²)"),
        )
    )

    st.markdown("**2. Temperature ka asar (dhoop fixed rakh kar)**")
    fixed_irr = st.slider("Fixed irradiation (kW/m²)", 0.1, IRR_MAX, 0.8, 0.01, key="ph_irr")
    tgrid = np.linspace(TEMP_MIN, TEMP_MAX, 40)
    tpred = predict_power(fixed_irr, tgrid, is_plant2)
    st.line_chart(
        pd.DataFrame(
            {"AC power per inverter (kW)": tpred},
            index=pd.Index(tgrid.round(1), name="Ambient temperature (°C)"),
        )
    )
    spread = float(tpred.max() - tpred.min())
    st.caption(
        f"{TEMP_MIN} se {TEMP_MAX} °C tak power mein farq: {spread:,.0f} kW "
        f"({spread / max(float(tpred.mean()), 1e-9):.1%}). Model ke hisaab se temperature ka asar "
        "dhoop ke muqable mein chhota hai."
    )

    st.markdown("**3. Plant 1 vs Plant 2**")
    st.line_chart(
        pd.DataFrame(
            {
                "Plant 1": predict_power(grid, 28.0, 0),
                "Plant 2": predict_power(grid, 28.0, 1),
            },
            index=pd.Index(grid.round(3), name="Irradiation (kW/m²)"),
        )
    )
    st.caption(
        "Plant 2 ke training data se faulty rows hata diye gaye thay, is liye ye normal "
        "chalte hue inverters ka andaza hai."
    )

    st.markdown("**4. Feature importance (XGBoost)**")
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    st.bar_chart(imp)

    st.markdown("**5. Models ka taqabul (7 din ka test, 11 se 17 June 2020)**")
    comp = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest", "XGBoost"],
            "MAE (faults ke saath)": [49.9, 42.3, 42.2],
            "RMSE (faults ke saath)": [91.5, 88.6, 88.9],
            "R2 (faults ke saath)": [0.924, 0.929, 0.929],
            "MAE (saaf test)": [46.3, 38.8, 38.6],
            "RMSE (saaf test)": [77.3, 74.3, 74.3],
            "R2 (saaf test)": [0.946, 0.950, 0.950],
        }
    )
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.caption(
        "Teeno models ka farq chhota hai aur test sirf 7 din ka hai. Saaf test mein faulty "
        "inverters ki rows hata di gayi hain, is liye wo aasan test hai."
    )

    with st.expander("Physics aur model ke baare mein"):
        st.markdown(
            """
- Solar power dhoop ki shiddat ke taqriban **proportional** hoti hai, isi liye `IRRADIATION`
  model ka sab se important feature hai (lagbhag 93% importance).
- Garmi mein panel ki efficiency kam hoti hai, is liye hawa ka temperature bhi feature hai.
- `MODULE_TEMPERATURE` feature se hata diya gaya, kyunke us se tree models kuch aise halat
  par ghalat andaza dete thay jo training mein nadir thay.
- Training mein sirf din ke rows aur theek kaam karte hue inverters shamil the.
- Ye weather se power ka **estimation** hai, aur Forecast tab mein weather API ki prediction
  use karke forecast ban jata hai.
"""
        )
