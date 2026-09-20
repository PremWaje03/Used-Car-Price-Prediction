import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# USED CAR PRICE PREDICTION
# MODERN STREAMLIT UI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "car_price_random_forest.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "car_price_preprocessor.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "car_price_metadata.joblib")

NUMERICAL_FEATURES = [
    "km",
    "No of Cylinder",
    "Valves per Cylinder",
    "Length",
    "Width",
    "Height",
    "Wheel Base",
    "Kerb Weight",
    "Seats",
    "Doors",
    "Cargo Volume",
    "Max Power Delivered",
    "Max Power At",
    "Max Torque Delivered",
    "Max Torque At",
    "vehicle_age",
]

CATEGORICAL_FEATURES = [
    "body",
    "transmission",
    "fuel",
    "oem",
    "model",
    "owner_type",
    "state",
    "Gear Box",
    "Drive Type",
    "Steering Type",
    "Front Brake Type",
    "Rear Brake Type",
    "Tyre Type",
    "Turbo Charger",
    "Super Charger",
    "Valve Configuration",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AutoValue AI | Used Car Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(59,130,246,0.12), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(139,92,246,0.10), transparent 25%),
            #0b0f17;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101722 0%, #0b1018 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .brand-box {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(59,130,246,0.20),
            rgba(139,92,246,0.16)
        );
        border: 1px solid rgba(255,255,255,0.09);
        margin-bottom: 20px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    .sidebar-card {
        padding: 14px 16px;
        border-radius: 14px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        margin-top: 12px;
    }

    .sidebar-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .sidebar-value {
        font-size: 15px;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 3px;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 30px 34px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(30,64,175,0.48),
                rgba(76,29,149,0.34)
            );
        border: 1px solid rgba(147,197,253,0.14);
        box-shadow: 0 20px 60px rgba(0,0,0,0.22);
        margin-bottom: 24px;
    }

    .hero-kicker {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(255,255,255,0.09);
        color: #bfdbfe;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: clamp(32px, 4vw, 52px);
        line-height: 1.05;
        font-weight: 850;
        color: #f8fafc;
        margin: 0;
    }

    .hero-text {
        max-width: 800px;
        color: #cbd5e1;
        font-size: 16px;
        line-height: 1.65;
        margin-top: 13px;
    }

    /* ---------- SECTION HEADINGS ---------- */

    .section-kicker {
        color: #60a5fa;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .section-description {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* ---------- FORM CARDS ---------- */

    .form-card {
        padding: 22px;
        border-radius: 18px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 12px 35px rgba(0,0,0,0.13);
        margin-bottom: 18px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        min-height: 50px;
        border-radius: 13px;
        border: 0;
        font-weight: 800;
        font-size: 15px;
        background: linear-gradient(135deg, #3b82f6, #7c3aed);
        color: white;
        box-shadow: 0 8px 24px rgba(59,130,246,0.24);
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(59,130,246,0.34);
    }

    /* ---------- INPUTS ---------- */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #151b27;
        border-color: rgba(255,255,255,0.08);
        border-radius: 10px;
    }

    /* ---------- PRICE RESULT ---------- */

    .price-result {
        position: relative;
        overflow: hidden;
        padding: 34px;
        border-radius: 24px;
        background:
            radial-gradient(circle at 20% 0%, rgba(34,197,94,0.15), transparent 35%),
            linear-gradient(135deg, #111827, #172033);
        border: 1px solid rgba(74,222,128,0.18);
        box-shadow: 0 18px 50px rgba(0,0,0,0.22);
        text-align: center;
        margin: 22px 0;
    }

    .price-result::after {
        content: "₹";
        position: absolute;
        right: 30px;
        top: -18px;
        font-size: 130px;
        font-weight: 900;
        color: rgba(255,255,255,0.025);
    }

    .price-label {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        font-weight: 800;
    }

    .price-value {
        color: #f8fafc;
        font-size: clamp(38px, 5vw, 62px);
        font-weight: 900;
        line-height: 1.1;
        margin: 10px 0;
    }

    .price-lakh {
        color: #4ade80;
        font-size: 21px;
        font-weight: 800;
    }

    .price-note {
        color: #64748b;
        font-size: 12px;
        margin-top: 12px;
    }

    /* ---------- SUMMARY CARDS ---------- */

    .summary-card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        height: 100%;
    }

    .summary-icon {
        font-size: 20px;
        margin-bottom: 8px;
    }

    .summary-label {
        color: #64748b;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .summary-value {
        color: #e2e8f0;
        font-size: 16px;
        font-weight: 750;
        margin-top: 4px;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        padding: 20px;
        border-radius: 18px;
        background: linear-gradient(145deg, #121a28, #0f1622);
        border: 1px solid rgba(255,255,255,0.07);
    }

    .metric-label {
        color: #64748b;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 850;
        margin-top: 7px;
    }

    .metric-help {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ---------- INFO / WORKFLOW ---------- */

    .workflow {
        padding: 24px;
        border-radius: 20px;
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(255,255,255,0.07);
    }

    .workflow-step {
        padding: 12px 15px;
        margin: 8px 0;
        border-radius: 11px;
        background: rgba(255,255,255,0.035);
        color: #cbd5e1;
        font-weight: 650;
    }

    .workflow-number {
        display: inline-block;
        width: 26px;
        height: 26px;
        line-height: 26px;
        text-align: center;
        border-radius: 50%;
        background: #2563eb;
        color: white;
        margin-right: 9px;
        font-size: 12px;
    }

    .tech-pill {
        display: inline-block;
        padding: 8px 12px;
        margin: 4px;
        border-radius: 999px;
        background: rgba(59,130,246,0.10);
        border: 1px solid rgba(96,165,250,0.14);
        color: #bfdbfe;
        font-size: 13px;
        font-weight: 650;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #475569;
        font-size: 12px;
        padding: 35px 0 10px;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def money(value):
    return f"₹{value:,.0f}"


def lakh(value):
    return f"₹{value / 100000:.2f} Lakh"


def clean_feature_name(name):
    """Convert preprocessing-generated feature names into readable labels."""
    name = str(name)

    if name.startswith("num__"):
        return name.replace("num__", "")

    if name.startswith("cat__"):
        name = name.replace("cat__", "")
        if "_" in name:
            feature, value = name.split("_", 1)
            return f"{feature}: {value}"
        return name

    return name


# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    metadata = {}
    if os.path.exists(METADATA_PATH):
        metadata = joblib.load(METADATA_PATH)

    return model, preprocessor, metadata


# ============================================================
# PREDICTION
# ============================================================

def predict_price(user_data, model, preprocessor):
    row = {feature: np.nan for feature in ALL_FEATURES}

    for key, value in user_data.items():
        if key in row:
            row[key] = value

    input_df = pd.DataFrame([row], columns=ALL_FEATURES)

    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]

    return max(0, float(prediction))


# ============================================================
# MODEL METRICS
# ============================================================

comparison_data = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "KNN Regression",
        "Decision Tree Regression",
        "Random Forest Regression",
        "Tuned Random Forest Regression",
    ],
    "MAE": [
        185114.607656,
        119556.909477,
        124371.998208,
        96330.799682,
        96062.310840,
    ],
    "MSE": [
        407171971660.89,
        278224225104.80,
        179589398934.57,
        145827561103.76,
        141801600000.00,
    ],
    "RMSE": [
        638100.283389,
        527469.643776,
        423779.894444,
        381873.750216,
        376565.491928,
    ],
    "R2": [
        0.695697,
        0.792067,
        0.865783,
        0.891015,
        0.894024,
    ],
})


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">🚘 AutoValue AI</div>
            <div class="brand-subtitle">
                Used Car Price Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "APPLICATION",
        [
            "🔮 Price Prediction",
            "📊 Model Analytics",
            "🌟 Feature Insights",
            "ℹ️ About Project",
        ],
        label_visibility="visible",
    )

    st.markdown("---")

    st.markdown("### 🤖 Active Model")

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-label">Algorithm</div>
            <div class="sidebar-value">Random Forest Regression</div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-label">Task</div>
            <div class="sidebar-value">Supervised Regression</div>
        </div>

        <div class="sidebar-card">
            <div class="sidebar-label">Target</div>
            <div class="sidebar-value">listed_price</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("College Machine Learning Project")
    st.caption("AutoValue AI • Used Car Resale Prediction")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model, preprocessor, metadata = load_artifacts()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.stop()

reference_year = int(metadata.get("reference_year", 2024))


# ============================================================
# PAGE 1 — PRICE PREDICTION
# ============================================================

if page == "🔮 Price Prediction":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Machine Learning • Regression</div>
            <h1 class="hero-title">Know the value of your car.</h1>
            <div class="hero-text">
                Estimate the resale/listed price of a used vehicle using
                vehicle age, mileage, specifications and market-related
                categorical information.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-kicker">01 • Vehicle Profile</div>
        <div class="section-title">Tell us about the vehicle</div>
        <div class="section-description">
            Enter the main details first. Technical specifications are optional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            model_year = st.number_input(
                "Model Year",
                min_value=1980,
                max_value=reference_year,
                value=min(2020, reference_year),
                step=1,
            )

            km = st.number_input(
                "Kilometers Driven",
                min_value=0.0,
                value=45000.0,
                step=1000.0,
            )

            body = st.selectbox(
                "Body Type",
                [
                    "suv",
                    "sedan",
                    "hatchback",
                    "muv",
                    "coupe",
                    "convertible",
                    "wagon",
                    "pickup",
                ],
            )

        with col2:
            transmission = st.selectbox(
                "Transmission",
                ["manual", "automatic"],
            )

            fuel = st.selectbox(
                "Fuel Type",
                ["petrol", "diesel", "cng", "electric", "lpg", "hybrid"],
            )

            oem = st.text_input(
                "Brand / OEM",
                value="maruti",
                placeholder="e.g. maruti, hyundai, tata",
            ).strip().lower()

        with col3:
            model_name = st.text_input(
                "Car Model",
                value="swift",
                placeholder="e.g. swift, creta, nexon",
            ).strip().lower()

            owner_type = st.text_input(
                "Owner Type",
                value="first owner",
                placeholder="e.g. first owner",
            ).strip().lower()

            state = st.text_input(
                "State",
                value="maharashtra",
                placeholder="e.g. maharashtra",
            ).strip().lower()

    # Technical specifications
    with st.expander("⚙️ Add Technical Specifications", expanded=False):

        st.caption(
            "Optional fields. Leave them blank if the specification is unknown."
        )

        t1, t2, t3 = st.columns(3)

        with t1:
            no_cylinder = st.number_input(
                "No of Cylinder",
                min_value=1.0,
                value=None,
                placeholder="Optional",
            )

            valves = st.number_input(
                "Valves per Cylinder",
                min_value=1.0,
                value=None,
                placeholder="Optional",
            )

            length = st.number_input(
                "Length",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

            width = st.number_input(
                "Width",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

            height = st.number_input(
                "Height",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

        with t2:
            wheel_base = st.number_input(
                "Wheel Base",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

            kerb_weight = st.number_input(
                "Kerb Weight",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

            seats = st.number_input(
                "Seats",
                min_value=1.0,
                value=None,
                placeholder="Optional",
            )

            doors = st.number_input(
                "Doors",
                min_value=1.0,
                value=None,
                placeholder="Optional",
            )

            cargo_volume = st.number_input(
                "Cargo Volume",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

        with t3:
            max_power = st.number_input(
                "Max Power Delivered",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

            max_power_at = st.number_input(
                "Max Power At",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

            max_torque = st.number_input(
                "Max Torque Delivered",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

            max_torque_at = st.number_input(
                "Max Torque At",
                min_value=0.0,
                value=None,
                placeholder="Optional",
            )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "✨  ESTIMATE RESALE VALUE",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:

        if not oem:
            st.error("Please enter the Brand / OEM.")
            st.stop()

        if not model_name:
            st.error("Please enter the Car Model.")
            st.stop()

        if not owner_type:
            st.error("Please enter the Owner Type.")
            st.stop()

        if not state:
            st.error("Please enter the State.")
            st.stop()

        vehicle_age = reference_year - int(model_year)

        user_data = {
            "km": km,
            "vehicle_age": vehicle_age,
            "body": body,
            "transmission": transmission,
            "fuel": fuel,
            "oem": oem,
            "model": model_name,
            "owner_type": owner_type,
            "state": state,
            "No of Cylinder": no_cylinder,
            "Valves per Cylinder": valves,
            "Length": length,
            "Width": width,
            "Height": height,
            "Wheel Base": wheel_base,
            "Kerb Weight": kerb_weight,
            "Seats": seats,
            "Doors": doors,
            "Cargo Volume": cargo_volume,
            "Max Power Delivered": max_power,
            "Max Power At": max_power_at,
            "Max Torque Delivered": max_torque,
            "Max Torque At": max_torque_at,
        }

        try:
            prediction = predict_price(
                user_data,
                model,
                preprocessor,
            )

            st.success("Prediction generated successfully!")

            st.markdown(
                f"""
                <div class="price-result">
                    <div class="price-label">Estimated Resale Value</div>
                    <div class="price-value">{money(prediction)}</div>
                    <div class="price-lakh">{lakh(prediction)}</div>
                    <div class="price-note">
                        ML estimate based on the trained used-car dataset
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="section-kicker">Prediction Summary</div>
                <div class="section-title">Vehicle snapshot</div>
                """,
                unsafe_allow_html=True,
            )

            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-icon">🚘</div>
                        <div class="summary-label">Vehicle</div>
                        <div class="summary-value">
                            {oem.title()} {model_name.title()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with s2:
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-icon">📅</div>
                        <div class="summary-label">Age</div>
                        <div class="summary-value">
                            {vehicle_age} years old
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with s3:
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-icon">🛣️</div>
                        <div class="summary-label">Mileage</div>
                        <div class="summary-value">
                            {km:,.0f} km
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with s4:
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-icon">📍</div>
                        <div class="summary-label">Location</div>
                        <div class="summary-value">
                            {state.title()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Important: this is a machine-learning estimate, not a guaranteed "
                "market price. Actual prices can vary based on condition, demand, "
                "location and factors not represented in the dataset."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))


# ============================================================
# PAGE 2 — MODEL ANALYTICS
# ============================================================

elif page == "📊 Model Analytics":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Model Evaluation</div>
            <h1 class="hero-title">How the models performed.</h1>
            <div class="hero-text">
                Compare the regression algorithms using MAE, MSE, RMSE and R²
                on the test dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best = comparison_data.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Best Test MAE</div>
                <div class="metric-value">₹{best["MAE"]:,.0f}</div>
                <div class="metric-help">Mean Absolute Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Best Test RMSE</div>
                <div class="metric-value">₹{best["RMSE"]:,.0f}</div>
                <div class="metric-help">Root Mean Squared Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test R²</div>
                <div class="metric-value">{best["R2"]:.4f}</div>
                <div class="metric-help">Coefficient of Determination</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Models Evaluated</div>
                <div class="metric-value">5</div>
                <div class="metric-help">Regression models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-kicker">Comparison</div>
        <div class="section-title">Evaluation metrics</div>
        <div class="section-description">
            Lower MAE/RMSE means smaller prediction errors. Higher R² means
            more target variation is explained on this test set.
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_df = comparison_data.copy()
    display_df["MAE"] = display_df["MAE"].map(lambda x: f"₹{x:,.2f}")
    display_df["MSE"] = display_df["MSE"].map(lambda x: f"{x:,.2f}")
    display_df["RMSE"] = display_df["RMSE"].map(lambda x: f"₹{x:,.2f}")
    display_df["R²"] = display_df["R2"].map(lambda x: f"{x:.4f}")
    display_df = display_df.drop(columns=["R2"])

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("#### 📈 R² Comparison")
        st.bar_chart(
            comparison_data.set_index("Model")[["R2"]],
            height=360,
        )

    with chart2:
        st.markdown("#### 📉 RMSE Comparison")
        st.bar_chart(
            comparison_data.set_index("Model")[["RMSE"]],
            height=360,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="workflow">
            <b style="color:#f8fafc;">Understanding the metrics</b><br><br>
            <span style="color:#94a3b8;">
            MAE measures the average absolute prediction error.
            MSE gives larger errors more penalty.
            RMSE expresses error in the same unit as price.
            R² indicates the proportion of target variation explained by the model.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 3 — FEATURE INSIGHTS
# ============================================================

elif page == "🌟 Feature Insights":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Model Explainability</div>
            <h1 class="hero-title">What influences the prediction?</h1>
            <div class="hero-text">
                Random Forest feature importance shows which processed input
                features contributed most to the model's decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    importance_csv = os.path.join(
        OUTPUT_DIR,
        "random_forest_feature_importance.csv",
    )

    importance_png = os.path.join(
        OUTPUT_DIR,
        "random_forest_feature_importance.png",
    )

    if os.path.exists(importance_csv):

        importance_df = pd.read_csv(importance_csv)

        if "Feature" in importance_df.columns and "Importance" in importance_df.columns:

            importance_df["Readable Feature"] = (
                importance_df["Feature"].apply(clean_feature_name)
            )

            importance_df = importance_df.sort_values(
                "Importance",
                ascending=False,
            )

            top20 = importance_df.head(20).copy()

            st.markdown(
                """
                <div class="section-kicker">Top Contributors</div>
                <div class="section-title">Feature importance</div>
                <div class="section-description">
                    Importance values describe the contribution of features
                    within this Random Forest model. They are not causal effects.
                </div>
                """,
                unsafe_allow_html=True,
            )

            left, right = st.columns([1.25, 1])

            with left:
                chart_df = top20[
                    ["Readable Feature", "Importance"]
                ].set_index("Readable Feature")

                st.bar_chart(
                    chart_df,
                    height=620,
                )

            with right:
                table_df = top20[
                    ["Readable Feature", "Importance"]
                ].copy()

                table_df["Importance"] = table_df["Importance"].map(
                    lambda x: f"{x * 100:.2f}%"
                )

                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True,
                    height=620,
                )

            if os.path.exists(importance_png):
                st.markdown("### 🖼️ Original Feature Importance Visualization")
                st.image(
                    importance_png,
                    use_container_width=True,
                )

            st.success(
                "Feature importance analysis loaded successfully from the "
                "saved Random Forest output."
            )

        else:
            st.warning(
                "The feature-importance CSV does not contain the expected columns."
            )

    else:
        st.info(
            "Feature importance output was not found. Run preprocessing_final.py "
            "once to generate it."
        )


# ============================================================
# PAGE 4 — ABOUT PROJECT
# ============================================================

else:

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">ML College Project</div>
            <h1 class="hero-title">Used Car Price Prediction.</h1>
            <div class="hero-text">
                A supervised machine-learning system that estimates the
                listed/resale price of used vehicles from historical vehicle data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3, a4 = st.columns(4)

    about_metrics = [
        ("🎯", "Task", "Regression"),
        ("🤖", "Primary Model", "Random Forest"),
        ("📌", "Target", "listed_price"),
        ("🧠", "ML Type", "Supervised"),
    ]

    for column, item in zip([a1, a2, a3, a4], about_metrics):
        icon, label, value = item
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:24px;">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:20px;">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            """
            <div class="section-kicker">Project Objective</div>
            <div class="section-title">Why this project?</div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "The objective is to develop a machine-learning system that "
            "estimates the listed/resale price of a used car from characteristics "
            "such as vehicle age, kilometers driven, engine specifications, "
            "dimensions and categorical information."
        )

        st.markdown("### 🧠 Models Implemented")

        models = [
            "Linear Regression",
            "KNN Regression",
            "Decision Tree Regression",
            "Random Forest Regression",
            "Random Forest hyperparameter tuning",
        ]

        for index, model_name in enumerate(models, 1):
            st.markdown(
                f"""
                <div class="workflow-step">
                    <span class="workflow-number">{index}</span>
                    {model_name}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            """
            <div class="section-kicker">Project Workflow</div>
            <div class="section-title">From dataset to prediction</div>
            """,
            unsafe_allow_html=True,
        )

        workflow = [
            "Dataset",
            "Data Cleaning",
            "Exploratory Data Analysis",
            "Feature Engineering",
            "Preprocessing",
            "Train/Test Split",
            "Regression Models",
            "Model Evaluation",
            "Saved Model",
            "Streamlit Prediction Application",
        ]

        st.markdown('<div class="workflow">', unsafe_allow_html=True)

        for index, step in enumerate(workflow, 1):
            st.markdown(
                f"""
                <div class="workflow-step">
                    <span class="workflow-number">{index}</span>
                    {step}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-kicker">Technology Stack</div>
        <div class="section-title">Tools used</div>
        """,
        unsafe_allow_html=True,
    )

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Matplotlib",
        "Seaborn",
        "Joblib",
        "Streamlit",
    ]

    st.markdown(
        "".join(
            f'<span class="tech-pill">{tech}</span>'
            for tech in technologies
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.warning(
        "The predicted price is an ML estimate based on historical training "
        "data. Actual market prices can differ because of vehicle condition, "
        "local demand and factors not represented in the dataset."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🚘 AutoValue AI &nbsp;•&nbsp;
        Used Car Resale Price Prediction &nbsp;•&nbsp;
        Machine Learning College Project
    </div>
    """,
    unsafe_allow_html=True,
)
