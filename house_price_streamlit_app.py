# house_price_streamlit_app.py  — Linear Regression version

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# 1) Streamlit page config (must be first)
st.set_page_config(page_title="🏠 Paris House Price Predictor", layout="centered")

# 2) Loading trained LinearRegression model
@st.cache_resource
def load_model():
    p = Path("linear_model.pkl")   # saves the trained LR as this
    if not p.exists():
        st.error("Model file 'linear_model.pkl' not found in this folder.")
        st.stop()
    m = joblib.load(p)
    return m

model = load_model()

# Resolves expected feature order from the trained model 
EXPECTED = list(getattr(model, "feature_names_in_", []))

# Fallback ONLY if feature_names_in_ wasn't stored; keep EXACT training order here
if not EXPECTED:
    EXPECTED = [
        'squareMeters','numberOfRooms','house_age','floors',
        'garage','basement','attic','hasPool_Yes','hasYard_Yes',
        'hasGuestRoom','hasStorageRoom_Yes','isNewBuilt_Yes',
        'hasStormProtector_Yes','cityPartRange','numPrevOwners',
        'neighborhoodClass_Low','neighborhoodClass_Medium'
    ]

# 3) UI
st.title("🏠 Paris House Price Prediction App")
st.markdown("Enter the property details below to get an estimated price.")

col1, col2 = st.columns(2)
with col1:
    squareMeters = st.slider("Size (m²)", 10, 1000, 150)
    numberOfRooms = st.slider("Number of Rooms", 1, 20, 3)
    house_age = st.slider("House Age (years)", 0, 100, 10)
    floors = st.slider("Number of Floors", 1, 5, 1)
    cityPartRange = st.slider("City Part Range (1–10)", 1, 10, 5)
    numPrevOwners = st.slider("Previous Owners", 0, 10, 1)

with col2:
    garage = st.selectbox("Garage", ["No", "Yes"])
    basement = st.selectbox("Basement", ["No", "Yes"])
    attic = st.selectbox("Attic", ["No", "Yes"])
    hasPool = st.selectbox("Has Pool", ["No", "Yes"])
    hasYard = st.selectbox("Has Yard", ["No", "Yes"])
    hasGuestRoom = st.selectbox("Has Guest Room", ["No", "Yes"])
    hasStorageRoom = st.selectbox("Has Storage Room", ["No", "Yes"])
    isNewBuilt = st.selectbox("Newly Built", ["No", "Yes"])
    hasStormProtector = st.selectbox("Storm Protector", ["No", "Yes"])
    neighborhoodClass = st.selectbox("Neighborhood Class", ["Low", "Medium", "High"])

# 4) Building input to match the training encodings
raw = {
    'squareMeters': squareMeters,
    'numberOfRooms': numberOfRooms,
    'house_age': house_age,
    'floors': floors,
    'garage': 1 if garage == "Yes" else 0,
    'basement': 1 if basement == "Yes" else 0,
    'attic': 1 if attic == "Yes" else 0,
    'hasPool_Yes': 1 if hasPool == "Yes" else 0,
    'hasYard_Yes': 1 if hasYard == "Yes" else 0,
    'hasGuestRoom': 1 if hasGuestRoom == "Yes" else 0,
    'hasStorageRoom_Yes': 1 if hasStorageRoom == "Yes" else 0,
    'isNewBuilt_Yes': 1 if isNewBuilt == "Yes" else 0,
    'hasStormProtector_Yes': 1 if hasStormProtector == "Yes" else 0,
    'cityPartRange': cityPartRange,
    'numPrevOwners': numPrevOwners,
    'neighborhoodClass_Low': 1 if neighborhoodClass == "Low" else 0,
    'neighborhoodClass_Medium': 1 if neighborhoodClass == "Medium" else 0,
}
# Ensuring every expected column exists; order exactly
for f in EXPECTED:
    raw.setdefault(f, 0)
X_input = pd.DataFrame([[raw[f] for f in EXPECTED]], columns=EXPECTED)

# Optional debugging
with st.expander("Debug: model feature order & input row"):
    st.write(EXPECTED)
    st.dataframe(X_input)

# 5) Predict
if st.button("Predict Price"):
    yhat = model.predict(X_input)[0]
    st.success(f"💰 Estimated Price: **€{yhat:,.2f}**")
