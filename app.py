"""
Malaysia Housing Price Estimator — Streamlit prototype

Deployment prototype for Objective 3: lets a user pick location, tenure(s),
property type(s), and transaction volume, and get an estimated Median_Price
from the best-performing model trained in the notebook.

Run with:
    streamlit run app.py

Requires, in the same folder:
    final_housing_price_model.pkl   (saved in notebook Part 14)
    model_metadata.json             (saved in notebook Part 14.1)
"""

import json
from pathlib import Path

import _pickle as cpickle
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


# ----------------------------------------------------------------------
# Must match the class defined in the notebook (Part 5.4) exactly — the
# pickled pipeline references it by name when unpickling.
# ----------------------------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.maps_ = {c: X[c].value_counts(normalize=True) for c in self.cols}
        return self

    def transform(self, X):
        X = X[self.cols].copy()
        for c in self.cols:
            X[c] = X[c].map(self.maps_[c]).fillna(0.0)
        return X.values

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Malaysia Housing Price Estimator",
    page_icon="🏠",
    layout="centered",
)

MODEL_PATH = Path("final_housing_price_model.pkl")
METADATA_PATH = Path("model_metadata.json")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return cpickle.load(f)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Missing `final_housing_price_model.pkl` and/or `model_metadata.json`. "
        "Run the notebook (through Part 14.1) first, then copy both files into "
        "this app's folder."
    )
    st.stop()

model = load_model()
meta = load_metadata()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏠 Malaysia Housing Price Estimator")
st.caption(
    "Estimate a township-level median house price from location, tenure, "
    "property type, and transaction volume — trained on 2025 Malaysian "
    "housing transaction data."
)

with st.expander("ℹ️ About this model", expanded=False):
    metrics = meta["test_metrics"]
    st.markdown(
        f"""
This is a **{meta['best_model_name']}** model, selected as the
best-performing of four candidates (Linear Regression, Decision Tree,
Random Forest, Gradient Boosting) compared on held-out test data.

| Metric | Value |
|---|---|
| Test RMSE | RM {metrics['Test RMSE']:,.0f} |
| Test MAE | RM {metrics['Test MAE']:,.0f} |
| Test R² | {metrics['Test R2']:.3f} |
| Test MAPE | {metrics['Test MAPE']:.1%} |

**This is not a substitute for professional valuation.** It's a
data-driven reference point — treat the estimate as a ballpark figure,
not a guaranteed price.
"""
    )

st.divider()

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Property details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", options=meta["states"])

with col2:
    areas_for_state = meta["areas_by_state"].get(state, [])
    area = st.selectbox("Area / Township region", options=areas_for_state)

tenures_selected = st.multiselect(
    "Tenure",
    options=meta["all_tenures"],
    default=[meta["all_tenures"][0]],
    help="Select more than one if the township mixes tenure types "
    "(e.g. both Freehold and Leasehold units).",
)

types_selected = st.multiselect(
    "Property type",
    options=meta["all_types"],
    default=[meta["all_types"][0]],
    help="Select more than one if the township mixes property types "
    "(e.g. Terrace House and Semi-D).",
)

transactions = st.slider(
    "Number of transactions (transaction volume)",
    min_value=int(meta["transactions_min"]),
    max_value=int(meta["transactions_max"]),
    value=int(meta["transactions_median"]),
    help="Higher transaction volume generally signals a more active, "
    "established market.",
)

st.divider()

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
predict_clicked = st.button("Estimate price", type="primary", use_container_width=True)


def build_input_row(state, area, tenures, types, transactions, meta):
    row = {
        "State": state,
        "Area": area,
        "Transactions": transactions,
        "n_types": len(types),
        "is_aggregated_type": int(len(types) > 1),
        "n_tenure": len(tenures),
        "is_aggregated_tenure": int(len(tenures) > 1),
    }
    for t in meta["all_types"]:
        row[f"Type_{t}"] = int(t in types)
    for t in meta["all_tenures"]:
        row[f"Tenure_{t}"] = int(t in tenures)
    return pd.DataFrame([row])[meta["feature_cols"]]


if predict_clicked:
    if not tenures_selected or not types_selected:
        st.warning("Please select at least one tenure and one property type.")
    else:
        input_df = build_input_row(
            state, area, tenures_selected, types_selected, transactions, meta
        )
        prediction = model.predict(input_df)[0]

        st.metric("Estimated median price", f"RM {prediction:,.0f}")

        rmse = meta["test_metrics"]["Test RMSE"]
        low, high = max(0, prediction - rmse), prediction + rmse
        st.caption(
            f"Typical model error (±1 RMSE) puts the likely range at "
            f"roughly **RM {low:,.0f} – RM {high:,.0f}**."
        )

        with st.expander("Show model input row"):
            st.dataframe(input_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Prototype built for BMDS2003 Data Science — Malaysia Housing Prices project. "
    "Not a substitute for professional property valuation."
)
