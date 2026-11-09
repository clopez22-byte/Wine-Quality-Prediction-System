import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("wine_quality.csv")

st.title("🍷 Wine Quality Prediction System")
st.caption(
    "Section B classroom prototype for preliminary red-wine quality classification. "
    "Predictions are screening aids, not final quality-control decisions."
)

try:
    model = load_model()
    data = load_dataset()
except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error}. "
        "Run `python train_model.py` in this project folder first."
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV containing the same predictor columns used during training.")

required_columns = list(model.feature_names_in_)
template = data.drop(columns=["quality"]).head(5)

st.download_button(
    "Download sample input CSV",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="sample_wine_input.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [column for column in required_columns if column not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            # Ignore extra columns and use only the required predictor columns.
            prediction_input = new_data[required_columns]
            predictions = model.predict(prediction_input)

            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                for value in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric(
                "Predicted good-quality samples",
                int((predictions == 1).sum())
            )

            st.download_button(
                "Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="wine_quality_predictions.csv",
                mime="text/csv"
            )
    except Exception as error:
        st.error(f"Unable to process this CSV file: {error}")
