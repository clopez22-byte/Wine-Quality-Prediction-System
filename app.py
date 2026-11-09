
import streamlit as st
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Prediction System")

st.caption(
    "Prototype for preliminary red-wine quality classification. "
    "Predictions are screening aids, not final quality-control decisions."
)


# Load dataset
@st.cache_data
def load_dataset():
    data = pd.read_csv("wine_quality.csv")
    return data


# Train model directly from the dataset
@st.cache_resource
def load_model():
    data = pd.read_csv("wine_quality.csv")
    data = data.drop_duplicates().copy()

    # Define prediction target
    y = (data["quality"] >= 6).astype(int)

    # Remove target from input features
    X = data.drop(columns=["quality"])

    # Create Random Forest pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Train using the available dataset
    model.fit(X, y)

    return model


# Load application data
try:
    data = load_dataset()
    model = load_model()

except Exception as error:
    st.error(f"Unable to load the dataset or train the model: {error}")
    st.stop()


# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Wine Records", len(data))

col2.metric(
    "Average Quality Score",
    round(data["quality"].mean(), 2)
)

col3.metric(
    "Good Quality Records",
    int((data["quality"] >= 6).sum())
)


# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)


# CSV upload section
st.subheader("Wine Quality Prediction")

st.write(
    "Upload a CSV file containing the 11 wine measurement columns."
)

required_columns = list(model.feature_names_in_)

sample_data = data.drop(columns=["quality"]).head(5)

st.download_button(
    label="Download Sample Input CSV",
    data=sample_data.to_csv(index=False).encode("utf-8"),
    file_name="sample_wine_input.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Upload Wine CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        new_data = pd.read_csv(uploaded_file)

        missing_columns = [
            column for column in required_columns
            if column not in new_data.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            prediction_input = new_data[required_columns]

            predictions = model.predict(prediction_input)

            results = new_data.copy()

            results["Prediction"] = [
                "GOOD QUALITY" if prediction == 1
                else "NEEDS IMPROVEMENT"
                for prediction in predictions
            ]

            st.subheader("Prediction Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            good_count = int((predictions == 1).sum())

            st.metric(
                "Predicted Good Quality Samples",
                good_count
            )

            st.download_button(
                label="Download Prediction Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="wine_quality_predictions.csv",
                mime="text/csv"
            )

    except Exception as error:

        st.error(f"Unable to process uploaded CSV: {error}")
