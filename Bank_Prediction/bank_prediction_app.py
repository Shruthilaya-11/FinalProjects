import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Shruthilaya\GUVI\data\bank_prediction\train.csv")
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    return df

data = load_data()
st.title("Bank Term Deposit Prediction")
st.write("Predict if a client will subscribe to a term deposit.")

X = data.drop("y", axis=1)
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

st.sidebar.header("Select Model")
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_choice]

clf = Pipeline(steps=[("preprocessor", preprocessor),
                      ("classifier", model)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"**Selected Model:** {model_choice}")
st.write(f"**Accuracy:** {acc:.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.subheader("Try a Prediction")
user_input = {}
for col in X.columns:
    if col in numeric_features:
        val = st.number_input(f"{col}", value=float(X[col].median()))
    else:
        val = st.selectbox(f"{col}", options=X[col].unique())
    user_input[col] = val

if st.button("Predict"):
    user_df = pd.DataFrame([user_input])
    pred = clf.predict(user_df)[0]
    prob = clf.predict_proba(user_df)[0][1]
    if pred == 1:
        st.success(f"Client is **likely to subscribe** (Probability: {prob:.2f})")
    else:
        st.error(f"Client is **not likely to subscribe** (Probability: {prob:.2f})")